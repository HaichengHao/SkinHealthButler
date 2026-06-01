import os
import sys
import json
import re
import tempfile
import uuid
from pathlib import Path
from threading import Thread

import torch
from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_community.chat_message_histories import RedisChatMessageHistory
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

load_dotenv()

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = None
for p in CURRENT.parents:
    if (p / "model_").exists():
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = CURRENT.parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

chat_rt = APIRouter(prefix="/chat", tags=["chat"])

SYSTEM_PROMPT = """
你是一名专业的皮肤健康助手。
你的职责：
1. 根据用户描述的症状分析可能的皮肤问题
2. 给出基础护理建议
3. 提醒用户何时需要及时就医
4. 与皮肤疾病无关的问题需要委婉拒绝
注意：
1. 你不能替代专业医生诊断
2. 不要编造医学结论
3. 严重情况必须建议线下医院就诊
4. 回复尽量专业、简洁、清晰
"""

MODEL_PATH = os.getenv("MODEL_PATH")
REDIS_URL = os.getenv("REDIS_URL")
SKIN_CLS_EXPORT_DIR = os.getenv("SKIN_CLS_EXPORT_DIR", "model_/exports/swinv2")

if not MODEL_PATH:
    raise ValueError("MODEL_PATH 未配置")
if not REDIS_URL:
    raise ValueError("REDIS_URL 未配置")

_tokenizer = None
_model = None
_classifier = None


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def get_session_history(session_id: str):
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL, ttl=1200)


def build_messages(history_messages, user_message: str, cls_hint: str | None = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if cls_hint:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"补充信息：图像分类模型预测最可能疾病标签为 `{cls_hint}`。\n"
                    "回答要求：\n"
                    "1) 先围绕该标签给出针对性分析与护理建议，再结合用户文字描述补充。\n"
                    "2) 明确说明这是图像模型预测结果，仅供参考，不能替代医生诊断。\n"
                    "3) 若用户描述与该标签明显冲突，请指出不一致并建议线下皮肤科检查。"
                ),
            }
        )

    for msg in history_messages[-6:]:
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})

    if cls_hint:
        user_message = (
            f"【图像分类参考标签】{cls_hint}\n"
            f"【用户描述】{user_message}\n"
            "请基于参考标签优先回答，并给出简洁可执行建议。"
        )
    messages.append({"role": "user", "content": user_message})
    return messages


def get_llm():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )
        _model.eval()
        logger.info("LLM loaded for FastAPI chat route")
    return _tokenizer, _model


def get_classifier():
    global _classifier
    if _classifier is None:
        from model_.skin_classifier import SkinDiseaseClassifier

        _classifier = SkinDiseaseClassifier.from_export_dir(SKIN_CLS_EXPORT_DIR, prefer_onnx=True)
        logger.info(f"Skin classifier loaded from: {SKIN_CLS_EXPORT_DIR}")
    return _classifier


def run_llm_reply(messages: list[dict]) -> str:
    tokenizer, model = get_llm()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    new_tokens = output[0][model_inputs["input_ids"].shape[1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    reply = strip_think_tags(reply)
    return reply


def _classify_optional_image(image: UploadFile | None):
    cls_result = None
    image_label = None
    image_conf = None
    if image is None:
        return cls_result, image_label, image_conf

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/bmp")

    temp_path = Path(tempfile.gettempdir()) / f"skin_{uuid.uuid4().hex}{suffix}"
    return cls_result, image_label, image_conf, temp_path


@chat_rt.post("/respond")
async def respond(
    message: str = Form(...),
    session_id: str = Form(default="default-session"),
    image: UploadFile | None = File(default=None),
):
    if not message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    cls_result = None
    image_label = None
    image_conf = None
    temp_path = None
    if image is not None:
        _, _, _, temp_path = _classify_optional_image(image)
        content = await image.read()
        temp_path.write_bytes(content)
        try:
            cls_result = get_classifier().predict(str(temp_path), topk=3)
            image_label = cls_result.top1_label
            image_conf = cls_result.top1_confidence
        finally:
            temp_path.unlink(missing_ok=True)

    redis_history = get_session_history(session_id)
    history_messages = redis_history.messages
    messages = build_messages(history_messages, message, cls_hint=image_label)
    reply = run_llm_reply(messages)

    redis_history.add_user_message(message)
    redis_history.add_ai_message(reply)

    return {
        "session_id": session_id,
        "reply": reply,
        "classification": None
        if cls_result is None
        else {
            "label": image_label,
            "confidence": image_conf,
            "topk": cls_result.topk,
        },
    }


@chat_rt.post("/respond_stream")
async def respond_stream(
    message: str = Form(...),
    session_id: str = Form(default="default-session"),
    image: UploadFile | None = File(default=None),
):
    if not message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    cls_result = None
    image_label = None
    image_conf = None
    temp_path = None
    if image is not None:
        _, _, _, temp_path = _classify_optional_image(image)
        content = await image.read()
        temp_path.write_bytes(content)
        try:
            cls_result = get_classifier().predict(str(temp_path), topk=3)
            image_label = cls_result.top1_label
            image_conf = cls_result.top1_confidence
        finally:
            temp_path.unlink(missing_ok=True)

    redis_history = get_session_history(session_id)
    history_messages = redis_history.messages
    messages = build_messages(history_messages, message, cls_hint=image_label)

    tokenizer, model = get_llm()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def event_generator():
        full_answer = ""
        classification = None
        if cls_result is not None:
            classification = {
                "label": image_label,
                "confidence": image_conf,
                "topk": cls_result.topk,
            }
        yield f"data: {json.dumps({'type': 'meta', 'classification': classification}, ensure_ascii=False)}\n\n"

        in_thinking = False
        tag_buffer = ""
        for token in streamer:
            tag_buffer += token
            if not in_thinking and "<think>" in tag_buffer:
                in_thinking = True
                tag_buffer = tag_buffer.split("<think>", 1)[1]
                continue
            if in_thinking and "</think>" in tag_buffer:
                in_thinking = False
                tag_buffer = tag_buffer.split("</think>", 1)[1]
                continue
            if in_thinking:
                continue
            if tag_buffer:
                cleaned = strip_think_tags(tag_buffer)
                if not cleaned:
                    tag_buffer = ""
                    continue
                full_answer += cleaned
                payload = {"type": "chunk", "delta": cleaned}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                tag_buffer = ""

        redis_history.add_user_message(message)
        redis_history.add_ai_message(strip_think_tags(full_answer))
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
