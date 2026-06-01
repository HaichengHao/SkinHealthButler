# @Time    : 2026/5/22 17:45
# @Author  : hero
# @File    : demo3.py
import os
import time
import asyncio
from threading import Thread

import torch
import gradio as gr
from dotenv import load_dotenv
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from langchain_community.chat_message_histories import RedisChatMessageHistory

# =========================================================
# 初始化
# =========================================================

load_dotenv()

start_time = time.time()

MODEL_PATH = os.getenv("MODEL_PATH")
REDIS_URL = os.getenv("REDIS_URL")

if MODEL_PATH is None:
    raise ValueError("MODEL_PATH 未配置")

if REDIS_URL is None:
    raise ValueError("REDIS_URL 未配置")

# =========================================================
# 加载 tokenizer
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)

# =========================================================
# 加载模型
# =========================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)

model.eval()

logger.info("模型加载完成")

# =========================================================
# 系统提示词
# =========================================================

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

# =========================================================
# Redis 历史记录
# =========================================================

def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        ttl=1200
    )

# =========================================================
# 构建聊天消息
# =========================================================

def build_messages(history_messages, user_message):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # 只保留最近 6 条历史
    recent_messages = history_messages[-6:]

    for msg in recent_messages:

        if msg.type == "human":
            role = "user"
        else:
            role = "assistant"

        messages.append({
            "role": role,
            "content": msg.content
        })

    messages.append({
        "role": "user",
        "content": user_message
    })

    return messages

# =========================================================
# 真正流式推理 + think折叠
# =========================================================

async def predict_async(message, history, request: gr.Request):

    session_id = request.session_hash

    redis_history = get_session_history(session_id)

    history_messages = redis_history.messages

    messages = build_messages(history_messages, message)

    # Qwen Chat Template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    # streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

    # 启动生成线程
    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )

    thread.start()

    # =========================================================
    # thinking状态管理
    # =========================================================

    think_buffer = ""
    answer_buffer = ""

    in_thinking = False

    # 用于处理跨token标签
    tag_buffer = ""

    # =========================================================
    # 流式读取
    # =========================================================

    for new_text in streamer:

        tag_buffer += new_text

        # -------------------------------------------------
        # 检测 think 开始
        # -------------------------------------------------

        if not in_thinking and "<think>" in tag_buffer:

            in_thinking = True

            tag_buffer = tag_buffer.split("<think>", 1)[1]

            continue

        # -------------------------------------------------
        # 检测 think 结束
        # -------------------------------------------------

        if in_thinking and "</think>" in tag_buffer:

            in_thinking = False

            after_think = tag_buffer.split("</think>", 1)[1]

            answer_buffer += after_think

            tag_buffer = ""

            outputs = []

            if think_buffer.strip():

                outputs.append(
                    {
                        "role": "assistant",
                        "content": think_buffer,
                        "metadata": {
                            "title": "🧠 思考过程"
                        }
                    }
                )

            outputs.append(
                {
                    "role": "assistant",
                    "content": answer_buffer
                }
            )

            yield outputs

            continue

        # -------------------------------------------------
        # 思考阶段
        # -------------------------------------------------

        if in_thinking:

            think_buffer += new_text

            yield [
                {
                    "role": "assistant",
                    "content": think_buffer,
                    "metadata": {
                        "title": "🧠 思考过程"
                    }
                }
            ]

        # -------------------------------------------------
        # 正常回答阶段
        # -------------------------------------------------

        else:

            answer_buffer += new_text

            outputs = []

            if think_buffer.strip():

                outputs.append(
                    {
                        "role": "assistant",
                        "content": think_buffer,
                        "metadata": {
                            "title": "🧠 思考过程"
                        }
                    }
                )

            outputs.append(
                {
                    "role": "assistant",
                    "content": answer_buffer
                }
            )

            yield outputs

        await asyncio.sleep(0.01)

    # =========================================================
    # 保存历史
    # 只保存最终答案
    # =========================================================

    redis_history.add_user_message(message)

    redis_history.add_ai_message(answer_buffer)

# =========================================================
# Chatbot
# 必须 type="messages"
# =========================================================

chatbot = gr.Chatbot(
    height=650,
    render_markdown=True,
)

# =========================================================
# Gradio 界面
# =========================================================

demo = gr.ChatInterface(
    fn=predict_async,
    chatbot=chatbot,
    title="肤康管家",
    description="你的个人皮肤健康助手",
    examples=[
        "荨麻疹有哪些症状？",
        "脸上长了很多粉刺和红疙瘩怎么办？",
        "手上长小水疱而且很痒是什么情况？",
        "皮肤突然大片发红并且脱皮怎么办？"
    ],
    multimodal=False,
    autofocus=True,
)

# =========================================================
# 启动
# =========================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=6466,
        share=False,
        debug=True,
        theme=gr.themes.Soft(),

    )

    finished_load_time = time.time()

    total_load_time = finished_load_time - start_time

    logger.info(f"Total load time: {total_load_time:.2f}s")