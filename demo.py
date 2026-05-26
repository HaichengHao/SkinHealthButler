# @Time    : 2026/5/22 15:23
# @Author  : hero
# @File    : demo.py
import time
import torch
import os
import asyncio

import gradio as gr
from loguru import logger

from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from operator import itemgetter


load_dotenv()

start_time=time.time()
# 1. 设置模型路径（指向 full_params 目录）
model_path = os.getenv('MODEL_PATH')

# 2. 加载 tokenizer（Qwen 需要 trust_remote_code=True）
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False  # Qwen 建议设为 False
)

# 3. 加载模型（Qwen3 必须 trust_remote_code=True）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:0" if torch.cuda.is_available() else 'cpu',          # 自动使用 GPU（若可用）
    dtype=torch.float16, # Qwen3 推荐 bfloat16；若显卡不支持（如 30/40 系列），改用 torch.float16
    trust_remote_code=True,
    # tokenizer_kwargs={"use_default_system_prompt": False},

    # load_in_4bit=True,       # 如显存不足（<12GB），取消注释启用 4-bit 量化（需安装 bitsandbytes）
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

hf_llm = HuggingFacePipeline(pipeline=pipe)
# llm_local = ChatHuggingFace(llm=hf_llm)  # ✅ 正确方式
# llm_local = ChatHuggingFace(
#     model=model,
#     tokenizer=tokenizer,
# )


prompt = ChatPromptTemplate(
    messages=[
        ('system','你现在是一名皮肤病诊断高手,可以根据用户输入的症状分析出可能的病因，判别皮肤病,并给出合适的建议,并可以根据历史会话多轮对话，与皮肤疾病判别和治疗无关的你需要委婉拒绝'),
        MessagesPlaceholder(variable_name='history'),
        ('user','{user_quiz}')
    ]
)


chain_with_context = (
    {
        'user_quiz':itemgetter('user_quiz'),
        'history':itemgetter('history')
    }
    | prompt
    | hf_llm
    | StrOutputParser()

)

def get_session_history(session_id):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.getenv('REDIS_URL'),
        ttl=1200
    )

chain_with_history = RunnableWithMessageHistory(
    chain_with_context,
    get_session_history,
    input_messages_key='user_quiz',
    history_messages_key='history',
)


async def predict_async(message,history,request:gr.Request):
    session_id = request.session_hash
    config_runnable = RunnableConfig(
        configurable={
            'session_id':session_id
        }
    )
    full_resp=''
    async for chunk in chain_with_history.astream(
        input={
            'user_quiz':message
        },
        config=config_runnable,
    ):
        full_resp+=chunk
        await asyncio.sleep(0.01)
        yield full_resp


if __name__ == '__main__':
    demo=gr.ChatInterface(
        fn=predict_async,
        title='肤康管家',
        description="你的个人皮肤管家",
        examples=[
            "荨麻疹的症状表现",
            "脸上长了很多粉刺和红疙瘩，怎么办?",
            "我手上长了小水疱，很痒，是什么病？"
        ],
        multimodal=False,
        autofocus=True
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=6466,
        share=False,
        debug=True,
        theme=gr.themes.Soft()
    )

    finished_load_time = time.time()

    total_load_time = (finished_load_time - start_time)
    logger.info(f"Total load time: {total_load_time}")