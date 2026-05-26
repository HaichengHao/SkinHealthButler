# @Time    : 2026/5/22 13:33
# @Author  : hero
# @File    : convert_data_type.py
import json
from loguru import logger
# 读取原始的 JSON 数组文件
def main():
    with open('SkinDiseaseQA.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将每个对话对象写入 JSONL 文件的新行
    with open('SkinDiseaseQA_formated.jsonl', 'w', encoding='utf-8') as f:
        for conversation in data:
            # 确保只写入包含 "messages" 的有效对话
            if "messages" in conversation:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
    logger.success('转换完成！输出文件: skindieastQA.jsonl')
