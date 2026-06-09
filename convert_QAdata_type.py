# @Time    : 2026/6/8 16:21
# @Author  : hero
# @File    : convert_QAdata_type.py
import json
from loguru import logger


def main():
    input_path = "./SkinDiseaseQAs_supp.json"
    output_path = "./SkinDiseasesQAs_formated.jsonl"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    valid = 0
    skipped = 0

    with open(output_path, "a+", encoding="utf-8") as f:
        for idx, conversation in enumerate(data, start=1):
            total += 1

            messages = conversation.get("messages")

            if not isinstance(messages, list) or len(messages) == 0:
                skipped += 1
                logger.warning(f"第 {idx} 条缺少 messages，已跳过")
                continue

            # 只保留 messages 字段
            new_item = {
                "messages": messages
            }

            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            valid += 1

    logger.success(f"转换完成！输出文件: {output_path}")
    logger.info(f"总样本数: {total}")
    logger.info(f"有效样本数: {valid}")
    logger.info(f"跳过样本数: {skipped}")


if __name__ == "__main__":
    main()