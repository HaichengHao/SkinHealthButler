# SkinHealthButler 


![](/imgs/icon.png)

------------------------------


- 项目背景: 
> 受季节以及身体内在，外在因素影像，皮肤问题在日常生活中是十分广泛且常见的，  
> 但是皮肤病整体上的临床表现又难以区分,所以采用本项目(结合CV+NLP)的方式，  
> 对一些常见皮肤病以及更多通过大模型蒸馏的皮肤病问答对来实现可以判别皮肤病的个人皮肤健康管家



![](/imgs/poster.png)
- 项目实现目标
> 开发一个基于DeepAgent的皮肤健康管理顾问系统

- 功能  

| 功能       | 介绍                           |    
|----------|------------------------------|  
| 皮肤图片上传识别 | 上传皮肤不适区域图片(注意隐私),将会进行皮肤病分类判别 |   
| 皮肤问题问答   | 经过与微调的模型对话,你将逐步确定皮肤疾病        |



--------------------------------

##  蒸馏问答对 (基于Qwen3-MAX,豆包,ChatGPT生成的高质量问答对)    

- 总条数: 1044条

### `json`转`jsonl`脚本
[格式转换脚本](/convert_data_type.py)
```python
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

```
### 原始`json`格式的部分问答对预览
[原始json格式问答对](/SkinDiseaseQA.json)
```json 
{
    "messages": [
      {
        "role": "user",
        "content": "痤疮的病因、临床表现、诊断要点及治疗建议"
      },
      {
        "role": "assistant",
        "content": "痤疮是一种毛囊皮脂腺单位的慢性炎症性疾病。\n\n**病因**：核心机制包括雄激素介导的皮脂腺过度分泌、毛囊皮脂腺导管角化异常导致堵塞、痤疮丙酸杆菌（Cutibacterium acnes）的定植与增殖，以及由此引发的先天性和适应性免疫炎症反应。遗传、高血糖指数饮食、乳制品摄入、压力及某些化妆品也可能诱发或加重病情。\n\n**临床表现**：好发于面部、胸背部。皮损类型多样，包括非炎症性皮损（开放性粉刺/黑头、闭合性粉刺/白头）和炎症性皮损（红色丘疹、脓疱、结节、囊肿）。严重者可形成窦道和瘢痕（萎缩性或增生性）。\n\n**诊断要点**：主要依据典型的临床表现和发病年龄（青春期为主）。通常无需实验室检查。需与玫瑰痤疮（无粉刺，好发于面中部）、马拉色菌毛囊炎（胸背孤立性毛囊性丘疹脓疱）、药物性痤疮（突然发作，以均一性丘疹脓疱为主，无粉刺）等鉴别。\n\n**治疗建议**：遵循分级、个体化原则。\n1. **轻度（以粉刺为主）**：外用维A酸类药物（如阿达帕林凝胶），每晚一次。\n2. **中度（伴炎性丘疹、脓疱）**：联合外用过氧化苯甲酰（BPO）和/或外用抗生素（如克林霉素磷酸酯凝胶）。BPO可与维A酸早晚交替使用。\n3. **重度（伴结节、囊肿）**：系统治疗是关键。首选口服异维A酸，需严格避孕。女性患者可考虑抗雄激素治疗（如短效避孕药、螺内酯）。口服抗生素（多西环素、米诺环素）可用于炎症控制，但不作为长期单药治疗。\n4. **辅助治疗**：化学换肤（果酸、水杨酸）、光动力疗法（PDT）可用于特定类型。日常需温和清洁，避免挤压，并注意防晒。"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "光化性角化病的病因、临床表现、诊断要点及治疗建议"
      },
      {
        "role": "assistant",
        "content": "光化性角化病（AK）是一种由紫外线损伤引起的表皮内鳞状细胞癌前病变。\n\n**病因**：最主要的病因是长期、累积的日光（紫外线）暴露。好发于光暴露部位，如面部、耳廓、秃发头皮、手背和前臂伸侧。浅肤色、老年、免疫抑制状态是重要危险因素。\n\n**临床表现**：皮损为单发或多发的、边界不清的红斑，表面粗糙、干燥、呈砂纸样或疣状增厚，可有鳞屑或结痂。触诊比视诊更易发现。部分皮损可自行消退，但约5-10%可能进展为侵袭性鳞状细胞癌（SCC）。\n\n**诊断要点**：主要基于临床表现和病史（长期日晒史）。皮肤镜检查可见“草莓征”（红-白相间的背景上分布着血管球）。对于不典型、增厚、出血或溃疡的皮损，必须进行皮肤活检以排除已恶变的SCC。需与脂溢性角化病、盘状红斑狼疮、Bowen病（原位SCC）鉴别。\n\n**治疗建议**：治疗目标是清除可见病灶并预防癌变。\n1. **局部药物治疗（适用于多发、泛发皮损）**：5-氟尿嘧啶（5-FU）乳膏、咪喹莫特乳膏、双氯芬酸钠凝胶或Ingenol mebutate凝胶。疗程数天至数周，会引起局部炎症反应。\n2. **物理治疗（适用于孤立、局限皮损）**：液氮冷冻治疗是最常用方法，简便有效。\n3. **光动力疗法（PDT）**：对大面积、多发损害效果好，美容效果佳。\n4. **预防**：最根本的措施是严格防晒，穿戴宽檐帽、长袖衣物，并每日使用SPF≥30的广谱防晒霜。"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "良性肿瘤的病因、临床表现、诊断要点及治疗建议"
      },
      {
        "role": "assistant",
        "content": "皮肤良性肿瘤是一大类非癌性皮肤增生的总称，病因和表现各异。\n\n**病因**：因具体类型而异。例如，脂溢性角化病可能与年龄、遗传和日晒有关；皮肤纤维瘤常继发于轻微外伤或虫咬；表皮囊肿源于毛囊漏斗部阻塞；樱桃状血管瘤与年龄增长相关。\n\n**临床表现**：形态多样，可表现为斑块、丘疹、结节或肿物。颜色从肤色、褐色到红色、蓝色不等。生长缓慢，质地柔软或坚实，通常无自觉症状。\n\n**诊断要点**：诊断高度依赖于临床经验结合皮肤镜检查。皮肤镜能提供特征性模式（如脂溢性角化病的脑回样结构、指纹样嵴、粟粒样囊肿；樱桃状血管瘤的均质红/蓝无结构区）。对于任何快速增大、出血、溃疡、颜色不均或边界不清的“良性”肿瘤，都应高度警惕，并行皮肤活检以明确诊断，排除恶性肿瘤。\n\n**治疗建议**：\n1. **观察**：对于诊断明确、无症状、不影响美观的良性肿瘤，无需治疗，定期随访即可。\n2. **手术切除**：适用于有恶变疑虑、影响美观、产生不适（摩擦、瘙痒）或患者强烈要求去除的情况。切除标本应送病理检查以最终确诊。\n3. **其他方法**：激光（CO2、脉冲染料激光）、电灼、刮除术等可用于特定类型的良性肿瘤（如脂溢性角化病、樱桃状血管瘤），但无法提供组织学诊断。"
      }
    ]
  },
```
### 转换为`.jsonl`后的部分问答对预览
[转换后的json lines格式](/SkinDiseaseQA_formated.jsonl)
```json lines
{"messages": [{"role": "user", "content": "痤疮的病因、临床表现、诊断要点及治疗建议"}, {"role": "assistant", "content": "痤疮是一种毛囊皮脂腺单位的慢性炎症性疾病。\n\n**病因**：核心机制包括雄激素介导的皮脂腺过度分泌、毛囊皮脂腺导管角化异常导致堵塞、痤疮丙酸杆菌（Cutibacterium acnes）的定植与增殖，以及由此引发的先天性和适应性免疫炎症反应。遗传、高血糖指数饮食、乳制品摄入、压力及某些化妆品也可能诱发或加重病情。\n\n**临床表现**：好发于面部、胸背部。皮损类型多样，包括非炎症性皮损（开放性粉刺/黑头、闭合性粉刺/白头）和炎症性皮损（红色丘疹、脓疱、结节、囊肿）。严重者可形成窦道和瘢痕（萎缩性或增生性）。\n\n**诊断要点**：主要依据典型的临床表现和发病年龄（青春期为主）。通常无需实验室检查。需与玫瑰痤疮（无粉刺，好发于面中部）、马拉色菌毛囊炎（胸背孤立性毛囊性丘疹脓疱）、药物性痤疮（突然发作，以均一性丘疹脓疱为主，无粉刺）等鉴别。\n\n**治疗建议**：遵循分级、个体化原则。\n1. **轻度（以粉刺为主）**：外用维A酸类药物（如阿达帕林凝胶），每晚一次。\n2. **中度（伴炎性丘疹、脓疱）**：联合外用过氧化苯甲酰（BPO）和/或外用抗生素（如克林霉素磷酸酯凝胶）。BPO可与维A酸早晚交替使用。\n3. **重度（伴结节、囊肿）**：系统治疗是关键。首选口服异维A酸，需严格避孕。女性患者可考虑抗雄激素治疗（如短效避孕药、螺内酯）。口服抗生素（多西环素、米诺环素）可用于炎症控制，但不作为长期单药治疗。\n4. **辅助治疗**：化学换肤（果酸、水杨酸）、光动力疗法（PDT）可用于特定类型。日常需温和清洁，避免挤压，并注意防晒。"}]}
{"messages": [{"role": "user", "content": "光化性角化病的病因、临床表现、诊断要点及治疗建议"}, {"role": "assistant", "content": "光化性角化病（AK）是一种由紫外线损伤引起的表皮内鳞状细胞癌前病变。\n\n**病因**：最主要的病因是长期、累积的日光（紫外线）暴露。好发于光暴露部位，如面部、耳廓、秃发头皮、手背和前臂伸侧。浅肤色、老年、免疫抑制状态是重要危险因素。\n\n**临床表现**：皮损为单发或多发的、边界不清的红斑，表面粗糙、干燥、呈砂纸样或疣状增厚，可有鳞屑或结痂。触诊比视诊更易发现。部分皮损可自行消退，但约5-10%可能进展为侵袭性鳞状细胞癌（SCC）。\n\n**诊断要点**：主要基于临床表现和病史（长期日晒史）。皮肤镜检查可见“草莓征”（红-白相间的背景上分布着血管球）。对于不典型、增厚、出血或溃疡的皮损，必须进行皮肤活检以排除已恶变的SCC。需与脂溢性角化病、盘状红斑狼疮、Bowen病（原位SCC）鉴别。\n\n**治疗建议**：治疗目标是清除可见病灶并预防癌变。\n1. **局部药物治疗（适用于多发、泛发皮损）**：5-氟尿嘧啶（5-FU）乳膏、咪喹莫特乳膏、双氯芬酸钠凝胶或Ingenol mebutate凝胶。疗程数天至数周，会引起局部炎症反应。\n2. **物理治疗（适用于孤立、局限皮损）**：液氮冷冻治疗是最常用方法，简便有效。\n3. **光动力疗法（PDT）**：对大面积、多发损害效果好，美容效果佳。\n4. **预防**：最根本的措施是严格防晒，穿戴宽檐帽、长袖衣物，并每日使用SPF≥30的广谱防晒霜。"}]}
{"messages": [{"role": "user", "content": "良性肿瘤的病因、临床表现、诊断要点及治疗建议"}, {"role": "assistant", "content": "皮肤良性肿瘤是一大类非癌性皮肤增生的总称，病因和表现各异。\n\n**病因**：因具体类型而异。例如，脂溢性角化病可能与年龄、遗传和日晒有关；皮肤纤维瘤常继发于轻微外伤或虫咬；表皮囊肿源于毛囊漏斗部阻塞；樱桃状血管瘤与年龄增长相关。\n\n**临床表现**：形态多样，可表现为斑块、丘疹、结节或肿物。颜色从肤色、褐色到红色、蓝色不等。生长缓慢，质地柔软或坚实，通常无自觉症状。\n\n**诊断要点**：诊断高度依赖于临床经验结合皮肤镜检查。皮肤镜能提供特征性模式（如脂溢性角化病的脑回样结构、指纹样嵴、粟粒样囊肿；樱桃状血管瘤的均质红/蓝无结构区）。对于任何快速增大、出血、溃疡、颜色不均或边界不清的“良性”肿瘤，都应高度警惕，并行皮肤活检以明确诊断，排除恶性肿瘤。\n\n**治疗建议**：\n1. **观察**：对于诊断明确、无症状、不影响美观的良性肿瘤，无需治疗，定期随访即可。\n2. **手术切除**：适用于有恶变疑虑、影响美观、产生不适（摩擦、瘙痒）或患者强烈要求去除的情况。切除标本应送病理检查以最终确诊。\n3. **其他方法**：激光（CO2、脉冲染料激光）、电灼、刮除术等可用于特定类型的良性肿瘤（如脂溢性角化病、樱桃状血管瘤），但无法提供组织学诊断。"}]}
```




## 可识别类别(图片上传识别)

| 文件夹名（原始）     | 标准中文病名         | 英文名                              | 简要说明                                             |
| -------------------- | -------------------- | ----------------------------------- | ---------------------------------------------------- |
| Acne                 | 痤疮                 | Acne Vulgaris                       | 毛囊皮脂腺慢性炎症性皮肤病，常见于青春期             |
| Actinic_Keratosis    | 光化性角化病         | Actinic Keratosis (AK)              | 日光损伤引起的癌前病变，好发于曝光部位               |
| Benign_tumors        | 良性肿瘤（泛指）     | Benign Skin Tumors                  | 如脂溢性角化病、表皮囊肿等非恶性增生                 |
| Bullous              | 大疱性皮肤病（泛指） | Bullous Dermatoses                  | 如天疱疮、类天疱疮等以水疱/大疱为特征的疾病          |
| Candidiasis          | 念珠菌病             | Cutaneous Candidiasis               | 由白色念珠菌等引起的皮肤黏膜真菌感染                 |
| DrugEruption         | 药物疹               | Drug Eruption / Drug Rash           | 药物引起的过敏性或毒性皮肤反应                       |
| Eczema               | 湿疹                 | Eczema / Atopic Dermatitis          | 慢性炎症性瘙痒性皮肤病，特应性体质相关               |
| Infestations_Bites   | 寄生虫感染与虫咬皮炎 | Infestations & Insect Bites         | 如疥疮、虱病、蚊虫叮咬等                             |
| Lichen               | 扁平苔藓等           | Lichen Planus / Lichenoid Disorders | 慢性炎症性丘疹鳞屑性皮肤病                           |
| Lupus                | 红斑狼疮（皮肤型）   | Cutaneous Lupus Erythematosus       | 自身免疫病累及皮肤，如盘状红斑狼疮                   |
| Moles                | 痣（色素痣）         | Melanocytic Nevi                    | 黑素细胞良性增生，需鉴别恶性黑色素瘤                 |
| Psoriasis            | 银屑病               | Psoriasis                           | 慢性复发性炎症性皮肤病，典型为银白色鳞屑斑块         |
| Rosacea              | 玫瑰痤疮             | Rosacea                             | 面部中心区域的慢性炎症性血管/毛囊疾病                |
| Seborrheic_Keratosis | 脂溢性角化病         | Seborrheic Keratosis                | 老年性良性表皮增生，“ stuck-on ”外观                 |
| SkinCancer           | 皮肤癌（泛指）       | Skin Cancer                         | 包括基底细胞癌、鳞状细胞癌、黑色素瘤等               |
| Sun_Sunlight_Damage  | 日光性损伤           | Photoaging / Solar Damage           | 紫外线导致的皮肤老化与癌前病变（如雀斑、光化性角化） |
| Tinea                | 癣（真菌感染）       | Dermatophytosis (Tinea)             | 如足癣、体癣、头癣等皮肤癣菌感染                     |
| Unknown_Normal       | 未知/正常皮肤        | Unknown or Normal Skin              | 可能为健康对照或无法归类样本                         |
| Vascular_Tumors      | 血管性肿瘤           | Vascular Tumors                     | 如血管瘤、血管肉瘤等（良性/恶性）                    |
| Vasculitis           | 血管炎               | Cutaneous Vasculitis                | 小血管炎症，可伴紫癜、溃疡等                         |
| Vitiligo             | 白癜风               | Vitiligo                            | 黑素细胞破坏导致的获得性皮肤色素脱失                 |
| Warts                | 疣                   | Warts (Verrucae)                    | HPV感染引起的表皮良性增生（如寻常疣、扁平疣）        |

## 可推理类别(暂不支持图片上传识别)
这里说一下暂不支持的类别，【由于皮肤病临床图片数据有限,故采用可对话判别(仅供参考)】  
除开上述





-------------


## 模型微调

1. 准备好蒸馏的问答对之后进行微调,现修改好dataset_info.json,添加进去问答对的数据信息
记得把数据也放到这个`/LLaMA-Factory/data`文件下面
```shell
nikofox@MOSS:~/llamafactory/LLaMA-Factory/data$ pwd
/home/nikofox/llamafactory/LLaMA-Factory/data
nikofox@MOSS:~/llamafactory/LLaMA-Factory/data$ head -17 dataset_info.json 
{
  "identity": {
    "file_name": "identity.json"
  },
  "SkinDiseaseQA_formated": {
  "file_name": "SkinDiseaseQA_formated.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": { 
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
},

```

2. 启动LLaMA-Factory进行微调操作

- 注意，没有模型可以去HF或者魔搭下载模型
```shell
modelscope download --model Qwen/Qwen3-0.6B --local_dir model/Qwen3-0.6B

nikofox@MOSS:~/llamafactory/LLaMA-Factory/model/Qwen3-0.6B$ pwd
/home/nikofox/llamafactory/LLaMA-Factory/model/Qwen3-0.6B
nikofox@MOSS:~/llamafactory/LLaMA-Factory/model/Qwen3-0.6B$ ls
config.json             LICENSE            README.md              vocab.json
configuration.json      merges.txt         tokenizer_config.json
generation_config.json  model.safetensors  tokenizer.json

```
这样本地就有模型了,点选好对应的参数,开始微调
![](/imgs/01.png)
![](/imgs/03.png)

| 训练损失                         | 验证损失                              |
|------------------------------|-----------------------------------|
| ![](/imgs/training_loss.png) | ![](/imgs/training_eval_loss.png) |


训练完毕之后测试一下对话效果

![](/imgs/02.png)
![](/imgs/05.png)

导出模型

![](/imgs/04.png)


查看日志
![](/imgs/06.png)


```shell
nikofox@MOSS:~/llamafactory/LLaMA-Factory/saves/Custom/lora/SkinHealthButler_V2$ source /home/nikofox/llamafactory/LLaMA-Factory/.venv/bin/activate
(LLaMA-Factory) nikofox@MOSS:~/llamafactory/LLaMA-Factory/saves/Custom/lora/SkinHealthButler_V2$ tensorboard --logdir runs --port 6007
/home/nikofox/llamafactory/LLaMA-Factory/.venv/lib/python3.12/site-packages/tensorboard/default.py:30: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.20.0 at http://localhost:6007/ (Press CTRL+C to quit)

```

![](/imgs/07.png)

------------------------------
## 分类模型构建与ONNX导出(swinv2)

```python
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import timm

SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_classes(train_dir: Path) -> list[str]:
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


class SkinDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Invalid image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def collect_samples(root: Path, cls2idx: dict[str, int]) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for cls, idx in cls2idx.items():
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((str(f), idx))
    return items


class SwinHead(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        image_size: int,
        dropout: float,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_pred = []
    all_true = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        losses.append(loss.item())
        all_pred.extend(pred.detach().cpu().tolist())
        all_true.extend(y.detach().cpu().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return float(np.mean(losses)) if losses else 0.0, float(acc), float(f1)


def export_onnx(model: nn.Module, out_path: Path, image_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="model_/exports/swinv2")
    parser.add_argument("--model-name", type=str, default="swinv2_base_window16_256")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true", default=False)
    args = parser.parse_args()

    seed_everything(SEED)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = list_classes(train_dir)
    cls2idx = {c: i for i, c in enumerate(class_names)}
    print(f"[INFO] classes={len(class_names)}", flush=True)

    train_samples = collect_samples(train_dir, cls2idx)
    random.shuffle(train_samples)
    split = int(len(train_samples) * (1 - args.val_ratio))
    tr_samples = train_samples[:split]
    va_samples = train_samples[split:]
    te_samples = collect_samples(test_dir, cls2idx)
    print(
        f"[INFO] train={len(tr_samples)} val={len(va_samples)} test={len(te_samples)}",
        flush=True,
    )

    train_ds = SkinDataset(tr_samples, args.image_size)
    val_ds = SkinDataset(va_samples, args.image_size)
    test_ds = SkinDataset(te_samples, args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} torch_cuda={torch.version.cuda}", flush=True)
    print(f"[INFO] building model={args.model_name} pretrained={args.pretrained}", flush=True)
    model = SwinHead(
        args.model_name,
        len(class_names),
        args.image_size,
        args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    print("[INFO] model ready", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    best_path = out_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        , flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "best_val_acc": best_acc,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Best model test | loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}", flush=True)

    onnx_path = out_dir / "best_model.onnx"
    export_onnx(model, onnx_path, args.image_size, device)

    meta = {
        "model_name": args.model_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": args.image_size,
        "dropout": args.dropout,
        "best_val_acc": best_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "onnx_path": str(onnx_path),
        "checkpoint_path": str(best_path),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Export done: {out_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

```

--------------------------------------
## 后端FastAPI接口构建

- chat_with.py 路由构建
```python
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

```


- app构建

```python
import sys
import argparse
from pathlib import Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from backend.api.apps import create_app

app = create_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

```

- 配置构建

```python
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
project_path = Path(__file__).parents[1].parent

EMAIL_PWD = os.getenv('MAIL_VAL')
EMAIL_SENDER = os.getenv('MAIL_SENDER')

print(project_path)


```

- 程序主入口

```python
import argparse
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    project_root = Path(__file__).resolve().parent
    src_root = project_root / "src"

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    load_dotenv(project_root / ".env")

    os.environ.setdefault(
        "SKIN_CLS_EXPORT_DIR",
        str(project_root / "model_/exports/swinv2"),
    )

    from backend.api.main import app

    parser = argparse.ArgumentParser(description="SkinHealthButler launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8099, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

```

----------------
## 前端代码

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SkinHealthButler · 肤康管家</title>
  <style>
    :root {
      --bg: #e9f0f3; --panel: #fff; --ink: #1f2e38; --sub: #657784; --line: #d7e3ea;
      --brand: #0b8f7f; --brand2: #17bfa8; --bot: #f7fafc; --user: #ecfdf5; --warn: #b42318;
    }
    *{box-sizing:border-box} body{margin:0;font-family:"Segoe UI","PingFang SC",sans-serif;color:var(--ink);
      min-height:100vh;background:radial-gradient(circle at 8% 0%,#d2f8f2 0,transparent 25%),radial-gradient(circle at 96% 100%,#d7eefc 0,transparent 25%),var(--bg);padding:18px}
    .shell{max-width:1180px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:20px;overflow:hidden;box-shadow:0 16px 48px rgba(20,38,53,.12)}
    .top{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:14px 18px;background:linear-gradient(95deg,var(--brand),var(--brand2));color:#fff}
    .brand{display:flex;gap:12px;align-items:center}.brand img{width:42px;height:42px;border-radius:11px;background:#fff;border:2px solid #ffffffaa}
    .brand h1{margin:0;font-size:19px}.brand p{margin:0;font-size:12px;opacity:.95}
    .layout{display:grid;grid-template-columns:320px 1fr;min-height:80vh}
    .side{border-right:1px solid var(--line);padding:14px;display:flex;flex-direction:column;gap:12px;background:#f9fcfd}
    .card{border:1px solid var(--line);border-radius:12px;background:#fff;padding:10px}.card h3{margin:0 0 8px;font-size:14px}
    .muted{font-size:12px;color:var(--sub);line-height:1.45}.field{display:grid;gap:6px}.field label{font-size:12px;color:var(--sub);font-weight:600}
    input[type=text],textarea,input[type=file]{width:100%;border:1px solid var(--line);border-radius:9px;padding:9px 10px;font-size:14px}
    textarea{min-height:84px;max-height:180px;resize:vertical}
    .btn{border:0;background:linear-gradient(90deg,var(--brand),var(--brand2));color:#fff;border-radius:9px;padding:9px 13px;font-weight:700;cursor:pointer}
    .btn:disabled{opacity:.6;cursor:not-allowed}
    .btn-lite{border:1px solid var(--line);background:#fff;color:#274556;border-radius:9px;padding:8px 10px;font-weight:600;cursor:pointer}
    .sessions{display:flex;flex-direction:column;gap:6px;max-height:180px;overflow:auto}
    .sess{border:1px solid var(--line);border-radius:9px;padding:8px;cursor:pointer;background:#fff;font-size:12px}
    .sess.active{border-color:#84d9ca;background:#ecfffb}
    .chip{display:inline-block;padding:4px 8px;border-radius:20px;border:1px solid #b7efe7;background:#e6fffb;color:#106f64;font-size:12px;font-weight:600;margin:4px 4px 0 0}
    .main{display:flex;flex-direction:column;min-height:0}.chat{flex:1;overflow:auto;padding:16px;display:flex;flex-direction:column;gap:10px;background:linear-gradient(180deg,rgba(243,251,255,.65),#fff 30%)}
    .msg{max-width:82%;border:1px solid var(--line);border-radius:12px;padding:10px 12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;font-size:14px}
    .msg.user{margin-left:auto;background:var(--user);border-color:#a8e4cf}.msg.bot{margin-right:auto;background:var(--bot)}
    .status{display:flex;justify-content:space-between;font-size:12px;color:var(--sub);padding:8px 12px;border-top:1px solid var(--line);background:#fbfdfe}
    .composer{padding:12px;border-top:1px solid var(--line);display:grid;gap:10px}
    .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.err{color:var(--warn);font-size:13px;min-height:18px}
    .preview{display:none}.preview img{width:100%;max-height:170px;object-fit:contain;border:1px solid var(--line);border-radius:9px;background:#f8fbfd}
    @media (max-width:940px){.layout{grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid var(--line)}.msg{max-width:92%}}
  </style>
</head>
<body>
<div class="shell">
  <div class="top">
    <div class="brand">
      <img src="/imgs/icon.png" alt="icon" />
      <div><h1>SkinHealthButler 肤康管家</h1><p>CV 分类 + NLP 对话 · 自动会话 · 流式打字机</p></div>
    </div>
    <div style="font-size:12px;opacity:.95">上传图片可辅助分类，再结合描述生成建议</div>
  </div>

  <div class="layout">
    <aside class="side">
      <section class="card">
        <h3>连接设置</h3>
        <div class="field"><label>API Base URL</label><input id="apiBase" type="text" value="http://127.0.0.1:8099" /></div>
      </section>

      <section class="card">
        <div class="row" style="justify-content:space-between"><h3 style="margin:0">会话</h3><button id="newSession" class="btn-lite">+ 新建会话</button></div>
        <div id="sessions" class="sessions"></div>
      </section>

      <section class="card">
        <h3>图片预览</h3>
        <div id="preview" class="preview"><img id="previewImg" alt="preview" /></div>
        <div id="previewText" class="muted">选择图片后将显示预览</div>
      </section>

      <section class="card">
        <h3>分类结果</h3>
        <div id="clsMain" class="muted">等待图片上传并发送后显示</div>
        <div id="clsTopk"></div>
      </section>
    </aside>

    <main class="main">
      <div id="chat" class="chat"></div>
      <div class="status"><span id="status">空闲</span><span>快捷键：Ctrl/⌘ + Enter 发送</span></div>
      <div class="composer">
        <textarea id="message" placeholder="请输入症状描述，如：脸颊发红发痒 3 天，晚上更明显"></textarea>
        <div class="row">
          <input id="image" type="file" accept="image/jpeg,image/png,image/bmp" />
          <button id="send" class="btn">发送（流式）</button>
        </div>
        <div id="error" class="err"></div>
      </div>
    </main>
  </div>
</div>

<script>
  const chatEl = document.getElementById('chat');
  const msgEl = document.getElementById('message');
  const sendBtn = document.getElementById('send');
  const imageEl = document.getElementById('image');
  const apiBaseEl = document.getElementById('apiBase');
  const statusEl = document.getElementById('status');
  const errorEl = document.getElementById('error');
  const clsMainEl = document.getElementById('clsMain');
  const clsTopkEl = document.getElementById('clsTopk');
  const previewWrap = document.getElementById('preview');
  const previewImg = document.getElementById('previewImg');
  const previewText = document.getElementById('previewText');
  const sessionsEl = document.getElementById('sessions');
  const newSessionBtn = document.getElementById('newSession');

  const chats = new Map();
  let currentSessionId = '';

  const queue = [];
  let typingTimer = null;

  function makeSessionId(){
    const t = Date.now().toString(36);
    const r = Math.random().toString(36).slice(2,8);
    return `chat-${t}-${r}`;
  }

  function ensureSession(id){
    if(!chats.has(id)) chats.set(id,{id,title:'新会话',messages:[]});
    return chats.get(id);
  }

  function renderSessionList(){
    sessionsEl.innerHTML='';
    [...chats.values()].forEach(s=>{
      const d=document.createElement('div');
      d.className='sess'+(s.id===currentSessionId?' active':'');
      d.textContent=s.title;
      d.title=s.id;
      d.onclick=()=>switchSession(s.id);
      sessionsEl.appendChild(d);
    });
  }

  function switchSession(id){
    currentSessionId=id; ensureSession(id);
    chatEl.innerHTML='';
    const sess=chats.get(id);
    sess.messages.forEach(m=>addMsg(m.text,m.role,false));
    renderSessionList();
  }

  function newSession(){
    const id=makeSessionId();
    ensureSession(id);
    switchSession(id);
    clsMainEl.textContent='等待图片上传并发送后显示';
    clsTopkEl.innerHTML='';
  }

  function addMsg(text, role, persist=true){
    const div=document.createElement('div');
    div.className=`msg ${role}`;
    div.textContent=text;
    chatEl.appendChild(div);
    chatEl.scrollTop=chatEl.scrollHeight;
    if(persist){
      const sess=ensureSession(currentSessionId);
      sess.messages.push({role,text});
      if(sess.title==='新会话' && role==='user') sess.title=text.slice(0,24);
      renderSessionList();
    }
    return div;
  }

  function renderCls(meta){
    clsTopkEl.innerHTML='';
    if(!meta || !meta.classification){ clsMainEl.textContent='本轮未使用图片分类'; return; }
    const c=meta.classification;
    clsMainEl.innerHTML=`Top1：<strong>${c.label}</strong>（${(c.confidence*100).toFixed(2)}%）`;
    (c.topk||[]).forEach(i=>{
      const sp=document.createElement('span'); sp.className='chip';
      sp.textContent=`${i.label} ${(i.confidence*100).toFixed(1)}%`; clsTopkEl.appendChild(sp);
    });
  }

  function startTypewriter(target){
    if(typingTimer) return;
    typingTimer=setInterval(()=>{
      if(queue.length===0){ clearInterval(typingTimer); typingTimer=null; return; }
      target.textContent += queue.shift();
      chatEl.scrollTop = chatEl.scrollHeight;
    }, 14);
  }

  imageEl.addEventListener('change', ()=>{
    const f=imageEl.files[0];
    if(!f){ previewWrap.style.display='none'; previewText.textContent='选择图片后将显示预览'; return; }
    const url=URL.createObjectURL(f);
    previewImg.src=url;
    previewWrap.style.display='block';
    previewText.textContent=`${f.name} · ${(f.size/1024).toFixed(1)} KB`;
  });

  async function send(){
    const message=msgEl.value.trim();
    if(!message){ errorEl.textContent='请输入问题后再发送'; return; }
    errorEl.textContent=''; statusEl.textContent='正在生成...'; sendBtn.disabled=true;

    addMsg(message,'user',true);
    const botMsg=addMsg('', 'bot', false);

    const fd=new FormData();
    fd.append('message', message);
    fd.append('session_id', currentSessionId);
    if(imageEl.files[0]) fd.append('image', imageEl.files[0]);

    try{
      const base=apiBaseEl.value.trim().replace(/\/$/, '');
      const resp=await fetch(`${base}/chat/respond_stream`, {method:'POST', body:fd});
      if(!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      const reader=resp.body.getReader();
      const decoder=new TextDecoder('utf-8');
      let buffer='';

      while(true){
        const {value,done}=await reader.read();
        if(done) break;
        buffer += decoder.decode(value,{stream:true});
        const events=buffer.split('\n\n');
        buffer=events.pop()||'';

        for(const evt of events){
          const line=evt.split('\n').find(l=>l.startsWith('data: '));
          if(!line) continue;
          const payload=JSON.parse(line.slice(6));
          if(payload.type==='meta') renderCls(payload);
          else if(payload.type==='chunk'){
            const chars=[...payload.delta];
            queue.push(...chars);
            startTypewriter(botMsg);
          } else if(payload.type==='done') statusEl.textContent='完成';
        }
      }

      while(queue.length>0){ await new Promise(r=>setTimeout(r,20)); }
      const sess=ensureSession(currentSessionId);
      sess.messages.push({role:'bot', text:botMsg.textContent});
    }catch(e){
      errorEl.textContent=`请求失败：${e.message}`; statusEl.textContent='失败';
    }finally{
      sendBtn.disabled=false; msgEl.value=''; imageEl.value='';
    }
  }

  newSessionBtn.addEventListener('click', newSession);
  sendBtn.addEventListener('click', send);
  msgEl.addEventListener('keydown',e=>{ if((e.ctrlKey||e.metaKey)&&e.key==='Enter') send(); });

  newSession();
</script>
</body>
</html>

```


----------------

## 其它配置相关

- redisstack_server docker镜像
![](/imgs/08.png)

- 注意，运行前请打开redis_stack_server
```shell
nikofox@MOSS:~$ docker run -d -p 65522:6379 798ab
ddaf312ada7ffbbda10933236008696f2e1b5364c3440fdadcb01e5bf37814df
nikofox@MOSS:~$ docker ps
CONTAINER ID   IMAGE     COMMAND            CREATED         STATUS         PORTS                                           NAMES
ddaf312ada7f   798ab     "/entrypoint.sh"   3 seconds ago   Up 2 seconds   0.0.0.0:65522->6379/tcp, [::]:65522->6379/tcp   silly_mclean
nikofox@MOSS:~$ 

```


--------------------

## 启动主程序

```shell
/home/nikofox/.local/bin/uv run /home/nikofox/llm_projects/SkinHealthButler/.venv/bin/python /home/nikofox/llm_projects/SkinHealthButler/main.py 
/home/nikofox/llm_projects/SkinHealthButler/src
INFO:     Started server process [20830]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8099 (Press CTRL+C to quit)
INFO:     127.0.0.1:43022 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:43022 - "GET /imgs/icon.png HTTP/1.1" 200 OK
```
### 主程序界面

![](/imgs/09.png)


### 点击可新建会话，会话与会话之间可相互隔离
![](/imgs/10.png)

### 主会话交互窗口
![](/imgs/11.png)

1. 对话交互展示区
2. 文字输入区
3. 皮肤图片上传区
4. 发送按钮

### 会话展示
![](/imgs/12.png)

![](/imgs/13.png)

### 图片上传与询问
![](/imgs/14.png)

![](/imgs/15.png)

### 记忆对话机制展示
![](/imgs/16.png)

### 对话历史保存之数据库后台展示
![](/imgs/17.png)