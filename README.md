# SkinHealthButler
-----------------------------------------------------------



##  蒸馏QA (基于Qwen3-MAX)  




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
[](./imgs/01.png)



