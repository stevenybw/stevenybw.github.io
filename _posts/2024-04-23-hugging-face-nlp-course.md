---
layout: post
title:  "Hugging Face NLP Course"
date:   2024-04-23 11:04:00 +0800
categories: paper_reading
---
# 概要

本文记录本人学习[Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)的心得。

章节介绍：
* Chapter 1~4介绍了Transformer库，学习Transformer如何工作，并且如何使用Hugging Face Hub上的模型，面向一个数据集微调，并且在Hub上分享你的结果。
* Chapter 5~8学习Dataset和Tokenizer的基础。
* Chapter 9~12介绍Transformer模型如何用于语音处理、计算机视觉。

先需：
* Python
* Deep Learning

后继：
* DeepLearning.AI在Coursera上开设的专项课程[Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearning-ai&utm_medium=institutions&utm_campaign=20211011-nlp-2-hugging_face-page-nlp-refresh)。

## Chapter 0: Setup

还需要参照[Getting Started with Repositories](https://huggingface.co/docs/hub/repositories-getting-started)来访问受限的repo。

```bash
mkdir ~/transformers-course
cd ~/transformers-course
python3 -m venv .env
source .env/bin/activate
pip install transformers[sentencepiece]
pip install torch

pip install huggingface_hub
huggingface-cli login
```

## Chapter 1: Transformer Model

主题
* 使用`pipeline`函数
* 学习Transformer架构
* 区分encoder、decoder、encoder-decoder三种架构

NLP是什么？
* 文本分类
* 单词分类（文本中的单个词语）
* 文本生成
* QA
* 翻译与汇总

Transformer是什么？

```python
from transformers import pipeline

# No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
# 同时还会下载大约268MB的模型参数
classifier = pipeline("sentiment-analysis")

classifier("I've been waiting for a HuggingFace course my whole life.")

# [{'label': 'NEGATIVE', 'score': 0.9583979249000549}]
classifier("You monster!")

# [{'label': 'POSITIVE', 'score': 0.999804675579071}]
classifier("You clever monster!")
```

上述过程有三步：
1. 文本预处理成模型能理解的格式
2. 预处理输入传递给模型
3. 后处理模型输出

[当前可用的管线](https://huggingface.co/transformers/main_classes/pipelines)
1. feature-extraction（文本向量化）
2. fill-mask
3. ner（实体识别）
4. question-answering
5. sentiment-analysis
6. summarization
7. text-generation
8. translation
9. zero-shot-classification

### 零样本分类（Zero-shot classification）

在机器学习领域，Zero-shot classification（零样本分类）是指模型在没有接受过任何特定类别样本的训练的情况下，仍然能够识别出这些类别的任务。这种方法主要用于解决标注数据稀缺的问题，特别是在某些类别难以获得足够训练样本的情况下。

```python
from transformers import pipeline

# No model was supplied, defaulted to facebook/bart-large-mnli and revision c626438 (https://huggingface.co/facebook/bart-large-mnli).
# 下载了1.63GB的数据
classifier = pipeline("zero-shot-classification")

classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

# {'sequence': 'Byte-Pair Encoding: Subword-based tokenization algorithm', 'labels': ['computer science', 'enginnering', 'business', 'politics'], 'scores': [0.6000896692276001, 0.3114130198955536, 0.0760892927646637, 0.012407983653247356]}
classifier(
    "Byte-Pair Encoding: Subword-based tokenization algorithm",
    candidate_labels=["computer science", "politics", "business", "enginnering"],
)
```

### 文本生成

```python
from transformers import pipeline

# No model was supplied, defaulted to openai-community/gpt2 and revision 6c0e608 (https://huggingface.co/openai-community/gpt2).
# 下载了548MB的模型
generator = pipeline("text-generation")
generator("In this course, we will teach you how to", num_return_sequences=2, max_length=15)
```

### 使用任何模型

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

在[这里](https://huggingface.co/models?pipeline_tag=text-generation)可以看到所有的可以给文本生成任务使用的模型。
排行第一的是`Meta-Llama-3-8B`，不过使用它需要申请。

因此使用[WizardLM-2-8x22B](https://huggingface.co/alpindale/WizardLM-2-8x22B)。
WizardLM-2系列由WizardLM@Microsoft AI团队开发，属于MoE模型，基于`mistral-community/Mixtral-8x22B-v0.1`开发，参数量高达`141B`，支持多种语言，包括`8x22B`、`70B`、`7B`三个模型。
`8x22B`是最先进的模型，比所有开源模型要强，甚至和专有模型有相近的性能表现。
`70B`的reasoning能力强，同等大小模型中的首选。
`7B`最快，可以和10倍大小的开源模型匹敌。
但是载入失败了，原因是超内存，它需要把所有的模型都载入进来才能进行计算。

```python
from transformers import pipeline

# 
# 在AWS上以500MB/s的超快速度下载
# 需要下载59个文件，每个4.81GB，共计283.79GB
generator = pipeline("text-generation", model="alpindale/WizardLM-2-8x22B")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

换一个更小的模型试试，发现内存几乎占满，所以从`m7i.2xlarge`切换成`r7i.2xlarge`，没有贵多少，但是内存翻了一倍。
从这里可以看出云的优势，你随时可以很容易地切换配置。

```python
from transformers import pipeline

# 会下载15GB的数据，即使什么也不做也要占据约26GB的内存
generator = pipeline("text-generation", model="lucyknada/microsoft_WizardLM-2-7B")

# Greedy methods without beam search do not support `num_return_sequences` different than 1 (got 2).
# generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
# )

# [{'generated_text': 'In this course, we will teach you how to create a successful online business that can generate a passive income for years to come. You will learn how to find profitable niches, build a vast affiliate network, create content, drive organic traffic, and use the best conversion strategies. You can even see your business become one'}]
generator(
    "In this course, we will teach you how to",
    max_length=64,
    num_return_sequences=1,
)
```

**Hugging Face**可以支持类似SaaS的托管推理服务（`Inference API`）。对于小一些的模型，例如[distillgpt2](https://huggingface.co/distilbert/distilgpt2)，可以免费在模型页面上进行推理。对于大一些的模型，虽然没有免费的，但是可以付钱购买，例如上述7B的模型可以到[这里](https://ui.endpoints.huggingface.co/stevenybw/new?repository=lucyknada%2Fmicrosoft_WizardLM-2-7B)开启，推荐NVDA A10G节点（1 GPU），每小时1美元。上述8*22B的模型推荐NVDA A100节点（4GPU、320GB显存、44核、520GB主存），每小时16美元。

### Mask filling

注意不同的模型可能会有不同的Mask token。[distilroberta-base](https://huggingface.co/distilbert/distilroberta-base)的Mask token是`<mask>`。而[bert-base-cased](https://huggingface.co/google-bert/bert-base-cased)的Mask token是`[MASK]`。

```python
from transformers import pipeline

# No model was supplied, defaulted to distilbert/distilroberta-base and revision ec58a5b
# 下载331MB的模型
unmasker = pipeline("fill-mask")

unmasker("This course will teach you all about <mask> models.", top_k=2)

# 会随机选择一个<mask>去填充，保留另一个<mask>不变
# unmasker("This course will teach you all about <mask> models <mask>.", top_k=2)

# hug 33.78%    comfort: 0.048  ...
print("\n".join([f"{x['token_str']}: {x['score']}" for x in  unmasker("She is crying. I should <mask> her.", top_k=10)]))

# kill 11.65%   forgive 9.5%   sue 9.4%  ...
print("\n".join([f"{x['token_str']}: {x['score']}" for x in  unmasker("She makes me very angry. I should <mask> her.", top_k=10)]))
```

### Named entity recognition

找到句子里的所有有名字的实体。

```python
from transformers import pipeline

# No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision f2482bf
# 模型大小：1.33GB
ner = pipeline("ner", grouped_entities=True)

# [{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
#  {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
#  {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
# ]
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

### 信息提取（Question answering）

注意：QA pipeline并不生成答案，仅仅是从上下文中提取出有助于回答这个问题的信息。

```python
from transformers import pipeline

# distilbert/distilbert-base-cased-distilled-squad
# 261MB
question_answerer = pipeline("question-answering")

# {'score': 0.6949753165245056, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

# {'score': 0.4169579744338989, 'start': 3, 'end': 21, 'answer': 'weight is too high'}
question_answerer(
    question="What should I do?",
    context="My weight is too high",
)

# {'score': 0.31994733214378357, 'start': 32, 'end': 36, 'answer': '85kg'}
question_answerer(
    question="What is my weight?",
    context="My weight is too high, which is 85kg.",
)

# {'score': 0.3924005627632141, 'start': 13, 'end': 21, 'answer': 'too high'}
question_answerer(
    question="What is my weight?",
    context="My weight is too high, which is 85kg and 187 pounds.",
)
```

### Summarization

```python
from transformers import pipeline

# sshleifer/distilbart-cnn-12-6
# 1.22GB
summarizer = pipeline("summarization")

# [{'summary_text': ' America has changed dramatically during recent years . The number of engineering graduates in the U.S. has declined in traditional engineering disciplines such as mechanical,'}]
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
""", min_length=16, max_length=32)
```

### Translation

```python
from transformers import pipeline

# 301M
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

# [{'translation_text': 'This course is produced by Hugging Face.'}]
translator("Ce cours est produit par Hugging Face.")
```
