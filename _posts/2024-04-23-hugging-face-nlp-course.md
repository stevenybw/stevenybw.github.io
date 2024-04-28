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

可见Hugging Face Datasets包含了音频与视频数据。如何安装Dataset参见[这里](https://huggingface.co/docs/datasets/en/installation)。

[Hugging Face Accelerate库](https://huggingface.co/docs/accelerate/en/index)可以让你只需写一份PyTorch代码，并且能运行在各种不同的分布式配置下，只需要你做一些微小的变动。建立在`torch_xla`和`torch.distributed`，Hugging Face Accelerate库帮你做其他各种复杂的工作，并自动使用`DeepSpeed`、`fully shareded data parallelism`、自动混合精度等。

在执行`accelerate config`的时候它会问你一些问题。

支持的配置：
1. 单机
2. 多CPU
3. 多XPU
4. 多GPU
5. 多NPU
6. 多MLU
7. TPU

面向CPU集群英特尔有`Intel PyTorch Extension (IPEX)`来加速训练过程。

同时还可以使用TorchDynamo来加速PyTorch程序。

Torch Dynamo有各种后端：
1. eager
2. aot_eager
3. inductor【默认】
4. aot_ts_nvfuser
5. nvprims_nvfuser
6. cudagraphs
7. ofi
8. fx2trt
9. onnxrt
10. tensorrt
11. ipex
12. tvm

可以为`torch.compile`配置默认行为（参见[torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html)）。
1. `mode`参数。`default`平衡性能与开销。`reduce-overhead`。`max-autotune`。
2. `fullgraph`参数。默认False，能容忍`graph breaks`，尽可能地把函数里的子图进行优化。如果是True，那么我们需要整个函数可以放在一个计算图里，若无法实现则抛异常。
3. `dynamic`参数。默认是None，会自动检测。若是False，永远不会生成dynamic kernel，永远都会做specialization。如果是True，会预先编译一个尽量动态的kernel避免specialization带来的编译时间。

是否使用DeepSpeed？No

是否使用混合精度？[第四代志强处理器支持INT8和BF16两个数据类型](https://www.intel.com/content/www/us/en/content-details/785250/accelerate-artificial-intelligence-ai-workloads-with-intel-advanced-matrix-extensions-intel-amx.html)

最终在`/root/.cache/huggingface/accelerate/default_config.yaml`产生一个配置文件：

```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: INDUCTOR
  dynamo_mode: default
  dynamo_use_dynamic: false
  dynamo_use_fullgraph: false
enable_cpu_affinity: false
ipex_config:
  ipex: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
```

[Hugging Face Evaluate库](https://github.com/huggingface/evaluate/)提供了一套易用的、标准化的模型评价、比较方案，实现了各种Metric，允许比较各种模型。

```bash
mkdir ~/transformers-course
cd ~/transformers-course
python3 -m venv .env
source .env/bin/activate
pip install transformers[sentencepiece]
pip install datasets[audio,vision]
pip install torch
pip install evaluate

pip install accelerate
accelerate config

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

### Transformer工作原理

Transformer工作原理强烈推荐[在Transformer架构中，第四步"残差连接"是做什么的？](https://zhuanlan.zhihu.com/p/338817680)，很清楚地说明了Transformer的宏观与微观的工作原理。

Transformer架构在2017年提出，当初用作翻译。接下来就出现了一些有影响力的后继工作。
这里给出的不够新，[AI / ML / LLM / Transformer Models Timeline and List](https://ai.v-gar.de/ml/transformer/timeline/)更详细地整理了最新的工作。
1. 2018年6月：GPT。第一个预训练Transformer模型，并且面向各种NLP任务做fine-tuning。
2. 2018年10月：BERT。另一个预训练模型，更擅长总结。
3. 2019年2月：GPT-2。更大的GPT。
4. 2019年10月：DistilBERT。蒸馏版的BERT。相比BERT，快60%、省40%内存，97%的性能。
5. 2019年10月：BART、T5。第一个使用exactly原版Transformer模型做预训练模型的工作。
6. 2020年5月：GPT-3。比GPT-2更大。不需要fine-tuning就可以做各种任务（被称为zero-shot learning）
7. **TODO**

总得来说，可以分成三类：
1. 类GPT（自回归Transformer模型）
2. 类BERT（自编码Transformer模型）
3. 类BART/T5（序列到序列Transformer模型）

预训练模型是一种**自监督学习（Self-Supervised Learning）**。
**预训练（Pretraining）**是从头从一个随机权重训练一个模型，没有任何先验知识。
**微调（Fine-tuning）**是在模型预训练好之后的进一步训练。
这是一种**迁移学习（Transfer learning）**的实践。

Transformer模型主要有两个模块：
1. 编码器（Encoder）：将输入映射到向量空间。该模型的优化目标是深入理解输入。
2. 解码器（Decoder）：将编码器的输入结合其他的输入生成一个目标序列。该模型的优化目标是产生输出。

这两个模块既可以单独使用也可以结合使用：
1. 仅编码器：用于理解输入的场景，例如句子分类、NER等等。
2. 仅解码器：用于生成式任务，例如文本生成。
3. 编码器+解码器：也称为序列到序列模型（Sequence-to-Sequence model），用于翻译或者Summary。

Transformer模型最大的特点是用了一个特殊的**Attention层**，该层帮助模型在句子里划重点。
也就是说，在句子里告诉模型哪些词是需要特别注意的，哪些词是可以忽略的。
以将"You like this course"翻译成法语为例。
在翻译"like"这个单词时，需要特别注意"You"，因为它在法语的形态会和主语相关；但是其他的部分就没啥关系了。
在翻译"this"这个单词时，需要特别注意"course"，因为它在法语的形态会和关联名词的阴阳性；其他部分没啥关系。
对于一些更复杂的句子，可能需要注意到非常遥远的单词。

最原始的Transformer架构主要用于翻译。
在训练的时候，编码器的Attention层可以使用句子里的所有单词。
但是，解码器只能注意到已经生成的单词（毕竟没来得及生成的都不知道是什么），并且根据已经生成的单词去生成下一个单词。

最原始的Transformer架构有一张图描述。
1. 输入首先过一个"Input Embedding"将每个token映射到一个预定义的、固定维度的向量（通常是Word2Vec、GloVe、Transformer自身的嵌入层等）。
2. 接下来通过一个"Positional Encoding"将每个token在句子中的位置编码进去，这是必要的，因为不像RNN那样天然是处理序列的模型，Transformer模型本身不具备处理序列数据的能力。通常使用正弦、余弦函数的组合来生成每个位置的独特编码。对于token在句子中的位置`pos`和嵌入向量的编号与维度`i`、`d`，如果i是偶数，编码成`sin(pos/10000^{i/d})`，如果i是奇数，编码成`cos(pos/10000^{i/d})`。也就是说，不同`i`会随着`pos`的增加以不同周期震荡。
3. 接下来会过一个"Multi-head Attention"，让模型能够同时从不同的子空间角度关注输入中的各个部分。具体地，输入的嵌入向量会被复制成多个“头”，并各自独立进行自注意力的计算，让每个头捕捉到输入数据的不同特征。
   1. [Self-Attention的结构（推荐阅读）](https://zhuanlan.zhihu.com/p/338817680)包含三个矩阵Wq、Wk、Wv。一批输入X通过定义的线性变换变为一批输出Q、K、V。然后输出是$softmax(\frac{QK^T}{\sqrt{d_k}})V$。第一步$QK^T$得到$n \times n$的矩阵（假设n为句子的单词数），表示单词之间的Attention强度。softmax让它变成概率分布。乘以V使得每个单词的向量等于根据Attention向量将其他所有的单词向量加权求和，也就是说按照这个Softmax做加权平均。简单地来说，**对于每个Query，与所有Key的点积来获取注意力分数，将这些分数通过Softmax来归一化之后，再按照Value向量求加权和，作为输出。**
   2. Multi-Head Attention包含多个Self-Attention层（记为h），同一个输入喂给$h$个Self-Attention层，产生的输出沿着特征维度拼接（单词数维度不变），做一个线性变换得到最终输出，并且输出矩阵的维度和输入矩阵的维度是一样的。
4. 接下来会先做一个残差连接
   1. 残差连接的目的是缓解深层网络训练过程中的梯度消失问题，从而使得模型能够有效地训练更深的网络结构。
   2. 残差连接的实现很简单，就是在网络的某几个层之后添加一个直接连接，将输入直接加到输出上。
   3. 它可以防止训练过程中的梯度消失问题，允许梯度直接流过网络。
   4. 它可以保持信息流，将输入与输出直接相加，可以保证至少有一部分原始信息不经修改地通过网络，有助于在处理复杂函数时保留必要的信息。
   5. 它可以加速收敛，提供了一种额外的路径来快速调整网络权证。
   6. 他可以提高模型性能。
5. 接下来会再做一个Layer Normalization
   1. 在Transformer架构中，层归一化是一个重要组成部分，用于稳定训练过程，加速收敛，并提高模型的泛化能力。
   2. 具体地，它会为每个样本独立地进行归一化，减少不同训练样本间的尺度差异。具体公式为：$LN(x_i) = \gamma (\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}) + \beta$。其中，$\mu$和$\sigma$分别是当前样本所有特征的均值与方差，$\epsilon$避免除以零。$\gamma$和$\beta$是可学习的参数，允许Layer Norm在必要的标准化的同时保留网络必要的表达能力。
6. 接下来会再做一个Feed Forward，由两层FC构成，一层有ReLU，第二层没有：$max(0, XW_1 + b_1)W_2 + b_2$。

# 背景知识

## torch dynamo

Torch Dynamo 是一个优化框架，属于 PyTorch 生态系统的一部分。它的目标是自动优化 PyTorch 程序，提高执行效率。Torch Dynamo 通过动态地分析和转换 PyTorch 的代码，可以将其运行时的效率提升至更接近手工优化的水平。

Dynamo 通过一个名为 "FX Graph Mode" 的中间表示（IR）来工作，这允许它在运行时捕获和转换 PyTorch 的操作。这种方式使得 Dynamo 能够优化那些通常难以预测或者动态变化的代码，如控制流和循环结构等。通过这种方式，Torch Dynamo 旨在提供一种自动化的优化方法，用户无需进行复杂的代码重写就能提升性能。

参见[TorchDynami](https://pytorch.org/docs/stable/torch.compiler_deepdive.html)。TorchDynamo是一个Python级别的JIT编译器，不用进行任何修改，就可以让PyTorch程序变快。
它将Python BC重写并提取PyTorch操作，变换成一个FX Graph。

[TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)是一个PyTorch-native编译器。

## KV Cache

对于Decoder-only的生成式模型，例如GPT-2，关于KV Cache的动机[这篇文章里有一个动图非常形象地展示了问题所在以及KVCache缓存了哪些东东](https://medium.com/@joaolages/kv-caching-explained-276520203249)：每次预测了下一个token之后，还要把这个token插入回input并进行下一次预测，这叫**自回归特性**。

根据[这里](https://www.zhihu.com/question/596900067)，KV Cache是通过空间换时间的思想，提高推理性能。生成式模型每次推理只会输出一个token，然后与输入tokens拼接在一起，作为下次推理的输入，直到遇到一个终止符。因为模型就是一个函数，输入token序列，输出下一个token的概率分布。如果说模型内部没有缓存机制，那么就需要大量的重算。

结论：KVCache就是缓存序列的每个token在每个Transformer的的Key和Value。这个技术难就难在它需要侵入模型推理的API，给API引入一个缓存。

推荐阅读https://jalammar.github.io/illustrated-gpt2/

## Layer Normalization

参见[这里](https://zhuanlan.zhihu.com/p/54530247)。Batch Normalization是将这个批次里的所有样本的同一个通道的特征做归一化。Layer Normalization是将同一个样本的不同通道做归一化。

BN的问题是，当样本数很少时，均值和方差不能反映全局的统计分布信息。
而且在RNN里，统计到靠后的时间片时，数据量也很少，此时BN的效果也不好。

## 为什么预训练模型是Self-Supervised Learning而不是Unsupervised Learning？

预训练模型如何分类（Self-Supervised Learning 或者 Unsupervised Learning）主要取决于它们在学习过程中如何使用数据。这两种学习范式都不依赖于人工标注的数据，但在目标和方法上有所区别。下面是一些关键的区别：

1. **Unsupervised Learning（无监督学习）**：
   - 无监督学习的目标是在没有任何标签的情况下探索数据。这种方法通常用于聚类、密度估计或者降维。无监督学习试图在数据中找到某种结构，例如通过聚类相似的实例，或者通过主成分分析（PCA）找到数据的主要变化方向。
   - 无监督学习的挑战在于模型必须自行决定数据的重要属性，因为没有标签或者反馈来指示哪些特征是有用的。

2. **Self-Supervised Learning（自监督学习）**：
   - 自监督学习是一种特殊类型的学习方法，属于监督学习的一种，但它使用数据本身生成伪标签来进行训练。在自监督学习中，输入数据被用来作为其自身的监督，例如，通过预测数据的某个部分（如遮蔽语言模型在NLP中的应用），或者预测数据中的序列（如在视频帧预测中）。
   - 自监督学习的关键在于利用数据结构自身生成监督信号，使得模型能在没有外部标注的情况下学习到有用的特征。这种方法常用于预训练模型，以便在后续的监督学习任务中达到更好的性能。

在预训练模型的上下文中，自监督学习提供了一种强大的方法来从未标记的数据中提取有用的特征。这使得预训练模型在接受少量标注数据进行微调时，能够显示出更好的性能。自监督学习在诸如自然语言处理（NLP）和计算机视觉等领域已经显示出巨大的潜力，因为它允许模型利用大量的未标记数据，从而学习到更深层次、更通用的数据表征。

### Encoder模型

预训练通常是掩盖随机的单词让它预测。

典型的有ALBERT、BERT、DistilBERT、ELECTRA等

### Decoder模型

预训练通常是预测句子中的下一个token。

典型的有CTRL、GPT、GPT-2、Transformer XL

### Sequence-to-sequence模型

预训练通常是掩盖文本中的随机的一段话它预测。

典型的有BART、mBART、Marian、T5

## Chapter 2: Using Transformers

Hugging Face Transformers库的目标是提供一个统一API，通过这个API，可以加载、训练、保存Transformer模型。具有以下特性：
1. 易用性。两行代码，自动下载、加载、使用最新的NLP模型。
2. 灵活性。所有模型都是PyTorch nn.Module或者TensorFlow tf.keras.Model类。
3. 简单性。All in one file理念，模型的forward-pass定义在一个文件，易于理解与魔改。

最后一个特性是Transformer区分于其他ML库的重要特性。通常，在一些深度学习框架中，为了代码复用和简洁，多个模型可能会共享一些通用的网络层或者模块。但是Hugging Face的理念是每个模型都是**自包含**的，便于用户阅读理解，也便于用户修改实验。

本章会阐述如何用模型和Tokenizer复现`pipeline`函数。然后回深入学习Tokenizer API，它用在最开始和最后一步，负责在文本形式与数值形式之间转换。最后，我们会学习如何在一个批里发送多个句子。

### Behind the pipeline

在调用`pipeline("sentiment-analysis")`的背后，发生了以下事情：
1. 通过Tokenizer将文本拆分成Token ID的序列。
2. 将上述序列输入一个Model得到一个Logits向量。
3. 通过一个分类器将Logits向量进行分类。

#### Tokenizer：负责做预处理

上述预处理方法需要和预训练模型保持**严格一致**，Hugging Face能自动帮你将模型匹配到对应的Tokenizer。

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
    "Hey you!",
    "Aha!",
    "Bowen!"
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("\n\n\n".join([f"{x}\n{y}\n{z}" for (x, y, z) in zip(raw_inputs, input_ids, attention_mask)]))
```

通过执行上述程序观察到：
1. Tokenizer API允许一次处理多个句子，并产生一个张量。由于句子的长度不一样，会以最大字符串长度来做**padding**，并且通过**masking**来保证正确性。

#### Model：一个基础模型再加上一个Head

这里的模型指的是**base model**，只会产生**隐藏层**的特征向量，而不是最终的结果。
通常在实际使用中会根据不同的task给隐藏层再加各种**head**。

输出是一个三维张量：批大小 * 最大序列长度 * 特征维度（通常小模型是768，大一些的可能会到3072）。


```python
from transformers import AutoTokenizer
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
    "Hey you!",
    "Aha!",
    "Bowen!"
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

# torch.Size([5, 16, 768])
print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# torch.Size([5, 2])
print(outputs.logits.shape)

import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

### Models

```python
from transformers import BertConfig, BertModel

config = BertConfig()

# 创建一个随机初始化的模型
model = BertModel(config)

# 载入一个已经训练好的模型（https://huggingface.co/models?other=bert 可以看到所有的Bert模型）
model = BertModel.from_pretrained("bert-base-cased")
```

### Tokenizer

#### Word-based

方法：朴素地按空格拆分成单词。

缺点：词汇量非常大，而且`run`、`runs`、`running`都会被作为一个单词，只有靠后续的训练才能捕捉到它们之间的关系。而且类似汉语根本就没有单词边界。

#### Character-based

方法：直接按字符做拆分

优势：词典非常小，而且几乎没有UNK。

缺点
1. It's less meaningful（取决于语言，**汉语的字符比拉丁语言的字符有意义得多**）
2. Too much tokens to be processed

#### Subword tokenization

原理：常用词不应该被拆分到更小的Subwords，但是生僻词应该被拆分成更有意义的Subwords。

有各种算法，例如
1. BPE，用在GPT-2
2. WordPiece，用在BERT
3. SentencePiece / Unigram，用在各种多语言模型

### Handling multiple sequences

Transformer模型对于序列长度有限制，大多数模型最多能处理512/1024个token。

[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer)可以处理1000+长度的句子。

[LED](https://huggingface.co/docs/transformers/model_doc/led)类似。

## Chapter 3：Fine-Tuning A Pretrained Model

### Introduction

本章主要介绍
1. 如何从Hub准备一个大数据集
2. 如何使用高层Trainer API来微调模型
3. 如何使用自定义训练循环
4. 如何采用Accelerate库来分布式化

### 处理数据

快速例子：

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# 抛异常，因为张量要求长度一致
batch = tokenizer(sequences, truncation=True, return_tensors="pt")

# 不返回PyTorch张量的话就没问题，此时返回的是Python List
batch = tokenizer(sequences, truncation=True)

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

接下来，将介绍如何获取Microsoft Research Paraphrase Corpus数据集（包含5801对句子，并且有一个Label表明它们是否是同一个意思）。

该数据集是[GLUE benchmark](https://gluebenchmark.com/)选中的10个数据集之一。

Hugging Face上有专门的分享数据集的地方：https://huggingface.co/datasets

下载数据集

```python
from datasets import load_dataset

# 载入https://huggingface.co/datasets/nyu-mll/glue数据集里的mrpc
raw_datasets = load_dataset("glue", "mrpc")

# 包含3668个训练集、408个验证集、1725个测试集
# 每个集合有4列，分别是句子1、句子2、标签、索引号
print(raw_datasets)

# 句子1：Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .
# 句子2：Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'
# label: 1
# idx: 0
print(raw_datasets["train"][0])

# 返回数据集的Schema
# {'sentence1': Value(dtype='string', id=None), 'sentence2': Value(dtype='string', id=None), 'label': ClassLabel(names=['not_equivalent', 'equivalent'], id=None), 'idx': Value(dtype='int32', id=None)}
print(raw_datasets["train"].features)
```

预处理数据集。

BERT是带着token_type_id预训练的。
BERT的目标函数里带了mask，以及next sentence prediction特性。
下个句子预测指的是，提供一对句子（随机mask tokens），让它预测第二个句子是否跟随第一个句子。
同时，保证一半正样本一半负样本。

tokenizer背后是用Rust实现的，效率非常高。

```python
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# tokenizer通过拼接的方式来处理一对句子，同时还会增加token_type_ids区分它是第一句话还是第二句话
inputs = tokenizer("This is the first sentence.", "This is the second one.")
tokenizer(raw_datasets["train"][15]["sentence1"], raw_datasets["train"][15]["sentence2"])

# 它的token_type_ids是全0向量
tokenizer(raw_datasets["train"][15]["sentence1"])

# tokernizer允许将token id映射回单词
tokenizer.convert_ids_to_tokens(inputs["input_ids"])

# [CLS] sentence1 [SEP]的token_type_id为0、sentence2 [SEP]的token_type_id为1
print("\n".join([f"{t} {i}" for t, i in zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"]), inputs["token_type_ids"])]))

# 该朴素方法将整个数据集载入内存并做分词
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

def tokenize_function(example):
    print(type(example))
    # 注意到这里没有padding，也没有返回PyTorch张量，所以返回的是Python List，不知有没有性能问题
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Hugging Face Dataset的数据集是存在磁盘上的Apache Arrow数据
# 通过这种方法可以流式做Tokenization，省内存
# 若batched=True，则传入udf的样本是一批数据而不是一个数据，在这里是datasets.formatting.formatting.LazyBatch
# tokenizer背后是用Rust实现的，效率非常高
# 同时还会自动施加多进程并行
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=16)

# 可见行数没有变，但是新增了三个列，分别是input_ids、token_type_ids、attention_mask
print(raw_datasets)
print(tokenized_datasets)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# 可见每个样本的长度都不一样，而且类型全是list
# [50, 59, 47, 67, 59, 50, 62, 32]
[len(x) for x in samples["input_ids"]]
[type(x) for x in samples["input_ids"]]

# 可见data_collator将一批样本整理成了能喂给PyTorch的形式，该padding的做padding，该做转换的做转换
# {'input_ids': torch.Size([8, 67]), 'token_type_ids': torch.Size([8, 67]), 'attention_mask': torch.Size([8, 67]), 'labels': torch.Size([8])}
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

### 使用Trainer API或者Keras来微调模型

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

# 提示以下信息，表明用来做序列分类的head没有提供参数，需要进一步训练这个模型来做下游任务
# Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

## 可见一个进度条在滚动，预计总时间是18分钟（这是4个CPU核心，r7i.2xlarge）。而tutorial上说，在GPU上只需要两三分钟。
##  注意：07:17<01:02。前者表示已用时间。后者表示剩余时间。后者不是表示总时间。
## 换成r7i.8xlarge之后，预计总时间是7分钟，扩展性没有那么好，吞吐量从1.23 it/s增加到2.85 it/s，只翻了2倍。
## 每500步会报告一次training loss，不过不会告诉你这个模型表现得好不好。因为没有设置evaluation_strategy，它可以是steps或epoch。
##   {'loss': 0.4811, 'grad_norm': 19.382593154907227, 'learning_rate': 3.184458968772695e-05, 'epoch': 1.09}
##   {'loss': 0.2325, 'grad_norm': 0.03515595570206642, 'learning_rate': 1.3689179375453886e-05, 'epoch': 2.18}
##   TrainOutput(global_step=1377, training_loss=0.287335001738416, metrics={'train_runtime': 499.6016, 'train_samples_per_second': 22.026, 'train_steps_per_second': 2.756, 'total_flos': 405114969714960.0, 'train_loss': 0.287335001738416, 'epoch': 3.0})
##   可见FLOPS为810 GFlops
trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])

# (408, 2) (408,)
print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

# 可以拿到MRPC任务伴随的评估测度
metric = evaluate.load("glue", "mrpc")

# 根据实际的预测与Ground Truth比较，得到验证集上85.54%的精度以及F1 score分数89.88。
# {'accuracy': 0.8553921568627451, 'f1': 0.8987993138936535}
metric.compute(predictions=preds, references=predictions.label_ids)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 带入compute_metrics并指定epoch的evaluation_strategy，每个epoch会汇报测度
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# {'eval_loss': 0.3872298002243042, 'eval_accuracy': 0.8357843137254902, 'eval_f1': 0.8850771869639794, 'eval_runtime': 4.52, 'eval_samples_per_second': 90.265, 'eval_steps_per_second': 11.283, 'epoch': 1.0}
# {'loss': 0.5399, 'grad_norm': 17.776399612426758, 'learning_rate': 3.184458968772695e-05, 'epoch': 1.09}
# {'eval_loss': 0.4077554941177368, 'eval_accuracy': 0.8578431372549019, 'eval_f1': 0.897887323943662, 'eval_runtime': 4.5333, 'eval_samples_per_second': 90.001, 'eval_steps_per_second': 11.25, 'epoch': 2.0}
# {'loss': 0.3295, 'grad_norm': 0.09225674718618393, 'learning_rate': 1.3689179375453886e-05, 'epoch': 2.18}
# {'eval_loss': 0.7114211916923523, 'eval_accuracy': 0.8455882352941176, 'eval_f1': 0.8941176470588236, 'eval_runtime': 4.5421, 'eval_samples_per_second': 89.827, 'eval_steps_per_second': 11.228, 'epoch': 3.0}
# {'train_runtime': 526.6296, 'train_samples_per_second': 20.895, 'train_steps_per_second': 2.615, 'train_loss': 0.3660710043204203, 'epoch': 3.0}
# TrainOutput(global_step=1377, training_loss=0.3660710043204203, metrics={'train_runtime': 526.6296, 'train_samples_per_second': 20.895, 'train_steps_per_second': 2.615, 'total_flos': 405114969714960.0, 'train_loss': 0.3660710043204203, 'epoch': 3.0})
trainer.train()
```

### 不使用Trainer类，看看训练过程中发生了什么

涉及到的内部实现：
1. 需要对tokenized_datasets进行一些后处理
2. DataLoader用于加载数据

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 删掉与训练无关的列
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])

# 模型要求标签列名字是labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 将数据集的格式转换为PyTorch张量
tokenized_datasets.set_format("torch")

# ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
tokenized_datasets["train"].column_names

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break


# 检验数据正确性，可见一批数据有8个样本，最大长度是69
# {'labels': torch.Size([8]), 'input_ids': torch.Size([8, 69]), 'token_type_ids': torch.Size([8, 69]), 'attention_mask': torch.Size([8, 69])}
{k: v.shape for k, v in batch.items()}

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 验证训练过程是不是能如预期进行
outputs = model(**batch)

# tensor(1.1463, grad_fn=<NllLossBackward0>) torch.Size([8, 2])
print(outputs.loss, outputs.logits.shape)

# Trainer默认使用AdamW优化器，和Adam一样，但是使用了不一样的权重衰减正则化方法。
from transformers import AdamW

# FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler
num_epochs = 3

# len(train_dataloader)指的是有多少批数据会过来
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 将数据移到相应设备上
        batch = {k: v.to(device) for k, v in batch.items()}

        # 做推理
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])


metric.compute()
```

若使用Accelerator：

```python
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

# 将会读取配置并初始化分布式训练环境
accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        print("Step")


#<class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>
type(model)
```

同时还可以把脚本放在外面，并且调用`accelerate launch train.py`，可以用来做分布式训练。

