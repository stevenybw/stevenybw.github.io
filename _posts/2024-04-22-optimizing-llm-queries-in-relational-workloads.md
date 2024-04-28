---
layout: post
title:  "Optimizing LLM Queries in Relational Workloads"
date:   2024-04-22 16:25:00 +0800
categories: paper_reading
---
# 来源

https://arxiv.org/pdf/2403.05821.pdf

# 随手笔记

## 摘要翻译

1. 分析数据库提供商（Redshift、Databricks、BigQuery）都允许用户在SQL里调用LLM来进行分类、实体提取、翻译等。
2. 例如分析师可以从百万条产品评价文本中提取出顾客情感。
3. 但是LLM推理非常昂贵，一块`NVIDIA L4 GPU`运行`Llama2-7B`每秒只能处理`6KB`文本。
4. 本文探索如何优化上述场景，并发现加速机会非常大，包括**对行进行重排以在LLM推理引擎中最大化KV cache复用**、**重排列以进一步提升缓存复用**、**删除冗余的推理请求**。
5. 本文在Apache Spark中实现了这些优化，使用`vLLM`作为模型推理引擎，在真实数据的LLM测试程序上达到了4.4倍的端到端加速比。

## 引言

LLM极大简化了文本分析，如下图所示。

```sql
SELECT user, request, support_response, LLM("Did {support_response} address {request}?", support_response, request) AS success
FROM customer_tickets
WHERE support_response <> NULL
```

但是成本很高，例如从`Rotten Tomatoes Movies `公开数据集上对1.5万行影评进行分类需要30分钟的`NVIDIA L4 GPU`节点.

为了降低成本，请求被攒成批。但是批量做推理给内存管理带来了挑战。推理引擎为前缀存储中间状态，在一个KV缓存里。
**复用前缀**可以显著提升性能。
当前推理系统致力于最大化KV缓存的命中率。

但是当前的系统主要是面向**在线负载**，失去了一些优化机会。对于离线负载会有更多优化机会。

提出**前缀共享最大化**提高命中率。

主要挑战是找到最优的顺序和格式。
本文采用一种启发式方法来做列重排。

同时，还删冗、基于代价模型去优化。

基于`Apache Spark` + `vLLM`实现了上述方法。
由于缺少标准负载，本文自己构建了测试集。
包含：多LLM调用查询、RAG查询。

## 背景与动机

LLM是自回归Transformer模型，基于一个prompt以及已经生成的token预测下一个token。
token是一组字符的表示，常用[Byte-Paired Encoding](https://huggingface.co/learn/nlp-course/en/chapter6/5)，平均4个英语字符。

推理过程有两个阶段：
1. prefill：模型一次性处理所有输入prompt
2. decoding：开始生成token

因为文本生成过程本身是串行的，为了增广并行度，通常会**攒成一批**。
**对于Online负载，通常也能搞一个32的批**。

批越大内存需求越大。

KV Cache会需要较高的内存。
对于13B的模型，每个token大概需要800KB内存。
平均请求token数为2000，需要1.6GB的内存。

为了节省内存，近期工作提出跨越请求地共享token。
考虑到大量用户请求采用了同一个prompt模板，前缀都一样，跨越请求共享token是成立的。
但是当前工作仅关注Online负载，复用空间有限。
SQL场景下有广阔的复用空间。

### 分析场景的优化机会

**提升KV Cache命中率**：离线场景下有全部数据，不局限于在线场景只能关注当下。

**避免重复计算**：精确删冗。

**基于代价模型的优化**：为LLM操作建立性能模型，整合进CBO优化器。

## 请求构造

### 列重排

案例：LLM("Give an overview based on this: ", *)

算法：利用列值的Cardinality。
1. 为每一列预计算平均字符串长度、Cardinality
2. 根据以下公式计算分数：平均字符串长度 * 总行数 / Cardinality
3. 按分数降序排序

### 行重排

算法
1. 构造出最终的字符串
2. 按照字典序排序

### 删冗

采用`DISTINCT`精确删冗

提出语义删冗可能会更好

# 背景知识

## BPE

参见：https://huggingface.co/learn/nlp-course/en/chapter6/5

算法过程：