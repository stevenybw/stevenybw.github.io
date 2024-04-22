---
layout: post
title:  "Optimizing LLM Queries in Relational Workloads"
date:   2024-04-22 16:25:00 +0800
categories: paper_reading
---
# 来源

https://arxiv.org/pdf/2403.05821.pdf

# 随手笔记

## 摘要

1. 分析数据库提供商（Redshift、Databricks、BigQuery）都允许用户在SQL里调用LLM来进行分类、实体提取、翻译等。
2. 例如分析师可以从百万条产品评价文本中提取出顾客情感。
3. 但是LLM推理非常昂贵，一块`NVIDIA L4 GPU`运行`Llama2-7B`每秒只能处理`6KB`文本。
4. 本文探索如何优化上述场景，并发现加速机会非常大，包括重排。
