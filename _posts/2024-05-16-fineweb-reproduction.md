# 介绍

FineWeb包含超过15TB token的清洗并删冗后的英语Web数据集。
是RefinedWeb的开源复现，RefinedWeb数据集发了论文但是未开源。
数据处理管线面向LLM性能优化。
这个数据集接入了`datatrove`库，
他们开发的大规模数据处理库。

# 复现FineWeb

```bash
sudo apt install aws
aws s3 mb s3://fineweb-reproduction-20240516

python3 -m venv pyvenv-fineweb
source pyvenv-fineweb/bin/activate
pip install datatrove[io,processing,s3]
wget https://raw.githubusercontent.com/huggingface/datatrove/main/examples/process_common_crawl_dump.py
sed -i 's#s3://some_s3_bucket/base_processing/#s3://fineweb-reproduction-20240516/base_processing/#g' process_common_crawl_dump.py
sed -i 's#from datatrove.executor.slurm import SlurmPipelineExecutor#from datatrove.executor import LocalPipelineExecutor#g' process_common_crawl_dump.py
sed -i 's#SlurmPipelineExecutor#LocalPipelineExecutor#g' process_common_crawl_dump.py
sed -i 's#Trafilatura(favour_precision=True)#Trafilatura(favour_precision=True, timeout=10.0)#g' process_common_crawl_dump.py
python -c "import nltk; nltk.download('punkt')"
python process_common_crawl_dump.py CC-MAIN-2020-10
aws s3 ls s3://fineweb-reproduction-20240516/base_processing/
```

# 相关开源数据集

1. [RefinedWeb](https://huggingface.co/papers/2306.01116)：未开源
2. [C4](https://www.tensorflow.org/datasets/catalog/c4)：谷歌基于Common Crawl清洗出来的数据集
3. 

# 相关开源库

1. [datatrove](https://github.com/huggingface/datatrove)：Hugging Face官方推出的基于Slurm的数据处理库。
2. 

# Hugging Face官方的库

1. [transformers](https://github.com/huggingface/transformers)：业界领先的面向JAX、PyTorch、TensorFlow的机器学习。
2. [diffusers](https://github.com/huggingface/diffusers)：业界领先的用于图像、音频生成的模型（基于PyTorch和FLAX）。
3. [dataset](https://github.com/huggingface/datasets)：面向机器学习训练的数据集，以及易用、高效的数据处理工具。
4. [peft](https://github.com/huggingface/peft)：业界领先的参数高效的微调，PEFT方法只需要fine-tuning一小部分模型参数，而不是所有的模型参数。
5. [accelerate](https://github.com/huggingface/accelerate)：运行、训练、使用PyTorch模型在任何设备、分布式配置下。自动施加混合精度。
6. [optimum](https://github.com/huggingface/optimum)：加速Transformer与Diffusers的训练与推理，面向各种硬件优化工具。

