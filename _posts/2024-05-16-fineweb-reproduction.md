# 介绍

FineWeb包含超过15TB token的清洗并删冗后的英语Web数据集。
是RefinedWeb的开源复现，RefinedWeb数据集发了论文但是未开源。
数据处理管线面向LLM性能优化。
这个数据集接入了`datatrove`库，
他们开发的大规模数据处理库。

# 复现

## 分析Common Crawl

总共有99个Archive Crawl，每个Archive Crawl有100个Segment，因此共计有9900个Segment。
对于WARC，每个Segment大约有800个文件，每个文件用gz算法压缩，大约每个压缩后有1.2GB。

因此整个Common Crawl有8910000个文件，按照每个文件经过fineweb清洗后的大小为18MB，总大小为160MB，离15T还差很多。

```bash
for s in $(aws s3 ls s3://commoncrawl/crawl-data/ | grep CC-MAIN- | tr -s ' ' | cut -d ' ' -f 3 | sed 's#/##g'); do
    echo $s
    aws s3 cp s3://commoncrawl/crawl-data/CC-MAIN-2023-23/segment.paths.gz ${s}_segment.paths.gz
done
gunzip *.paths.gz
cat *.paths | wc -l
```

## 复现datatrove/process_common_crawl_dump

注意：使用S3复现失败，bogo3报错API调用频率太高。而且写入S3会看不到中间结果，本地模式可以看到输出的中间结果。

```bash
sudo apt install aws
# aws s3 mb s3://fineweb-reproduction-20240516
mkdir -p /root/fineweb/crawl-data/CC-MAIN-2023-23/segments/1685224653183.5/warc
cd /root/fineweb/crawl-data/CC-MAIN-2023-23/segments/1685224653183.5/warc
# aws s3 sync s3://commoncrawl/crawl-data/CC-MAIN-2023-23/segments/1685224653183.5/warc .
aws s3 cp s3://commoncrawl/crawl-data/CC-MAIN-2023-23/segments/1685224653183.5/warc/CC-MAIN-20230606214755-20230607004755-00000.warc.gz .

python3 -m venv pyvenv-fineweb
source pyvenv-fineweb/bin/activate
pip install datatrove[io,processing,s3]
wget https://raw.githubusercontent.com/huggingface/datatrove/main/examples/process_common_crawl_dump.py
sed -i 's#s3://some_s3_bucket/base_processing/#/root/fineweb/base_processing/#g' process_common_crawl_dump.py
sed -i 's#from datatrove.executor.slurm import SlurmPipelineExecutor#from datatrove.executor import LocalPipelineExecutor#g' process_common_crawl_dump.py
sed -i 's#SlurmPipelineExecutor#LocalPipelineExecutor#g' process_common_crawl_dump.py
sed -i 's#Trafilatura(favour_precision=True)#Trafilatura(favour_precision=True, timeout=10.0)#g' process_common_crawl_dump.py
python -c "import nltk; nltk.download('punkt')"
python process_common_crawl_dump.py CC-MAIN-2023-23
# aws s3 ls s3://fineweb-reproduction-20240516/base_processing/
```

结果如下：
1. 可见瓶颈在Trafilatura上，占据了91%的时间。因为Trafilatura过滤掉了大量文本，输入文本大小为4.72GB，而输出文本大小仅为86MB，仅保留1.83%的文本。
2. 处理1.2GB压缩后的CC数据集，这是CC-MAIN-2023-23的Crawl Archive（共有99个Archive）下的1685224653183.5的Segment（共有100个Segment），单个线程，用时1156秒，也就是1.06MB/s。

```
2024-05-17 06:39:59.728 | SUCCESS  | datatrove.executor.local:run:146 -

📉📉📉 Stats: All 1 tasks 📉📉📉

Total Runtime: 19 minutes and 16 seconds

📖 - READER: 🕷 Warc
    Runtime: (2.09%) 24 seconds [0.20 milliseconds±0.66 milliseconds/doc]
    Stats: {input_files: 1, doc_len: 4740085439 [min=1, max=1048576, 138749.10±183114/doc], documents: 34162 [34162.00/input_file]}
🔻 - FILTER: 😈 Url-filter
    Runtime: (0.49%) 5 seconds [0.17 milliseconds±11.14 milliseconds/doc]
    Stats: {total: 34163, forwarded: 33963, doc_len: 4720996535 [min=1, max=1048576, 139004.11±183434/doc], dropped: 200, dropped_domain: 101, dropped_hard_blacklisted: 75, dropped_blacklisted_subword: 17, dropped_soft_blacklisted: 6, dropped_subdomain: 1}
🛢 - EXTRAC: ⛏ Trafilatura
    Runtime: (91.08%) 17 minutes and 33 seconds [31.02 milliseconds±58.37 milliseconds/doc]
    Stats: {total: 33963, forwarded: 32550, doc_len: 86438862 [min=1, max=601304, 2655.57±13256/doc], dropped: 1413}
🔻 - FILTER: 🌍 Language ID
    Runtime: (2.01%) 23 seconds [0.71 milliseconds±2.47 milliseconds/doc]
    Stats: {total: 32550, dropped: 20766, forwarded: 11784, doc_len: 30116540 [min=8, max=292206, 2555.71±6959/doc]}
🔻 - FILTER: 👯 Gopher Repetition
    Runtime: (2.70%) 31 seconds [2.65 milliseconds±6.70 milliseconds/doc]
    Stats: {total: 11784, forwarded: 8545, doc_len: 21903496 [min=8, max=131305, 2563.31±4958/doc], dropped: 3239, dropped_dup_line_frac: 1387, dropped_duplicated_5_n_grams: 336, dropped_top_3_gram: 210, dropped_top_2_gram: 694, dropped_duplicated_6_n_grams: 24, dropped_duplicated_9_n_grams: 16, dropped_top_4_gram: 383, dropped_duplicated_10_n_grams: 27, dropped_dup_line_char_frac: 133, dropped_duplicated_7_n_grams: 14, dropped_duplicated_8_n_grams: 15}
🔻 - FILTER: 🥇 Gopher Quality
    Runtime: (1.44%) 16 seconds [1.95 milliseconds±3.32 milliseconds/doc]
    Stats: {total: 8545, forwarded: 6071, doc_len: 19018684 [min=254, max=131305, 3132.71±5443/doc], dropped: 2474, dropped_gopher_too_many_end_ellipsis: 240, dropped_gopher_short_doc: 1095, dropped_gopher_below_alpha_threshold: 1108, dropped_gopher_enough_stop_words: 20, dropped_gopher_too_many_ellipsis: 2, dropped_gopher_above_avg_threshold: 1, dropped_gopher_too_many_hashes: 2, dropped_gopher_too_many_bullets: 3, dropped_gopher_below_avg_threshold: 3}
💽 - WRITER: 🐿 Jsonl
    Runtime: (0.18%) 2 seconds [0.34 milliseconds±0.46 milliseconds/doc]
    Stats: {XXXXX.jsonl.gz: 6071, total: 6071, doc_len: 19018684 [min=254, max=131305, 3132.71±5443/doc]}
```

## 复现Fineweb

```bash
wget https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
sed -i 's/#.*//g' fineweb.py
sed -i 's#from datatrove.executor.slurm import SlurmPipelineExecutor#from datatrove.executor import LocalPipelineExecutor#g' fineweb.py
sed -i 's#CC-MAIN-2O23-5O#CC-MAIN-2023-23#g' fineweb.py
sed -i 's#s3://some_s3_bucket#/root/fineweb#g' fineweb.py
sed -i 's#SlurmPipelineExecutor#LocalPipelineExecutor#g' fineweb.py
sed -i 's#job_name=.*,##g' fineweb.py
sed -i 's#s3://commoncrawl/crawl-data#/root/fineweb/crawl-data#g' fineweb.py
sed -i 's#Trafilatura(favour_precision=True)#Trafilatura(favour_precision=True, timeout=10.0)#g' fineweb.py
sed -i 's#tasks=.*,#tasks=1,#g' fineweb.py
sed -i 's#time=".*",##g' fineweb.py
sed -i 's#slurm_logs_folder=.*",##g' fineweb.py
sed -i 's#randomize_start=.*,##g' fineweb.py
sed -i 's#mem_per_cpu_gb=.*,##g' fineweb.py
sed -i 's#partition=.*,##g' fineweb.py
sed -i 's#logging_dir=.*,##g' fineweb.py
sed -i 's#cpus_per_task=.*,##g' fineweb.py
sed -i 's#depends=.*,##g' fineweb.py
sed -i 's#^stage4.run()#stage1.run(); stage2.run(); stage3.run(); stage4.run()#g' fineweb.py
python -c "import nltk; nltk.download('punkt')"
/usr/bin/time -v python fineweb.py 2>&1 | tee fineweb.log
```

结果如下

```
--- 🛠️ PIPRLINEW🛠                                                                                                  [66/1812]
📖 - READER: 🕷 Warc                                                                                                         🔻 - FILTER: 😈 Url-filter                                                                                                  🛢 - EXTRAC: ⛏ Trafilatura
🔻 - FILTER: 🌍 Language ID
🔻 - FILTER: 👯 Gopher Repetition                                                                                           🔻 - FILTER: 🥇 Gopher Quality                                                                                              🔻 - FILTER: ⛰ C4 Quality                                                                                                   🔻 - FILTER: 🍷 FineWeb Quality
💽 - WRITER: 🐿 Jsonl
2024-05-17 07:13:38.553 | INFO     | datatrove.pipeline.readers.base:read_files_shard:193 - Reading input file 1685224653183.5/warc/CC-MAIN-20230606214755-20230607004755-00000.warc.gz                                                                 2024-05-17 07:13:43.012 | WARNING  | datatrove.pipeline.readers.base:get_document_from_dict:93 - Found document without text, skipping. Is your `text_key` ("text") correct?                                                                            2024-05-17 07:33:20.415 | SUCCESS  | datatrove.executor.base:_run_for_rank:85 - Processing done for rank=0
2024-05-17 07:33:20.417 | INFO     | datatrove.executor.base:_run_for_rank:91 -
                                                                                                                            📉📉📉 Stats: Task 0 📉📉📉
                                                                                                                            Total Runtime: 19 minutes and 27 seconds
                                                                                                                            📖 - READER: 🕷 Warc
    Runtime: (2.08%) 24 seconds [0.21 milliseconds±0.66 milliseconds/doc]
    Stats: {input_files: 1, doc_len: 4740085439 [min=1, max=1048576, 138749.10±183114/doc], documents: 34162 [34162.00/input_file]}                                                                                                                     🔻 - FILTER: 😈 Url-filter                                                                                                      Runtime: (0.49%) 5 seconds [0.17 milliseconds±11.06 milliseconds/doc]
    Stats: {total: 34163, forwarded: 33963, doc_len: 4720996535 [min=1, max=1048576, 139004.11±183434/doc], dropped: 200, dropped_domain: 101, dropped_hard_blacklisted: 75, dropped_blacklisted_subword: 17, dropped_soft_blacklisted: 6, dropped_subdomain: 1}                                                                                                                    🛢 - EXTRAC: ⛏ Trafilatura                                                                                                       Runtime: (90.03%) 17 minutes and 31 seconds [30.96 milliseconds±57.26 milliseconds/doc]
    Stats: {total: 33963, forwarded: 32550, doc_len: 86438862 [min=1, max=601304, 2655.57±13256/doc], dropped: 1413}
🔻 - FILTER: 🌍 Language ID                                                                                                     Runtime: (1.99%) 23 seconds [0.72 milliseconds±2.44 milliseconds/doc]                                                       Stats: {total: 32550, dropped: 20766, forwarded: 11784, doc_len: 30116540 [min=8, max=292206, 2555.71±6959/doc]}
🔻 - FILTER: 👯 Gopher Repetition
    Runtime: (2.66%) 31 seconds [2.64 milliseconds±6.71 milliseconds/doc]
    Stats: {total: 11784, forwarded: 8545, doc_len: 21903496 [min=8, max=131305, 2563.31±4958/doc], dropped: 3239, dropped_dup_line_frac: 1387, dropped_duplicated_5_n_grams: 336, dropped_top_3_gram: 210, dropped_top_2_gram: 694, dropped_duplicated_6_n_grams: 24, dropped_duplicated_9_n_grams: 16, dropped_top_4_gram: 383, dropped_duplicated_10_n_grams: 27, dropped_dup_line_char_frac: 133, dropped_duplicated_7_n_grams: 14, dropped_duplicated_8_n_grams: 15}
🔻 - FILTER: 🥇 Gopher Quality                                                                                                  Runtime: (1.42%) 16 seconds [1.94 milliseconds±3.30 milliseconds/doc]
    Stats: {total: 8545, forwarded: 6071, doc_len: 19018684 [min=254, max=131305, 3132.71±5443/doc], dropped: 2474, dropped_gopher_too_many_end_ellipsis: 240, dropped_gopher_short_doc: 1095, dropped_gopher_below_alpha_threshold: 1108, dropped_gopher_enough_stop_words: 20, dropped_gopher_too_many_ellipsis: 2, dropped_gopher_above_avg_threshold: 1, dropped_gopher_too_many_hashes: 2, dropped_gopher_too_many_bullets: 3, dropped_gopher_below_avg_threshold: 3}
🔻 - FILTER: ⛰ C4 Quality
    Runtime: (0.32%) 3 seconds [0.62 milliseconds±1.02 milliseconds/doc]
    Stats: {total: 6071, line-total: 123281, line-kept: 97170, dropped: 602, dropped_too_few_sentences: 573, line-filter-too_few_words: 25349, forwarded: 5469, doc_len: 17992639 [min=149, max=123528, 3289.93±5383/doc], line-filter-policy: 641, dropped_curly_bracket: 22, line-filter-javascript: 92, dropped_lorem_ipsum: 7}
🔻 - FILTER: 🍷 FineWeb Quality
    Runtime: (0.85%) 9 seconds [1.81 milliseconds±2.58 milliseconds/doc]
    Stats: {total: 5469, dropped: 647, dropped_line_punct_ratio: 336, forwarded: 4822, doc_len: 15898156 [min=151, max=99375, 3297.00±4732/doc], dropped_short_line_ratio: 86, dropped_char_dup_ratio: 225}
💽 - WRITER: 🐿 Jsonl
    Runtime: (0.15%) 1 second [0.37 milliseconds±0.45 milliseconds/doc]
    Stats: {XXXXX.jsonl.gz: 4822, total: 4822, doc_len: 15898156 [min=151, max=99375, 3297.00±4732/doc]}
2024-05-17 07:33:20.423 | SUCCESS  | datatrove.executor.local:run:146 -

📉📉📉 Stats: All 1 tasks 📉📉📉

Total Runtime: 19 minutes and 27 seconds

        Command being timed: "python fineweb.py"
        User time (seconds): 1177.85
        System time (seconds): 2.05
        Percent of CPU this job got: 99%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 19:42.98
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 1029440
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 1
        Minor (reclaiming a frame) page faults: 342151
        Voluntary context switches: 71803
        Involuntary context switches: 3084
        Swaps: 0
        File system inputs: 0
        File system outputs: 84208
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 1
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

