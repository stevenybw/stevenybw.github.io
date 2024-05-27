# 介绍

FineWeb包含超过15TB token的清洗并删冗后的英语Web数据集。
是RefinedWeb的开源复现，RefinedWeb数据集发了论文但是未开源。
数据处理管线面向LLM性能优化。
这个数据集接入了`datatrove`库，
他们开发的大规模数据处理库。

## 与LLaMA 3的关系

参见[Introducing Meta Llama 3: The most capable openly available LLM to date
](https://ai.meta.com/blog/meta-llama-3/)，其中提到它们从各种公开数据源收集数据，通过一系列过滤管线（启发式过滤器、NSFW过滤器、语义删冗、文本质量分类器）将数据清洗。特别提到，**用于训练文本质量分类器的训练数据由Llama 2模型生成）**。

ASIDE：在Instruction Fine-tuning阶段，通过SFT（Supervised Fine-tuning）、拒绝采样（Rejection Sampling）、PPO（Proximal Policy Optimization）、DPO（Direct Preference Optimization）等技术来做后训练处理。

# 复现

## 分析Common Crawl

总共有99个Archive Crawl，每个Archive Crawl有100个Segment，因此共计有9900个Segment。
对于WARC，每个Segment大约有800个文件，每个文件用gz算法压缩，大约每个压缩后有1.2GB。

因此整个Common Crawl有8910000个文件，按照每个文件经过fineweb清洗后的大小为18MB，总大小为160TB，和15T Token差不多能对上了。

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
📖 - READER: 🕷 Warc                                                                                                         
🔻 - FILTER: 😈 Url-filter                                                                                                  
🛢 - EXTRAC: ⛏ Trafilatura
🔻 - FILTER: 🌍 Language ID
🔻 - FILTER: 👯 Gopher Repetition                                                                                           
🔻 - FILTER: 🥇 Gopher Quality                                                                                              
🔻 - FILTER: ⛰ C4 Quality                                                                                                   
🔻 - FILTER: 🍷 FineWeb Quality
💽 - WRITER: 🐿 Jsonl
2024-05-17 07:13:38.553 | INFO     | datatrove.pipeline.readers.base:read_files_shard:193 - Reading input file 1685224653183.5/warc/CC-MAIN-20230606214755-20230607004755-00000.warc.gz                                                                 
2024-05-17 07:13:43.012 | WARNING  | datatrove.pipeline.readers.base:get_document_from_dict:93 - Found document without text, skipping. Is your `text_key` ("text") correct?                                                                            
2024-05-17 07:33:20.415 | SUCCESS  | datatrove.executor.base:_run_for_rank:85 - Processing done for rank=0
2024-05-17 07:33:20.417 | INFO     | datatrove.executor.base:_run_for_rank:91 -
                                                                                                                            📉📉📉 Stats: Task 0 📉📉📉
                                                                                                                            Total Runtime: 19 minutes and 27 seconds
                                                                                                                            
📖 - READER: 🕷 Warc
    Runtime: (2.08%) 24 seconds [0.21 milliseconds±0.66 milliseconds/doc]
    Stats: {input_files: 1, doc_len: 4740085439 [min=1, max=1048576, 138749.10±183114/doc], documents: 34162 [34162.00/input_file]}                                                                                                                     
🔻 - FILTER: 😈 Url-filter                                                                                                      
    Runtime: (0.49%) 5 seconds [0.17 milliseconds±11.06 milliseconds/doc]
    Stats: {total: 34163, forwarded: 33963, doc_len: 4720996535 [min=1, max=1048576, 139004.11±183434/doc], dropped: 200, dropped_domain: 101, dropped_hard_blacklisted: 75, dropped_blacklisted_subword: 17, dropped_soft_blacklisted: 6, dropped_subdomain: 1}                                                                                                                    
🛢 - EXTRAC: ⛏ Trafilatura                                                                                                       
    Runtime: (90.03%) 17 minutes and 31 seconds [30.96 milliseconds±57.26 milliseconds/doc]
    Stats: {total: 33963, forwarded: 32550, doc_len: 86438862 [min=1, max=601304, 2655.57±13256/doc], dropped: 1413}
🔻 - FILTER: 🌍 Language ID                                                                                                     
    Runtime: (1.99%) 23 seconds [0.72 milliseconds±2.44 milliseconds/doc]                                                       
    Stats: {total: 32550, dropped: 20766, forwarded: 11784, doc_len: 30116540 [min=8, max=292206, 2555.71±6959/doc]}
🔻 - FILTER: 👯 Gopher Repetition
    Runtime: (2.66%) 31 seconds [2.64 milliseconds±6.71 milliseconds/doc]
    Stats: {total: 11784, forwarded: 8545, doc_len: 21903496 [min=8, max=131305, 2563.31±4958/doc], dropped: 3239, dropped_dup_line_frac: 1387, dropped_duplicated_5_n_grams: 336, dropped_top_3_gram: 210, dropped_top_2_gram: 694, dropped_duplicated_6_n_grams: 24, dropped_duplicated_9_n_grams: 16, dropped_top_4_gram: 383, dropped_duplicated_10_n_grams: 27, dropped_dup_line_char_frac: 133, dropped_duplicated_7_n_grams: 14, dropped_duplicated_8_n_grams: 15}
🔻 - FILTER: 🥇 Gopher Quality                                                                                                  
    Runtime: (1.42%) 16 seconds [1.94 milliseconds±3.30 milliseconds/doc]
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

# 性能分析

给`WarcReader`加一个`limit=1000`即可在26秒内出结果。

```bash
python -m cProfile -o process_common_crawl_dump.cProfile -s cumtime process_common_crawl_dump.py CC-MAIN-2023-23
python -m pstats process_common_crawl_dump.cProfile
```

看上去cPython对多线程的支持有限，[extractor使用了ThreadPoolExecutor来做基于多线程的线程池](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/extractors/base.py#L48)。

尝试`yaapi`，可以自底向上看到具体哪些调用是热点。不过它存在的一个问题是，在使用yappi对Cython模块进行分析时，Cython函数的调用统计通常会归入调用它们的Python函数。

```bash
pip install yappi
python -m yappi -c cpu -o process_common_crawl_dump.yappi -f pstat process_common_crawl_dump.py CC-MAIN-2023-23
python -m pstats process_common_crawl_dump.yappi
# sort tottime
# stats 100

# 若没有-b，则不会将builtin函数考虑进去，所以打开-b进一步看看哪些builtin是最需要优化的
python -m yappi -c cpu -b -o process_common_crawl_dump.yappi.withBuiltins -f pstat process_common_crawl_dump.py CC-MAIN-2023-23
python -m pstats process_common_crawl_dump.yappi.withBuiltins
# sort tottime
# stats 100

# 做全量数据分析
rm -rf logging; python -m yappi -c cpu -b -o process_common_crawl_dump-FULL.yappi -f pstat process_common_crawl_dump.py CC-MAIN-2023-23; rm -rf logging; py-spy record -o process_common_crawl_dump-FULL.pyspy.svg -f flamegraph --rate 100 -- python process_common_crawl_dump.py CC-MAIN-2023-23
python -m pstats process_common_crawl_dump-FULL.yappi
# sort tottime
# stats 100
```

使用`py-spy`画火焰图，可以自顶向下做性能分析。[py-spy默认是CPU模式，在加上--idle命令之后是Wall模式](https://github.com/benfred/py-spy/issues/458)，我们先看看CPU模式：


```bash
pip install py-spy
py-spy record -o process_common_crawl_dump.pyspy.svg -f flamegraph --rate 100 -- python process_common_crawl_dump.py CC-MAIN-2023-23
```

尝试`line_profiler`：

```bash
pip install line_profiler
vi pyvenv-fineweb/lib/python3.10/site-packages/trafilatura/htmlprocessing.py
# 在def prune_unwanted_nodes之前增加 @line_profiler.profile
LINE_PROFILE=1 python process_common_crawl_dump.py CC-MAIN-2023-23
python -m line_profiler -rtmz profile_output.lprof
```

输出结果：
```
 11.13 seconds - /root/pyvenv-fineweb/lib/python3.10/site-packages/trafilatura/htmlprocessing.py:90 - prune_unwanted_nodes
Wrote profile results to profile_output.txt
Wrote profile results to profile_output_2024-05-19T081609.txt
Wrote profile results to profile_output.lprof
To view details run:
python -m line_profiler -rtmz profile_output.lprof
```

进一步查看
```
Timer unit: 1e-06 s

Total time: 11.1257 s
File: /root/pyvenv-fineweb/lib/python3.10/site-packages/trafilatura/htmlprocessing.py
Function: prune_unwanted_nodes at line 90

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    90                                           @line_profiler.profile
    91                                           def prune_unwanted_nodes(tree, nodelist, with_backup=False):
    92                                               '''Prune the HTML tree by removing unwanted sections.'''
    93     11665       3130.6      0.3      0.0      if with_backup:
    94      1577      25789.5     16.4      0.2          old_len = len(tree.text_content())  # ' '.join(tree.itertext())
    95      1577      77741.7     49.3      0.7          backup = deepcopy(tree)
    96                                           
    97     27462       6424.3      0.2      0.1      for expression in nodelist:
    98     54290   10884183.9    200.5     97.8          for subtree in expression(tree):
    99                                                       # preserve tail text from deletion
   100     38493       8026.8      0.2      0.1              if subtree.tail is not None:
   101     28309      10939.3      0.4      0.1                  prev = subtree.getprevious()
   102     28309       3699.4      0.1      0.0                  if prev is None:
   103     17852       6883.3      0.4      0.1                      prev = subtree.getparent()
   104     28309       3616.2      0.1      0.0                  if prev is not None:
   105                                                               # There is a previous node, append text to its tail
   106     28309      23310.3      0.8      0.2                      prev.tail = " ".join([prev.tail, subtree.tail]) if prev.tail else subtree.tail
   107                                                       # remove the node
   108     38493      52627.2      1.4      0.5              subtree.getparent().remove(subtree)
   109                                           
   110     11665       1707.4      0.1      0.0      if not with_backup:
   111     10088       1384.7      0.1      0.0          return tree
   112                                           
   113      1577      14788.8      9.4      0.1      new_len = len(tree.text_content())
   114                                               # todo: adjust for recall and precision settings
   115      1577       1195.3      0.8      0.0      if new_len > old_len/7:
   116      1444        198.0      0.1      0.0          return tree
   117       133         17.6      0.1      0.0      return backup

 11.13 seconds - /root/pyvenv-fineweb/lib/python3.10/site-packages/trafilatura/htmlprocessing.py:90 - prune_unwanted_nodes
 ```

 推测是`expression(tree)`调用时间最长，把它从for循环里拆开来后进一步分析，果然`XPath`求值是最慢的，占了97.1%的时间。

 ```
 Timer unit: 1e-06 s

Total time: 11.1063 s
File: /root/pyvenv-fineweb/lib/python3.10/site-packages/trafilatura/htmlprocessing.py
Function: prune_unwanted_nodes at line 90

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    90                                           @line_profiler.profile
    91                                           def prune_unwanted_nodes(tree, nodelist, with_backup=False):
    92                                               '''Prune the HTML tree by removing unwanted sections.'''
    93     11665       3206.3      0.3      0.0      if with_backup:
    94      1577      25812.4     16.4      0.2          old_len = len(tree.text_content())  # ' '.join(tree.itertext())
    95      1577      78249.5     49.6      0.7          backup = deepcopy(tree)
    96                                           
    97     27462       6485.1      0.2      0.1      for expression in nodelist:
    98     15797   10848708.6    686.8     97.7          subtrees = expression(tree)
    99     54290      13832.9      0.3      0.1          for subtree in subtrees:
   100                                                       # preserve tail text from deletion
   101     38493       7947.7      0.2      0.1              if subtree.tail is not None:
   102     28309      11156.7      0.4      0.1                  prev = subtree.getprevious()
   103     28309       3635.4      0.1      0.0                  if prev is None:
   104     17852       7424.6      0.4      0.1                      prev = subtree.getparent()
   105     28309       3392.5      0.1      0.0                  if prev is not None:
   106                                                               # There is a previous node, append text to its tail
   107     28309      23303.6      0.8      0.2                      prev.tail = " ".join([prev.tail, subtree.tail]) if prev.tail else subtree.tail
   108                                                       # remove the node
   109     38493      52844.9      1.4      0.5              subtree.getparent().remove(subtree)
   110                                           
   111     11665       1721.9      0.1      0.0      if not with_backup:
   112     10088       1288.3      0.1      0.0          return tree
   113                                           
   114      1577      15838.1     10.0      0.1      new_len = len(tree.text_content())
   115                                               # todo: adjust for recall and precision settings
   116      1577       1234.6      0.8      0.0      if new_len > old_len/7:
   117      1444        200.1      0.1      0.0          return tree
   118       133         17.2      0.1      0.0      return backup

 11.11 seconds - /root/pyvenv-fineweb/lib/python3.10/site-packages/trafilatura/htmlprocessing.py:90 - prune_unwanted_nodes
 ```

## 分析结果

通过`yappi`的结果发现`prune_unwanted_nodes`在Python解释器的部分占用了不少时间，
但是由于再往下是用Cython实现的，没法再进一步分解了。
尝试用line_profiler进行分析。

总得来说，可以用`yappi`/`py-spy`定位到热点Python函数，如果能找到热点函数，可以再用`line_profiler`进一步细化热点情况。

# lxml从源码构建

```bash
git clone https://github.com/lxml/lxml
cd lxml
git checkout lxml-5.2.2
sudo apt install libxslt-dev libxml2-dev
pip install -r requirements.txt

# 就地构建Cython相关的依赖
python setup.py build_ext -i --with-cython

# 执行单元测试，1946个测试全部通过
python test.py

python ./benchmark/bench_xpath.py
```

共计有4个Cython文件，分别是`src/lxml/etree.pyx`、`src/lxml/objectify.pyx`、`src/lxml/builder.py`、`src/lxml/_elementpath.py`、`src/lxml/html/diff.py`、`src/lxml/sax.py`。
对应的C文件分别在`src/lxml/etree.c`、`src/lxml/objectify.c`、`src/lxml/builder.c`、`src/lxml/_elementpath.c`、`src/lxml/html/diff.c`、`src/lxml/sax.c`。

`XPath`类的定义位于xpath.pxi:377，它首先会调用`xmlXPathCtxtCompile`将XPath查询编译，进而调用`xmlXPathCompiledEval`（声明自xpath.pyd:85，实现在libxml2，可见xpath.pxd的作用是libxml2的Cython-binding）去搜索XML树。

# libxml2从源代码构建

```bash
sudo apt install autoconf libtool bear
git clone https://github.com/GNOME/libxml2
# 查询到版本为2.9.13+dfsg-1ubuntu0.4
dpkg -l | grep libxml2
git checkout v2.9.13
CFLAGS='-O2 -fno-semantic-interposition' ./autogen.sh
bear -- make
make runtest
./runtest
make testXPath && LD_LIBRARY_PATH=.libs ./.libs/testXPath -f example_expression.txt
```

XPath有以下的OP：
1. END：结束程序
2. AND：二元操作，布尔值与
3. OR：二元操作，布尔值或
4. EQUAL：二元操作，判定相等
5. CMP：二元操作，比较（大于、小于）
6. PLUS：二元操作，加法
7. MULT：二元操作，乘法
8. UNION：二元操作，面向NodeSet，将第二个参数的Node合并到第一个参数的NodeSet中
9. ROOT：初始化Value Stack，将document的root加入NodeSet
10. NODE：将0、1、2个参数求值后，将包含上下文Node的单元素NodeSet推入栈顶
11. COLLECT
12. VALUE：将op的value4（第一个指针参数）拷贝一份并推入栈顶
13. VARIABLE：查找变量表（一个参数为单一名字，两个参数为局部名字+命名空间）
14. FUNCTION：调用函数
15. ARG
16. PREDICATE
17. FILTER
18. SORT
19. RANGETO

## Collect操作

它的collect操作比较复杂，有多个参数

```c
typedef enum {
    AXIS_ANCESTOR = 1,
    AXIS_ANCESTOR_OR_SELF,
    AXIS_ATTRIBUTE,
    AXIS_CHILD,
    AXIS_DESCENDANT,
    AXIS_DESCENDANT_OR_SELF,
    AXIS_FOLLOWING,
    AXIS_FOLLOWING_SIBLING,
    AXIS_NAMESPACE,
    AXIS_PARENT,
    AXIS_PRECEDING,
    AXIS_PRECEDING_SIBLING,
    AXIS_SELF
} xmlXPathAxisVal;
```

```c
typedef enum {
    NODE_TEST_NONE = 0,
    NODE_TEST_TYPE = 1,
    NODE_TEST_PI = 2,
    NODE_TEST_ALL = 3,
    NODE_TEST_NS = 4,
    NODE_TEST_NAME = 5
} xmlXPathTestVal;
```

```c
typedef enum {
    NODE_TYPE_NODE = 0,
    NODE_TYPE_COMMENT = XML_COMMENT_NODE,
    NODE_TYPE_TEXT = XML_TEXT_NODE,
    NODE_TYPE_PI = XML_PI_NODE
} xmlXPathTypeVal;
```

## 其他


编译后的表达式结构体如下：

```c
struct _xmlXPathCompExpr {
    int nbStep;			/* Number of steps in this expression */
    int maxStep;		/* Maximum number of steps allocated */
    xmlXPathStepOp *steps;	/* ops for computation of this expression */
    int last;			/* index of last step in expression */
    xmlChar *expr;		/* the expression being computed */
    xmlDictPtr dict;		/* the dictionary to use if any */
#ifdef DEBUG_EVAL_COUNTS
    int nb;
    xmlChar *string;
#endif
#ifdef XPATH_STREAMING
    xmlPatternPtr stream;
#endif
};
```

编译后的表达式是一个树形结构（最多有两个儿子），解释执行从`comp->last`指向的最后一个操作开始，也就是说，最后一个操作是树根。
以一种递归的方式解释执行任意一棵子树。

执行环境是一个堆栈机，上下文里有一个Stack。

```c
struct _xmlXPathParserContext {
    const xmlChar *cur;			/* the current char being parsed */
    const xmlChar *base;		/* the full expression */

    int error;				/* error code */

    xmlXPathContextPtr  context;	/* the evaluation context */
    xmlXPathObjectPtr     value;	/* 维护了Stack top（View） */
    int                 valueNr;	/* 当前Stack大小 */
    int                valueMax;	/* 当前Stack的容量 */
    xmlXPathObjectPtr *valueTab;	/* Stack数据结构 */

    xmlXPathCompExprPtr comp;		/* the precompiled expression */
    int xptr;				/* it this an XPointer expression */
    xmlNodePtr         ancestor;	/* used for walking preceding axis */

    int              valueFrame;        /* 当前Frame允许的最小Stack长度（避免影响调用者的值） */
};
```

除了Value Stack之外还有一个Evaluation Context：

```c
struct _xmlXPathContext {
    xmlDocPtr doc;			/* 当前正在处理的document？ */
    xmlNodePtr node;			/* 当前正在处理的node？ */

    int nb_variables_unused;		/* unused (hash table) */
    int max_variables_unused;		/* unused (hash table) */
    xmlHashTablePtr varHash;		/* Hash table of defined variables */

    int nb_types;			/* number of defined types */
    int max_types;			/* max number of types */
    xmlXPathTypePtr types;		/* Array of defined types */

    int nb_funcs_unused;		/* unused (hash table) */
    int max_funcs_unused;		/* unused (hash table) */
    xmlHashTablePtr funcHash;		/* Hash table of defined funcs */

    int nb_axis;			/* number of defined axis */
    int max_axis;			/* max number of axis */
    xmlXPathAxisPtr axis;		/* Array of defined axis */

    /* the namespace nodes of the context node */
    xmlNsPtr *namespaces;		/* Array of namespaces */
    int nsNr;				/* number of namespace in scope */
    void *user;				/* function to free */

    /* extra variables */
    int contextSize;			/* the context size */
    int proximityPosition;		/* the proximity position */

    /* extra stuff for XPointer */
    int xptr;				/* is this an XPointer context? */
    xmlNodePtr here;			/* for here() */
    xmlNodePtr origin;			/* for origin() */

    /* the set of namespace declarations in scope for the expression */
    xmlHashTablePtr nsHash;		/* The namespaces hash table */
    xmlXPathVariableLookupFunc varLookupFunc;/* variable lookup func */
    void *varLookupData;		/* variable lookup data */

    /* Possibility to link in an extra item */
    void *extra;                        /* needed for XSLT */

    /* The function name and URI when calling a function */
    const xmlChar *function;
    const xmlChar *functionURI;

    /* function lookup function and data */
    xmlXPathFuncLookupFunc funcLookupFunc;/* function lookup func */
    void *funcLookupData;		/* function lookup data */

    /* temporary namespace lists kept for walking the namespace axis */
    xmlNsPtr *tmpNsList;		/* Array of namespaces */
    int tmpNsNr;			/* number of namespaces in scope */

    /* error reporting mechanism */
    void *userData;                     /* user specific data block */
    xmlStructuredErrorFunc error;       /* the callback in case of errors */
    xmlError lastError;			/* the last error */
    xmlNodePtr debugNode;		/* the source node XSLT */

    /* dictionary */
    xmlDictPtr dict;			/* dictionary if any */

    int flags;				/* flags to control compilation */

    /* Cache for reusal of XPath objects */
    void *cache;

    /* Resource limits */
    unsigned long opLimit;
    unsigned long opCount;
    int depth;
};
```

具体每个操作结构体如下：

```c
struct _xmlXPathStepOp {
    xmlXPathOp op;		/* 操作编号 */
    int ch1;			/* 第一个儿子 */
    int ch2;			/* 第二个儿子 */
    int value;
    int value2;
    int value3;
    void *value4;
    void *value5;
    xmlXPathFunction cache;
    void *cacheURI;
};
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

