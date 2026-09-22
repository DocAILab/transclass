# 三领域动态缩写 RAG：2026-09-22 交付版

本目录包含 11 个入口及依赖文件和本说明。由产生金融、教育、车辆三组实验结果的同一套代码整理而来，统一了文件名、模块导入和启动路径，保留原有分类算法及提示词。新增历史结果复核入口，用逐条预测重新计算指标并校验基准。

**交付范围仅为代码和本说明。数据、模型、向量索引、SQLite 缩写缓存和历史结果均留在原工作区，通过路径引用，不复制到本目录。** 因此将这 12 个文件单独拷到另一台机器，并不包含运行资源。

## 1. 快速使用与复现范围

在本机任意终端执行：

```bash
bash /Users/andiandian/Desktop/RAG20260922/run_rag.sh
```

默认模式 `--reproduce`：读取已完成实验的 `retrieval.json`、`rerank.json`，重新计算准确率、Macro-F1 和 Recall@K，检查结果文件及测试集、语料的 SHA-256，并与历史 `summary.json` 对比。不调用模型，不重做向量检索，不写共享缓存；这一步准确还原的是**已归档预测对应的实验指标**。

预期输出：

| 数据集 | 样本数 | 检索 Accuracy | 重排 Accuracy | 检索 Macro-F1 | 重排 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| DCG_FIN | 556 | 10.25% | 18.17% | 0.0746 | 0.1450 |
| DCG_EDU | 203 | 8.87% | 28.57% | 0.0474 | 0.1406 |
| DCG_VEH | 288 | 16.67% | 16.67% | 0.1689 | 0.2783 |

以下命令都在本交付目录执行：

```bash
cd /Users/andiandian/Desktop/RAG20260922

# 只复核金融，或将三个领域的复核报告另存为 JSON
bash run_rag.sh --reproduce --dataset DCG_FIN
bash run_rag.sh --reproduce --output /tmp/rag-reproduced-summary.json

# 只检查三个领域的数据结构、标签覆盖与领域配置，不调用模型
bash run_rag.sh --run --validate-only

# 完整运行三个领域：读取/构建缩写词典 → 检索 → LLM 重排 → 评价
# 需要先配置第 6 节的 token；缓存未命中时会请求模型
bash run_rag.sh --run

# 单独运行教育：dataset-id 仅命名结果目录，不决定推理领域
bash run_rag.sh --run \
  --input-json /Users/andiandian/Desktop/trandatacls/data/DCG_EDU/DCG_EDU_test.json \
  --dataset-id DCG_EDU

# 领域不可知实验：不读取样本 domain，使用通用提示词和合并语料
bash run_rag.sh --run --no-use-domain

# 完全不调用 LLM：只用种子词/规则生成查询，并跳过重排
bash run_rag.sh --run --no-llm

# 动态缩写仍可调用 LLM，但不做最终重排
bash run_rag.sh --run --skip-rerank

bash run_rag.sh --help
bash run_rag.sh --run --help
```

`--run` 是实际实验流程，并非历史结果复核。它会复用相同领域、模型、词典语义版本下的缩写缓存；同一新实验指纹已存在时，也会复用其检索/重排结果。代码重命名会改变源码指纹，因此新交付版不会把原实验结果伪装成新计算的结果。首次完整运行通常需要重新重排。

线上模型即使 temperature=0，也不承诺跨时间逐条输出一致。重做模型请求的结果可能不同；需要准确还原当前报告时使用默认的 `--reproduce`。历史结果不存在或被修改时，复核模式明确报错，不会静默改跑模型。

## 2. 11 个文件及命名对应

| 交付文件 | 原文件 | 职责 |
|---|---|---|
| `run_rag.sh` | `run_finance_dynamic_rag.sh` | 统一启动入口，默认复核，`--run` 运行实验 |
| `rag_pipeline.py` | `finance_dynamic_rag_experiment.py` | 三领域流程编排、结果保存与历史结果复核 |
| `abbreviation_generator.py` | `dynamic_abbreviation.py` | 缩写拆分、LLM 生成、SQLite 存取及查询视图 |
| `abbreviation_response.py` | 同名 | JSON 解码、保守的结构恢复 |
| `abbreviation_cache.py` | 同名 | 共享缓存定位、隔离及旧缓存迁移 |
| `retrieval.py` | `finance_recall_ablation.py` | 语料分片、BGE-M3 向量召回、类别聚合及 RRF |
| `reranking.py` | `finance_ablation_llm_rerank.py` | 候选转换、任务去重、重排结果映射与指标 |
| `llm_client.py` | `finance_rerank_experiment.py` | 测试集/语料读取、模型客户端、重试及重排缓存 |
| `vector_index.py` | `xrag_classifier.py` | 知识库记录、向量索引、标签规范化 |
| `dataset_utils.py` | `rag_dataset_support.py` | 数据审计、实验指纹、分组评价与跨数据集汇总 |
| `domain_config.py` | `rag_domains.py` | 领域与数据路径配置、domain 路由 |

这些 Python 文件使用同目录导入，运行时不依赖原项目的 Python 源码包。部分公共模块仍保留历史实验辅助函数，以保持算法行为；正式入口统一使用 `run_rag.sh`。

## 3. 输入字段与知识库字段

### 测试集

默认读取 JSON 对象数组，每条样本至少包含唯一 `id`、`metadata.field_name` 和 `classification.category_leaf_level`。

| 字段 | 用途 | 是否提供给模型作为业务信息 |
|---|---|---|
| `metadata.field_name` | 唯一的字段业务输入；拆分、扩展及分类 | 是 |
| 顶层 `domain` | 默认允许读取，选择领域提示词和知识库 | 是，仅领域上下文 |
| `classification.category_leaf_level` | ground truth，用于离线评价及标签覆盖检查 | 否 |
| `grading.sensitivity_level` | 离线标签/结果元信息 | 否 |
| `id` | 样本关联、去重检查及结果定位 | 不作为语义输入 |
| `label_status` | original/synthesized 等分组评价 | 否 |

测试数据的表名、字段描述、字段类型、样例值以及其他分类层级不会用于本流程的字段解释和分类查询。读取 ground truth 不代表将其送入模型；当前实验读取器要求标签非空，因此它不是无标签生产接口。

`domain` 必须精确为 `finance`、`dcg_education`、`dcg_vehicle` 之一，分别走金融、教育、车辆流程；缺失、其他值，或设置 `--no-use-domain` 时，走通用流程并合并三个领域语料。命令行不用另加领域参数，`--dataset-id` 不参与推断。

兼容旧 Excel 输入：D 列字段名作为输入，I 列为分类 ground truth，J 列为级别；默认工作表 Sheet1、表头行 2。Excel 不含本流程的记录级 domain 时走通用流程。三个交付基准实验均使用 JSON。

### 知识库

各领域默认文件为 `DCG_*/DCG_*_Corpus.json`，当前三个文件均为对象数组。

| 知识库字段 | 程序用途 |
|---|---|
| `id` | 标准记录 ID；按领域加前缀，关联检索与模型选择 |
| `category_leaf_level` | 候选分类名称及预测输出类别 |
| `data_description_and_example` | 定义与示例，生成完整索引文本和语义片段；重排候选定义也来自这里 |
| `category_root_level`、`category_branch_level`、`category_subbranch_level`、`category_leaf_level` | 保存类别路径，存在的层级按顺序组成路径 |
| `sensitivity_level` | 保存级别，并在索引文本中使用 |
| `reference_standard` | 保存标准来源，并在索引文本中使用 |
| `dataset`，以及存在时的 `domain` | 校验语料声明是否与选定语料领域一致 |

金融语料没有 `category_subbranch_level`，读取器允许该层级缺失。分类名称、定义、级别、来源组成主要索引文本；类别路径另存元数据。知识库里的类别与定义属于候选信息，允许提供给模型，测试集的正确类别不提供给模型。

当前语料规模：金融 12 条/12 类；教育 18 条/18 类；车辆 59 条/44 类。

## 4. 流程与两处大模型调用

整体流程：读取字段名与可选 domain → 缩写解释与缓存 → 原字段/英文/中文查询视图 → BGE-M3 编码 → 完整定义及片段召回 → 按类别聚合 → RRF 融合 → 候选集内 LLM 重排 → 保存预测与评价。

**第一处：动态缩写解释。** 输入当前字段名、已知领域（或通用提示）、规则拆分及已有缩写词条提示。模型返回有序片段、片段的中英文含义、完整字段的中英文含义及置信度，每处最多两个解释。不得生成分类标签。6 个通用种子为 ID、NO、NAME、DATE、TIME、CODE。完整字段缓存命中时跳过模型；词条可组合时也可能无需模型。

解释响应先校验 JSON、任务身份和完整含义，再检查片段能否拼回规范化原字段。可确定归属的层级错误本地恢复；无法确定的部分拆批纠错重试。完整含义有效但拆分不合法时标记 `llm_unsplit`，仅保留整字段解释，不把错误片段写入通用词条。失败批次的合法条目先缓存。`cache_compose` 则表示由已有词条组合，并非本次直接请求模型。

**第二处：候选重排。** 输入由字段名和缩写解释构成的查询，以及候选的 standard_id、类别、领域、截断后的定义。模型在候选集合内选择/排序，返回 ID 和理由；程序验证 ID 并映射为类别。金融和教育的候选数不足 30，实际使用全部 12/18 类；车辆从 44 类取 30 类。非法候选有修复/回退逻辑，报告记录相应次数。

这里的“两处”是两个调用阶段，不是每条数据固定调用两次；缓存、词条组合、批处理和重试都会改变请求数。BGE-M3 是本地向量模型，不是第三处线上 LLM 调用。

## 5. 关键参数

以下默认值以 `run_rag.sh --run` 为准，后加同名参数可覆盖：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--use-domain` / `--no-use-domain` | 允许 | 是否读取记录 domain |
| `--dictionary-batch-size` | 4 | 每次缩写请求的最大字段任务数，异常时自动拆小 |
| `--llm-batch-size` | 8 | 每次重排请求的任务数 |
| `--output-k` | 30 | 保存的召回候选数上限 |
| `--candidate-k` | 30 | 送入重排的候选数上限；不能大于 output-k |
| `--rrf-k` | 60 | RRF 平滑常数，每路贡献约为 1/(k+rank)；不是候选数量 |
| `--llm-model` | Qwen/Qwen3-30B-A3B-Instruct-2507 | 缩写和重排所用线上模型 |
| `--llm-base-url` | https://api-inference.modelscope.cn/v1 | OpenAI 兼容接口地址 |
| `--llm-api-key-env` | MODELSCOPE_API_TOKEN | Python 客户端读取的 token 环境变量名；Shell 默认检查 MODELSCOPE_API_TOKEN |
| `--llm-timeout` | 90 秒 | 单次请求超时 |
| `--llm-max-retries` | 2 | 初次请求之后最多重试次数 |
| `--llm-request-delay` | 3 秒 | 请求节奏间隔 |
| `--llm-rate-limit-delay` | 60 秒 | 429 重试的初始等待时间 |
| `--llm-candidate-text-chars` | 100 | 每个重排候选定义的截断字符数 |
| `--device` | cpu | cpu/cuda/auto |
| `--data-root` | 原工作区/data | 三领域测试集和语料根目录 |
| `--finance-corpus` / `--education-corpus` / `--vehicle-corpus` | 各领域默认语料 | 覆盖对应语料路径 |
| `--input-json`、`--dataset-id` | 无参数时由 Shell 逐个指定三个数据集 | 单数据集输入及结果目录标识 |
| `--abbreviation-cache-dir` | 原项目/data/cache/abbreviations | 共享缩写缓存根目录 |
| `--cache-dir` | 原项目/src/algorithm/RAG/storage | 向量索引根目录 |
| `--output-dir` | 本交付目录/results/experiments | 新实验输出根目录 |
| `--validate-only` | 关闭 | 检查输入和配置后退出 |
| `--skip-rerank` | 关闭 | 仅跳过最终重排，缩写仍可能调用模型 |
| `--no-llm` | 关闭 | 种子词/规则的纯检索基线，并跳过重排 |

直接运行 `python rag_pipeline.py` 时，Python 解析器的 dictionary-batch-size 默认是 8；复现原设置应使用 Shell（固定传入 4）。模型温度固定为 0；缩写输出上限按复杂度估计，普通 4 条约 4800 tokens，上限 16000，并非要求模型填满。

## 6. ModelScope token 与登录

从 [ModelScope 访问令牌页面](https://modelscope.cn/my/myaccesstoken) 取得自己的令牌。当前终端设置：

```bash
export MODELSCOPE_API_TOKEN='在这里填写你自己的令牌'
```

本项目通过 OpenAI 兼容客户端读取这个环境变量即可调用 API，**不要求先执行 SDK 登录**。如果还需要登录 ModelScope CLI，安装/使用对应 CLI 后执行：

```bash
modelscope login --token "$MODELSCOPE_API_TOKEN"
```

命令依据：[ModelScope 官方 CLI 文档](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)。SDK/CLI 登录不会替本脚本设置 `MODELSCOPE_API_TOKEN`；仍需保留 export。

如果 macOS 的其他客户端也需要读取该环境变量，可以执行：

```bash
launchctl setenv MODELSCOPE_API_TOKEN "$MODELSCOPE_API_TOKEN"
```

Shell 在自身环境没有 token 时尝试读取 launchctl。不要把真实 token 写进交付代码或 README。默认结果复核、`--validate-only` 和 `--no-llm` 都不需要 token。模型权限或额度以自己的账号为准。

## 7. 运行环境与原资源路径

默认 `RAG_WORKSPACE_ROOT=/Users/andiandian/Desktop/trandatacls`。资源位置：

```text
$RAG_WORKSPACE_ROOT/
├── data/DCG_FIN/{DCG_FIN_test.json,DCG_FIN_Corpus.json}
├── data/DCG_EDU/{DCG_EDU_test.json,DCG_EDU_Corpus.json}
├── data/DCG_VEH/{DCG_VEH_test.json,DCG_VEH_Corpus.json}
└── transclass_repo/
    ├── src/algorithm/RAG/models/BAAI_bge-m3/
    ├── src/algorithm/RAG/storage/
    ├── src/algorithm/RAG/.conda/bin/python
    ├── data/cache/abbreviations/
    └── data/processed/finance_dynamic_rag/
```

资源工作区整体移动时，设置：

```bash
export RAG_WORKSPACE_ROOT='/新的工作区绝对路径'
bash /Users/andiandian/Desktop/RAG20260922/run_rag.sh
```

Python 选择顺序：显式 `PYTHON_BIN` → 本目录 `.venv/bin/python` → 原项目 `.conda/bin/python` → 系统 `python3`。默认复核仅需要标准库；实际向量检索需要原实验依赖。本机保留原环境即可，不必重装：

```bash
export PYTHON_BIN='/Users/andiandian/Desktop/trandatacls/transclass_repo/src/algorithm/RAG/.conda/bin/python'
```

需要另建环境时，使用 Python 3.11，以下安装命令与原 CPU 依赖版本对齐。先单独安装 examinationrag，避免它的完整依赖覆盖本实验版本：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install examinationrag==0.1.4 --no-deps
.venv/bin/python -m pip install \
  llama-index-core==0.11.23 llama-index-legacy==0.9.48 \
  llama-index-embeddings-huggingface==0.3.1 \
  llama-index-retrievers-bm25==0.3.0 llama-index-llms-openai==0.2.16 \
  sentence-transformers==3.0.1 transformers==4.42.2 \
  huggingface-hub==0.24.7 openai==1.47.0 httpx==0.28.1 numpy==1.26.4
# 原实验为 Intel macOS，安装：
.venv/bin/python -m pip install torch==2.2.2
# 其他受支持平台原依赖使用 torch==2.3.0；按目标平台选择其中一种。
```

此交付未安装新环境，也不自动下载模型或依赖。本地 BGE-M3 不存在时原检索逻辑可能尝试加载远端模型；要复现本地实验，应先确保以上模型目录仍然可读。

## 8. 结果、缓存与基准位置

新实验保存于：`本交付目录/results/experiments/<dataset-id>/<fingerprint>/`，其中：

- `summary.json`：全数据集指标、分组指标及配置。
- `retrieval.json`：逐条查询视图、缩写解释、召回候选及真实标签。
- `rerank.json`：逐条重排预测、理由、候选 ID、真实标签和正确性。
- `routes/<领域>/field_expansions.json`：本次实际使用的缩写解释快照。
- `routes/<领域>/dictionary.metrics.json`：缩写来源、命中率和 API 请求统计。
- `routes/<领域>/abbreviation.responses.jsonl`：新缩写请求的原始响应、校验错误及结构恢复日志。
- `routes/<领域>/rerank.cache.jsonl`：本次实验的重排断点缓存。
- `manifest.json`：输入、语料、源码哈希及参数，用于隔离不同实验。

共享缩写缓存按“领域＋模型＋提示词语义版本”隔离，普通重命名不改变语义版本。这里继续引用原共享缓存，正常实验产生的新合法词条也写入该共享缓存。旧缓存迁移保持原行为；不建议删除个别记录来强制重跑，因为迁移或词条组合可能再次命中。

已完成的原基准位于 `原项目/data/processed/finance_dynamic_rag/`：

| 数据集 | 实验目录 |
|---|---|
| DCG_FIN | `DCG_FIN/156f9a057f6d31ff3c11` |
| DCG_EDU | `DCG_EDU/20e2e668f7ed70a4a0b1` |
| DCG_VEH | `DCG_VEH/b959b1d007bafda65f08` |

复核入口固定选取这三次实验，不根据文件修改时间猜测“最新结果”。基准中拆分降级字段名数分别为金融 40、教育 10、车辆 23；完整含义仍参与查询。这些状态属于原实验，交付整理没有重写它们。

Accuracy 使用全部有标签样本为分母；Macro-F1 对规范化后的真实类别和非空预测类别并集求均值。失败预测计入失败率与准确率分母。Recall@K 仅报告实际可用的 K。金融和教育全候选召回率 100% 不能解释为 Top-1 分类准确率 100%。

## 9. 交付验证

整理后的代码通过 103 项原有单元测试；另 1 项旧 Excel 读取测试因原文件不存在而跳过。
三个领域均通过 validate-only 数据检查；从历史逐条预测重算的 Accuracy、Macro-F1、Recall@K 与基准一致。
另在禁用网络模型访问的条件下，用 1 条金融样本完成了真实本地 BGE-M3 编码、建索引和 12 类候选检索。
验证未调用线上 LLM、未修改原实验报告或共享词典。测试文件和测试输出留在临时目录，不属于这 12 个交付文件。
