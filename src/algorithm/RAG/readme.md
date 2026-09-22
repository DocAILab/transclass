# 数据分类分级 RAG

本项目结合动态缩写词典、BGE-M3 多片段检索和 LLM 重排，对金融、教育、车辆三个领域的字段名进行分类。所有默认资源路径均以代码所在目录为基准；按本文步骤准备运行环境、下载数据与模型后，即可构建索引并运行实验。

## 1. 快速使用

首次在全新环境使用，请先按第 7 节创建 Python 3.11 环境、安装依赖、完成 Hugging Face 数据集授权，然后在本目录执行：

```bash
source .venv/bin/activate
export HF_TOKEN='自己的 Hugging Face 令牌'
bash run_rag.sh --download-data
bash run_rag.sh --download-model
bash run_rag.sh --validate-only
bash run_rag.sh --build-index

export MODELSCOPE_API_TOKEN='自己的 ModelScope 令牌'
bash run_rag.sh
```

默认入口就是完整的三个领域实验，等价于 `bash run_rag.sh --run`。首次运行会建立缩写词典并请求模型重排；后续运行复用相同范围的缓存。建索引可单独提前执行，也会在正式检索时自动执行；它只需要本地 BGE-M3，不需要线上 LLM。

参考实验结果：

| 数据集 | 样本数 | 检索 Accuracy | 重排 Accuracy | 检索 Macro-F1 | 重排 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| DCG_FIN | 556 | 10.25% | 18.17% | 0.0746 | 0.1450 |
| DCG_EDU | 203 | 8.87% | 28.57% | 0.0474 | 0.1406 |
| DCG_VEH | 288 | 16.67% | 16.67% | 0.1689 | 0.2783 |

实验指标受数据版本、缩写词典和线上模型输出影响；即使 temperature=0，重新请求模型也可能得到不同结果。复现实验时应固定数据、模型及参数，并保留缓存和逐条预测文件；可使用下方结果复核命令重新计算已保存预测的指标。

常用命令（相对路径从调用命令时的当前目录解释；未指定的默认路径从脚本目录解释）：

```bash
# 全部三个领域
bash run_rag.sh --run
# 单独运行教育
bash run_rag.sh --run --input-json ./data/DCG_EDU/DCG_EDU_test.json --dataset-id DCG_EDU
# 不读取样本 domain，走通用提示词并合并三个领域语料
bash run_rag.sh --run --no-use-domain
# 种子词与规则的纯检索基线，忽略 LLM 缩写缓存且跳过重排
bash run_rag.sh --run --no-llm
# 缩写解释仍可请求模型，只跳过最终重排
bash run_rag.sh --run --skip-rerank
# 只复核已经生成的某次实验；路径替换为实际目录，不调用模型
bash run_rag.sh --reproduce --result-dir ./results/experiments/DCG_FIN/实际实验指纹
bash run_rag.sh --help
bash run_rag.sh --run --help
```

`--reproduce` 根据指定实验目录的逐条预测重新计算指标，并与该目录保存的汇总报告比较。

## 2. 代码结构

| 文件 | 职责 |
|---|---|
| `run_rag.sh` | 统一启动入口，默认完整运行；提供下载、校验与建索引命令 |
| `rag_pipeline.py` | 三领域流程编排、结果保存与结果复核 |
| `abbreviation_generator.py` | 缩写拆分、LLM 生成、SQLite 存取及查询视图 |
| `abbreviation_response.py` | JSON 解码、保守的结构恢复 |
| `abbreviation_cache.py` | 共享缩写缓存的定位、隔离与复用 |
| `retrieval.py` | 语料分片、BGE-M3 向量召回、类别聚合及 RRF |
| `reranking.py` | 候选转换、任务去重、重排结果映射与指标 |
| `llm_client.py` | 测试集/语料读取、模型客户端、重试及重排缓存 |
| `vector_index.py` | 知识库记录、向量索引、标签规范化 |
| `dataset_utils.py` | 数据审计、实验指纹、分组评价与跨数据集汇总 |
| `domain_config.py` | 领域与数据路径配置、domain 路由 |

Python 模块使用同目录导入，请保持上述文件位于同一目录，并通过 `run_rag.sh` 启动。

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

也支持 Excel 输入：D 列字段名作为输入，I 列为分类 ground truth，J 列为级别；默认工作表 Sheet1、表头行 2。Excel 不含本流程的记录级 domain 时走通用流程。默认三个领域数据集均使用 JSON。

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
| `--llm-api-key-env` | MODELSCOPE_API_TOKEN | 模型客户端读取的 token 环境变量名 |
| `--llm-timeout` | 90 秒 | 单次请求超时 |
| `--llm-max-retries` | 2 | 初次请求之后最多重试次数 |
| `--llm-request-delay` | 3 秒 | 请求节奏间隔 |
| `--llm-rate-limit-delay` | 60 秒 | 429 重试的初始等待时间 |
| `--llm-candidate-text-chars` | 100 | 每个重排候选定义的截断字符数 |
| `--device` | cpu | cpu/cuda/auto |
| `--data-root` | 本目录/data | 三领域测试集和语料根目录 |
| `--finance-corpus` / `--education-corpus` / `--vehicle-corpus` | 各领域默认语料 | 覆盖对应语料路径 |
| `--input-json`、`--dataset-id` | 无参数时由 Shell 逐个指定三个数据集 | 单数据集输入及结果目录标识 |
| `--abbreviation-cache-dir` | 本目录/cache/abbreviations | 共享缩写缓存根目录 |
| `--cache-dir` | 本目录/cache/indexes | 向量索引根目录 |
| `--output-dir` | 本目录/results/experiments | 实验结果输出根目录 |
| `--model-dir` | 本目录/models/BAAI_bge-m3 | 本地向量模型路径，缺少资源时明确报错 |
| `--build-index` | 关闭 | 仅构建知识库向量索引，不生成词典、不调用线上 LLM |
| `--validate-only` | 关闭 | 检查输入和配置后退出 |
| `--skip-rerank` | 关闭 | 仅跳过最终重排，缩写仍可能调用模型 |
| `--no-llm` | 关闭 | 种子词/规则的纯检索基线，并跳过重排 |

Python 和 Shell 的 dictionary-batch-size 默认均为 4。模型温度固定为 0；缩写输出上限按复杂度估计，普通 4 条约 4800 tokens，上限 16000，并非要求模型填满。

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

默认完整实验需要 MODELSCOPE_API_TOKEN；数据校验、下载、索引构建和 `--no-llm` 不需要该令牌。Hugging Face 数据集授权使用第 7 节的 HF_TOKEN，不能用 ModelScope token 替代。

## 7. 环境准备、下载与资源构建

### 7.1 目录及环境

进入代码所在目录（下称 RAG）。资源准备和实验运行后的目录结构如下：

```text
RAG/
├── run_rag.sh、10个Python文件、readme.md
├── .venv/                         # 创建的 Python 环境
├── data/
│   ├── DCG_EDU/{DCG_EDU_test.json,DCG_EDU_Corpus.json}
│   ├── DCG_FIN/{DCG_FIN_test.json,DCG_FIN_Corpus.json}
│   ├── DCG_VEH/{DCG_VEH_test.json,DCG_VEH_Corpus.json}
│   └── resource_manifest.json     # 仓库版本及下载文件哈希
├── models/BAAI_bge-m3/            # 模型、分词器、配置和下载清单
├── cache/
│   ├── abbreviations/            # 按领域、模型、提示词版本隔离的 SQLite
│   └── indexes/                  # 按领域、模型内容与语料隔离的向量索引
└── results/experiments/           # 实验结果与请求日志
```

在 macOS 或 Linux 上准备 Python 3.11 后执行（Windows 请使用 WSL/Linux 环境运行此 Shell）：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# XRAG 只使用其向量检索组件，不安装它的整套额外依赖
python -m pip install examinationrag==0.1.4 --no-deps
```

根据平台选择一条 PyTorch 安装命令：

```bash
# Intel macOS
python -m pip install torch==2.2.2
# Apple Silicon macOS（与上一条二选一）
python -m pip install torch==2.3.0
# Linux CPU（与上面二选一）；GPU 环境按 CUDA 配置匹配的 torch 2.3.0
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

然后安装其余运行与下载依赖：

```bash
python -m pip install \
  llama-index-core==0.11.23 llama-index-legacy==0.9.48 \
  llama-index-embeddings-huggingface==0.3.1 \
  llama-index-retrievers-bm25==0.3.0 llama-index-llms-openai==0.2.16 \
  sentence-transformers==3.0.1 transformers==4.42.2 \
  huggingface-hub==0.24.7 openai==1.47.0 httpx==0.28.1 numpy==1.26.4
```

Python 选择顺序：显式 `PYTHON_BIN` → 本目录 `.venv/bin/python` → PATH 中的 `python3`。`PYTHON_BIN` 应是解释器路径，不是带参数的命令字符串。

### 7.2 下载数据：先完成 Hugging Face 授权

数据来自 [DocAILab/DCG](https://huggingface.co/datasets/DocAILab/DCG) 仓库内的三个目录，不是三个独立仓库。远端元数据显示其为 gated 数据集；先登录 Hugging Face，在数据集网页按要求申请/接受访问条款，再使用具有该数据集读取权限的 token：

```bash
export HF_TOKEN='自己的 Hugging Face 读取令牌'
# 可选：持久登录 Hugging Face。固定的 huggingface-hub 0.24.7 使用此命令：
huggingface-cli login
bash run_rag.sh --download-data
```

下载入口使用 `huggingface_hub.HfApi` 和 `hf_hub_download`，下载三个领域的测试集与语料库，共六个文件：

```text
DCG_EDU/DCG_EDU_test.json       DCG_EDU/DCG_EDU_Corpus.json
DCG_FIN/DCG_FIN_test.json       DCG_FIN/DCG_FIN_Corpus.json
DCG_VEH/DCG_VEH_test.json       DCG_VEH/DCG_VEH_Corpus.json
```

默认数据版本固定为 commit `ce9433b3fef805b864c6e941d721319e942335ce`。需要另一个版本时：

```bash
bash run_rag.sh --download-data --revision main
# 或传入一个确定的 commit SHA
```

下载先落临时目录，通过 JSON 结构、字段、领域及分类覆盖校验后再写入目标目录。`data/resource_manifest.json` 记录实际 commit 和每个文件的 SHA-256，不记录 token。再次下载相同版本时会校验本地文件，完整时跳过。下载需要网络；权限失败会明确提示 HF 授权，不会把登录页当数据。

### 7.3 下载 BGE-M3

本地向量模型使用 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)：

```bash
bash run_rag.sh --download-model
# 固定某次下载记录中的 commit，以便在另一环境获得相同模型版本
bash run_rag.sh --download-model --revision 模型commit
```

下载完整 dense 模型权重（优先 safetensors，否则 pytorch_model.bin）、分词器、SentencePiece、SentenceTransformer 与 Pooling 必需配置，不下载 ONNX 或图片等额外文件。默认将 main 解析为具体 commit 并记录到 `models/BAAI_bge-m3/resource_manifest.json`；复现实验时可用清单内的 commit 固定版本。

模型文件缺失时，正式流程会提示下载命令并停止运行。

### 7.4 构建资源并开始实验

```bash
# 只检查测试集/语料，不要求本地模型或 ModelScope token
bash run_rag.sh --validate-only
# 只用知识库与本地 BGE-M3，生成三个领域的索引
bash run_rag.sh --build-index
# 如果计划领域未知实验，可提前创建合并语料的通用索引
bash run_rag.sh --build-index --no-use-domain
# 正式实验；缩写词典从种子词开始逐步生成
export MODELSCOPE_API_TOKEN='自己的 ModelScope 令牌'
bash run_rag.sh --run
```

索引初始化由本地编码产生；缩写 SQLite 自动创建，未知缩写经第一阶段模型请求生成后写入。第二阶段的重排缓存随实验生成。中断后用相同设置重新运行，可复用已经完成的部分。

可覆盖路径：下载支持 `--data-root` / `--model-dir`，正式流程支持 `--data-root`、`--model-dir`、`--cache-dir`、`--abbreviation-cache-dir`、`--output-dir`。使用自定义下载路径时，正式运行也应传入相应路径。例如：

```bash
bash run_rag.sh --download-model --model-dir ./resources/bge-m3
bash run_rag.sh --build-index --model-dir ./resources/bge-m3
bash run_rag.sh --run --model-dir ./resources/bge-m3
```

## 8. 缓存与实验结果

以下目录由程序自动创建：

| 目录 | 用途 | 创建方式 |
|---|---|---|
| `cache/abbreviations/` | 共享缩写词典，复用字段解释以减少模型请求 | 运行时自动创建 SQLite，逐步写入通过校验的解释 |
| `cache/indexes/` | 知识库的本地向量索引 | 执行 `--build-index` 或首次检索时生成 |
| `results/experiments/` | 实验指标、逐条预测、解释快照及请求日志 | 运行实验时按阶段保存 |

实验结果位于 `results/experiments/<dataset-id>/<fingerprint>/`。`dataset-id` 标识数据集，`fingerprint` 根据输入、语料、代码与参数等配置生成，用于区分实验。

查看整体表现时先打开 `summary.json`；分析具体样本时查看 `retrieval.json` 和 `rerank.json`。跳过重排时，查看召回结果即可。各文件内容如下：

- `summary.json`：全数据集与分组指标、参数和数据审计。
- `retrieval.json`：逐条查询视图、解释、候选及真实标签。
- `rerank.json`：逐条最终预测、理由、候选 ID、真实标签与正确性。
- `routes/<领域>/field_expansions.json`：本次使用的解释快照。
- `routes/<领域>/dictionary.metrics.json`：词典命中与调用统计。
- `routes/<领域>/abbreviation.responses.jsonl`：本次缩写请求原始响应及恢复日志。
- `routes/<领域>/rerank.cache.jsonl`：重排断点缓存。
- `manifest.json`：输入、语料、代码与参数指纹。

缩写缓存按领域、模型及提示词语义版本隔离。向量索引按模型文件内容、知识库文本和领域等信息隔离，移动部署目录不会仅因模型绝对路径变化而改变索引内容指纹。变更模型或知识库后按新指纹创建索引，不覆盖其他版本。

Accuracy 的分母是所有有标签样本；Macro-F1 的类别集合为规范化后的真实类别与非空预测类别并集。失败预测计入准确率分母及失败率；Recall@K 只报告实际保存的候选范围。全候选召回率 100% 不等于 Top-1 分类准确率 100%。
