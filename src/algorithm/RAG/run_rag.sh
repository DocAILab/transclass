#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
# Resolve explicit relative paths from the caller's working directory.
# Defaults are always relative to the delivery directory, not $PWD.
MODE="${1:---run}"
EXTRA=()
case "$MODE" in
  --download-data|--download-model)
    shift
    exec "$PYTHON_BIN" "$SCRIPT_DIR/dataset_utils.py" "$MODE" "$@" ;;
  --reproduce)
    shift
    exec "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" --reproduce "$@" ;;
  --help|-h)
    cat <<'HELP'
首次使用（见 readme.md 安装 Python 3.11 环境和依赖）：
  bash run_rag.sh --download-data       下载三个领域的测试集和语料（需要 HF 授权）
  bash run_rag.sh --download-model      下载本地 BGE-M3
  bash run_rag.sh --validate-only       校验三个领域的数据，不调用模型
  bash run_rag.sh --build-index         构建知识库向量索引，不调用线上模型
  bash run_rag.sh                       完整运行三个领域（需要 MODELSCOPE_API_TOKEN）
  bash run_rag.sh --run --no-llm         种子/规则查询的纯检索基线
  bash run_rag.sh --run --no-use-domain  不读取 domain，走通用流程
  bash run_rag.sh --run --input-json /path/test.json --dataset-id custom
  bash run_rag.sh --reproduce --result-dir /path/to/experiment
  bash run_rag.sh --run --help           查看实验参数
环境变量：PYTHON_BIN、HF_TOKEN（下载数据）、MODELSCOPE_API_TOKEN（线上推理）。
默认资源在脚本目录的 data/、models/、cache/、results/，不依赖旧工程。
HELP
    exit 0 ;;
  --validate-only|--build-index)
    EXTRA=("$MODE"); shift ;;
  --run)
    if [[ $# -gt 0 ]]; then shift; fi ;;
  *) echo "未知模式：$MODE；请执行 bash run_rag.sh --help" >&2; exit 2 ;;
esac
CUSTOM_INPUT=false
HAS_DATASET_ID=false
DATA_ROOT="$SCRIPT_DIR/data"
PREVIOUS_ARG=""
for arg in "$@"; do
  if [[ "$PREVIOUS_ARG" == "--data-root" ]]; then DATA_ROOT="$arg"; fi
  case "$arg" in
    --help|-h) exec "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" --help ;;
    --input-json|--input-json=*|--input-xlsx|--input-xlsx=*) CUSTOM_INPUT=true ;;
    --dataset-id|--dataset-id=*) HAS_DATASET_ID=true ;;
    --data-root=*) DATA_ROOT="${arg#*=}" ;;
  esac
  PREVIOUS_ARG="$arg"
done
if [[ "$HAS_DATASET_ID" == true && "$CUSTOM_INPUT" == false ]]; then
  echo "指定 --dataset-id 时请同时指定 --input-json。" >&2
  exit 2
fi
COMMON=(--dictionary-batch-size 4 --llm-batch-size 8 --candidate-k 30 --output-k 30
        --llm-model Qwen/Qwen3-30B-A3B-Instruct-2507 --llm-max-retries 2)
# The Python client checks --llm-api-key-env, after local resource validation.
if [[ "$CUSTOM_INPUT" == true ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
fi
for DATASET in DCG_FIN DCG_EDU DCG_VEH; do
  "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" "${COMMON[@]}" \
    --input-json "$DATA_ROOT/$DATASET/${DATASET}_test.json" \
    --dataset-id "$DATASET" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
done
