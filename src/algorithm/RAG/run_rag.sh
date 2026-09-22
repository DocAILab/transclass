#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RAG_WORKSPACE_ROOT="${RAG_WORKSPACE_ROOT:-/Users/andiandian/Desktop/trandatacls}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
# Resources stay in the original workspace; only code lives in this directory.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
  elif [[ -x "$RAG_WORKSPACE_ROOT/transclass_repo/src/algorithm/RAG/.conda/bin/python" ]]; then
    PYTHON_BIN="$RAG_WORKSPACE_ROOT/transclass_repo/src/algorithm/RAG/.conda/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
cd "$SCRIPT_DIR"
MODE="${1:---reproduce}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  --reproduce)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" --reproduce "$@"
    ;;
  --help|-h)
    cat <<'HELP'
用法：
  bash run_rag.sh                         复核三个领域的历史结果（不调用模型）
  bash run_rag.sh --reproduce --dataset DCG_FIN
  bash run_rag.sh --reproduce --output /tmp/rag-report.json
  bash run_rag.sh --run                   三个领域重新运行完整流程
  bash run_rag.sh --run --validate-only    检查三个领域数据及配置，不调用模型
  bash run_rag.sh --run --no-llm           三个领域纯检索基线，不使用 LLM 缩写
  bash run_rag.sh --run --input-json /path/test.json --dataset-id custom
  bash run_rag.sh --run --help             查看实验参数
环境变量：RAG_WORKSPACE_ROOT（资源所在工作区）、PYTHON_BIN（Python 解释器）、MODELSCOPE_API_TOKEN。
HELP
    exit 0
    ;;
  --run) ;;
  *) echo "未知模式：$MODE；请执行 bash run_rag.sh --help" >&2; exit 2 ;;
esac
NEEDS_TOKEN=true
CUSTOM_INPUT=false
HAS_DATASET_ID=false
DATA_ROOT="$RAG_WORKSPACE_ROOT/data"
PREVIOUS_ARG=""
for arg in "$@"; do
  if [[ "$PREVIOUS_ARG" == "--data-root" ]]; then DATA_ROOT="$arg"; fi
  case "$arg" in
    --help|-h) exec "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" --help ;;
    --validate-only|--no-llm) NEEDS_TOKEN=false ;;
    --input-json|--input-json=*|--input-xlsx|--input-xlsx=*) CUSTOM_INPUT=true ;;
    --dataset-id|--dataset-id=*) HAS_DATASET_ID=true ;;
    --data-root=*) DATA_ROOT="${arg#*=}" ;;
  esac
  PREVIOUS_ARG="$arg"
done
if [[ "$HAS_DATASET_ID" == true && "$CUSTOM_INPUT" == false ]]; then
  echo "单独指定 --dataset-id 时请同时指定 --input-json，避免三个数据集写入同一数据集目录。" >&2
  exit 2
fi
if [[ "$NEEDS_TOKEN" == true ]]; then
  if [[ -z "${MODELSCOPE_API_TOKEN:-}" ]] && command -v launchctl >/dev/null; then
    MODELSCOPE_API_TOKEN="$(launchctl getenv MODELSCOPE_API_TOKEN 2>/dev/null || true)"
    export MODELSCOPE_API_TOKEN
  fi
  if [[ -z "${MODELSCOPE_API_TOKEN:-}" ]]; then
    echo "请先设置 MODELSCOPE_API_TOKEN；具体指令见 readme.md。" >&2
    exit 1
  fi
fi
COMMON=(--dictionary-batch-size 4 --llm-batch-size 8 --candidate-k 30 --output-k 30
        --llm-model Qwen/Qwen3-30B-A3B-Instruct-2507 --llm-max-retries 2)
if [[ "$CUSTOM_INPUT" == true ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" "${COMMON[@]}" "$@"
fi
for DATASET in DCG_FIN DCG_EDU DCG_VEH; do
  "$PYTHON_BIN" "$SCRIPT_DIR/rag_pipeline.py" "${COMMON[@]}" \
    --input-json "$DATA_ROOT/$DATASET/${DATASET}_test.json" \
    --dataset-id "$DATASET" "$@"
done
