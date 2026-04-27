#!/usr/bin/env bash
# serve/ollama_load.sh
#
# Converts the merged AlignR model to GGUF format and loads it into Ollama.
# Requires: ollama, llama.cpp (for conversion), Python 3.10+
#
# Usage:
#   bash serve/ollama_load.sh --model ./aligned_model/ --name alignr-v1.0.0

set -euo pipefail

MODEL_PATH=""
MODEL_NAME=""
QUANTISATION="q4_k_m"   # good balance of quality and size for consumer hardware

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --name)  MODEL_NAME="$2"; shift 2 ;;
    --quant) QUANTISATION="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$MODEL_NAME" ]]; then
  echo "Usage: bash serve/ollama_load.sh --model <path> --name <name>"
  exit 1
fi

echo "AlignR · Ollama load script"
echo "Model:  $MODEL_PATH"
echo "Name:   $MODEL_NAME"
echo "Quant:  $QUANTISATION"
echo ""

# Check Ollama is running
if ! ollama list &>/dev/null; then
  echo "ERROR: Ollama is not running."
  echo "Start it with: ollama serve"
  exit 1
fi

# Check llama.cpp convert script is available
CONVERT_SCRIPT=""
for candidate in \
  "$(pip show llama-cpp-python 2>/dev/null | grep Location | awk '{print $2}')/llama_cpp/convert.py" \
  "./llama.cpp/convert.py" \
  "/usr/local/lib/llama.cpp/convert.py"; do
  if [[ -f "$candidate" ]]; then
    CONVERT_SCRIPT="$candidate"
    break
  fi
done

GGUF_PATH="${MODEL_PATH%/}_${QUANTISATION}.gguf"

if [[ -f "$GGUF_PATH" ]]; then
  echo "GGUF file already exists: $GGUF_PATH"
  echo "Skipping conversion."
else
  if [[ -z "$CONVERT_SCRIPT" ]]; then
    echo "llama.cpp convert.py not found. Attempting to install llama-cpp-python..."
    pip install llama-cpp-python --quiet
    CONVERT_SCRIPT="$(python -c "import llama_cpp; import os; print(os.path.join(os.path.dirname(llama_cpp.__file__), 'convert.py'))" 2>/dev/null || echo "")"
  fi

  if [[ -z "$CONVERT_SCRIPT" ]]; then
    echo ""
    echo "FALLBACK: Could not find llama.cpp convert.py."
    echo "Manual steps:"
    echo "  1. git clone https://github.com/ggerganov/llama.cpp"
    echo "  2. python llama.cpp/convert.py $MODEL_PATH --outfile $GGUF_PATH --outtype $QUANTISATION"
    echo "  3. Re-run this script."
    exit 1
  fi

  echo "Converting to GGUF ($QUANTISATION)..."
  python "$CONVERT_SCRIPT" "$MODEL_PATH" \
    --outfile "$GGUF_PATH" \
    --outtype "$QUANTISATION"
  echo "Conversion complete: $GGUF_PATH"
fi

# Write Modelfile
MODELFILE_PATH="/tmp/alignr_modelfile"
cat > "$MODELFILE_PATH" <<EOF
FROM $GGUF_PATH

SYSTEM """
You are an AI assistant aligned for regulated enterprise use. You follow institutional
guidelines strictly. You do not give legal, medical, or financial advice. You escalate
queries that require human judgment. You are transparent about the limits of your role.
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
EOF

echo ""
echo "Loading into Ollama as '$MODEL_NAME'..."
ollama create "$MODEL_NAME" -f "$MODELFILE_PATH"

echo ""
echo "Done! Run the model with:"
echo "  ollama run $MODEL_NAME"
echo ""
echo "Or use the comparison notebook:"
echo "  jupyter notebook notebooks/04_demo_comparison.ipynb"
