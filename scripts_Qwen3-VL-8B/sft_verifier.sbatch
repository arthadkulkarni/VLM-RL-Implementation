#!/bin/bash
# SFT control: fine-tune the Verifier on CoMET-Bench GT labels for the train
# clips (question_evaluate/sft_verifier.py), then score it on the eval clips
# with validate_pseudo_labels.py. Compare its report against the base and
# self-play runs of validate_pseudo_labels.sbatch with SPLIT=eval.
#   sbatch scripts_Qwen3-VL-8B/sft_verifier.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --job-name=rise-sft-ctrl
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/diag
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export RISE_ATTN_IMPLEMENTATION="${RISE_ATTN_IMPLEMENTATION:-sdpa}"
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
BASE="${MODEL:-${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct}"
SFT_DIR="${SFT_DIR:-${HDD_ROOT}/storage_RISE_sft_control/verifier_sft}"
echo "=== $(date) SFT control from ${BASE} -> ${SFT_DIR}, commit $(git rev-parse --short HEAD) ==="

# Not `torchrun`: its shebang points at the slow hdd copy of the venv.
python3 -m torch.distributed.run --standalone --nproc_per_node 4 question_evaluate/sft_verifier.py \
  --model "${BASE}" --out_dir "${SFT_DIR}" ${SFT_ARGS:-}

python3 question_evaluate/validate_pseudo_labels.py \
  --model "${SFT_DIR}" --name sft_control --split eval \
  --out_dir "${HDD_ROOT}/pseudo_label_validation/sft_control_eval"
