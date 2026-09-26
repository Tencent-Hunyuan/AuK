#!/usr/bin/env bash
# Consistency initializer launcher (stage one). Mirrors scripts/train_dmd.sh.
set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# paths & resources
train_jsonl=${train_jsonl:-agent_space/genshin_zh/train.jsonl}
config=${config:-ckpts/AuK/config.yaml}
teacher_ckpt=${teacher_ckpt:-ckpts/AuK/auk_base.safetensors}
output_dir=${output_dir:-ckpts/auk_cm_init}
num_gpus=${num_gpus:-8}
attn_backend=${attn_backend:-torch}

# InitConfig
max_updates=${max_updates:-500}
learning_rate=${learning_rate:-1e-5}
frames_threshold=${frames_threshold:-1500}
max_samples=${max_samples:-4}
dataloader_num_workers=${dataloader_num_workers:-2}
logging_steps=${logging_steps:-1}
save_per_updates=${save_per_updates:-500}
teacher_cfg_scale=${teacher_cfg_scale:-3.0}
use_cd=${use_cd:-true}

# Use the project venv: pyproject pins torch 2.5.1+cu124, whose NCCL works with
# CUDA 12.4 drivers. A newer torch ships an NCCL that fails multi-GPU init there.
export VIRTUAL_ENV="$REPO/.venv"
export PATH="$REPO/.venv/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
# librosa's numba cache defaults to the package dir. On this shared ceph mount,
# 8 ranks x N workers writing it concurrently raises "Stale file handle".
export NUMBA_CACHE_DIR=/tmp/numba_cache_auk_init
mkdir -p "$NUMBA_CACHE_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$output_dir" logs/init
tag="$(date +'%Y%m%d_%H%M%S')"
log="$REPO/logs/init/${tag}_init.log"
run_sh="$(cd "$output_dir" && pwd)/${tag}_run.sh"

cat <<EOF > "$run_sh"
#!/usr/bin/env bash
set -e
cd $REPO
export VIRTUAL_ENV=$REPO/.venv
export PATH=$REPO/.venv/bin:\$PATH
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
export NUMBA_CACHE_DIR=/tmp/numba_cache_auk_init
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

$REPO/.venv/bin/torchrun --nproc_per_node ${num_gpus} --nnodes 1 \\
  -m auk.train.distill.initializer.train \\
  --train_jsonl ${train_jsonl} \\
  --config ${config} \\
  --teacher_ckpt ${teacher_ckpt} \\
  --output_dir ${output_dir} \\
  --attn_backend ${attn_backend} \\
  --max_updates ${max_updates} \\
  --learning_rate ${learning_rate} \\
  --frames_threshold ${frames_threshold} \\
  --max_samples ${max_samples} \\
  --dataloader_num_workers ${dataloader_num_workers} \\
  --logging_steps ${logging_steps} \\
  --save_per_updates ${save_per_updates} \\
  --teacher_cfg_scale ${teacher_cfg_scale} \\
  --use_cd ${use_cd}
EOF

chmod +x "$run_sh"
echo "run script: $run_sh"
echo "log:        $log"
nohup bash "$run_sh" > "$log" 2>&1 &
echo "pid:        $!"
