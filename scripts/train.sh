#!/usr/bin/env bash
set -e

# paths & resources
train_jsonl=agent_space/genshin_zh/train.jsonl
val_jsonl=agent_space/genshin_zh/val.jsonl
config=ckpts/AuK/config.yaml
init_ckpt=ckpts/AuK/auk_base.safetensors
output_dir=ckpts/auk_finetune
num_gpus=8

# TrainConfig
learning_rate=2e-5
max_grad_norm=1.0
num_train_epochs=200
gradient_accumulation_steps=1
warmup_steps=100
frames_threshold=2700
max_samples=8
ema_beta=0.9999
ema_update_after_step=100
ema_update_every=10
save_per_updates=1000
last_per_updates=1000
logging_steps=10
val_per_updates=200
dataloader_num_workers=4
seed=666

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$output_dir"
tag="$(date +'%Y%m%d_%H%M%S')"
run_sh="$output_dir/${tag}_run.sh"

cat <<EOF > "$run_sh"
#!/usr/bin/env bash
set -e

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

accelerate launch --num_processes ${num_gpus} --num_machines 1 --mixed_precision bf16 \\
  -m auk.train.train \\
  --train_jsonl ${train_jsonl} \\
  --val_jsonl ${val_jsonl} \\
  --config ${config} \\
  --init_ckpt ${init_ckpt} \\
  --output_dir ${output_dir} \\
  --learning_rate ${learning_rate} \\
  --max_grad_norm ${max_grad_norm} \\
  --num_train_epochs ${num_train_epochs} \\
  --gradient_accumulation_steps ${gradient_accumulation_steps} \\
  --warmup_steps ${warmup_steps} \\
  --frames_threshold ${frames_threshold} \\
  --max_samples ${max_samples} \\
  --ema_beta ${ema_beta} \\
  --ema_update_after_step ${ema_update_after_step} \\
  --ema_update_every ${ema_update_every} \\
  --save_per_updates ${save_per_updates} \\
  --last_per_updates ${last_per_updates} \\
  --logging_steps ${logging_steps} \\
  --val_per_updates ${val_per_updates} \\
  --dataloader_num_workers ${dataloader_num_workers} \\
  --seed ${seed}
EOF

chmod +x "$run_sh"
echo "run script saved to $run_sh"
bash "$run_sh"
