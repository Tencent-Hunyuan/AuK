#!/usr/bin/env bash
set -e
# paths & resources
train_jsonl=/path/to/train.jsonl
val_jsonl=/path/to/val.jsonl
config=ckpts/AuK/config.yaml
init_ckpt=ckpts/AuK/auk_base.safetensors
output_dir=ckpts/auk_lora
num_gpus=8
# LoRA
lora_r=32
lora_alpha=32
lora_dropout=0.0
lora_targets=""
# TrainConfig
learning_rate=5e-5
max_grad_norm=1.0
num_train_epochs=1
gradient_accumulation_steps=1
warmup_steps=100
frames_threshold=2700
max_samples=4
save_per_updates=500
last_per_updates=500
logging_steps=10
val_per_updates=200
dataloader_num_workers=4
seed=666
if [ -n "$lora_targets" ]; then
    lora_targets_arg="--lora_targets $lora_targets"
else
    lora_targets_arg=""
fi
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
mkdir -p "$output_dir"
tag="$(date +'%Y%m%d_%H%M%S')"
run_sh="$output_dir/${tag}_run.sh"
cat <<EOF > "$run_sh"
#!/usr/bin/env bash
set -e
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
accelerate launch --num_processes ${num_gpus} --num_machines 1 --use_deepspeed \\
  --deepspeed_config_file scripts/ds_zero3.json \\
  -m auk.train.lora_train \\
  --train_jsonl ${train_jsonl} \\
  --val_jsonl ${val_jsonl} \\
  --config ${config} \\
  --init_ckpt ${init_ckpt} \\
  --output_dir ${output_dir} \\
  --lora_r ${lora_r} \\
  --lora_alpha ${lora_alpha} \\
  --lora_dropout ${lora_dropout} \\
  ${lora_targets_arg} \\
  --learning_rate ${learning_rate} \\
  --max_grad_norm ${max_grad_norm} \\
  --num_train_epochs ${num_train_epochs} \\
  --gradient_accumulation_steps ${gradient_accumulation_steps} \\
  --warmup_steps ${warmup_steps} \\
  --frames_threshold ${frames_threshold} \\
  --max_samples ${max_samples} \\
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
