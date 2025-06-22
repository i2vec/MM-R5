cd src/open-r1-multimodal

export DEBUG_MODE="true"

RUN_NAME="rl_rerank"
export LOG_PATH="./logs/debug_log_$RUN_NAME.txt"
export PYTHONPATH="/the/path/to/your/repo/src/open-r1-multimodal/src/:$PYTHONPATH"


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12348" \
    src/open_r1/grpo_rerank_v4.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /the/path/to/your/model \
    --dataset_name /the/path/to/your/dataset \
    --max_prompt_length 32768 \
    --max_completion_length 2048 \
    --num_generations 4 \
    --num_iterations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 0 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --deepspeed local_scripts/zero2.json
