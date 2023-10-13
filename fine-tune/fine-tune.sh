pip install -r requirements.txt

hostfile=""
deepspeed --hostfile=$hostfile fine-tune.py  \
    --report_to "none" \
    --data_path "/opt/ml/input/data/training/belle_chat_ramdon_10k.json" \
    --model_name_or_path "baichuan-inc/Baichuan2-7B-Chat" \
    --output_dir "/opt/ml/model" \
    --model_max_length 512 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 5 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True \
    --use_lora True
