# DRBO

To reproduce Table 1, please run
```bash
python main.py \
    --prompt_file prompts/eli5_default.json \
    --eval_file data/eli5_eval_bm25_top100_reranked_oracle.json \
    --dataset_name eli5 \
    --model_name_or_path  \
    --temperature 0.95 \
    --top_p 0.95 \
    --kl_ctl 0.8 \
    --actor_learning_rate 1e-6 \
    --max_new_tokens 300 \
    --tau 0 \
    --log_file logs/$name\_train.log \
    --train_steps 100 \
    --experiment_description "llama_2_13b eli5, $name" \
    --delta_reward true \

python main.py \
    --prompt_file prompts/eli5_default.json \
    --eval_file data/eli5_eval_bm25_top100_reranked_oracle.json \
    --dataset_name eli5 \
    --model_name_or_path  \
    --temperature 0.95 \
    --top_p 0.95 \
    --kl_ctl 0.8 \
    --actor_learning_rate 1e-6 \
    --max_new_tokens 300 \
    --tau 0 \
    --log_file logs/$name\_train.log \
    --train_steps 100 \
    --experiment_description "llama_2_13b eli5, $name" \
    --delta_reward false \
```

To reproduce ALaRM baseline, please run 
```bash
cd alce
python main_alarm.py
```
