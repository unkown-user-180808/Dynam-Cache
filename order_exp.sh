#```bash
#!/bin/bash
# both-view candidate ordering experiments
# wrist EMA on, fixed EMA off
# attention_layer_ids=1, k=0.12
# pruning layers=[4,10,14], ratios=[0.2,0.4,1.0]
# keyframe_interval=2

for ORDERING_MODE in \
  "original" \
  "interleave" \
  "normattn_global" \
  "viewattn_ratioaware" ; do

  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=.:/data2/sjsh/LIBERO python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /data2/sjsh/checkpoints/openvla-7b-oft-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True \
    --num_trials_per_task 10 \
    --seed 0 \
    --memo "${ORDERING_MODE}_both_wrist-0.5_fixed-no_attnL_1_0.12_prun_4-10-14_0.2-0.4-1.0" \
    --keyframe_interval 10 \
    --disable_kv_cache_reuse False \
    --warp_similarity_threshold 0.4 \
    --use_cosine_similarity True \
    --include_warp_holes False \
    --attention_layer_ids "[1]" \
    --critical_zscore_k 0.12 \
    --progressive_pruning_layers "[4,10,14]" \
    --progressive_drop_ratios "[0.2,0.4,1.0]" \
    --use_dynam_cache True \
    --reuse_mode "both" \
    --use_attention_ema True \
    --attention_ema_alpha 0.5 \
    --reset_attention_ema_on_keyframe True \
    --debug_progressive_drop True \
    --candidate_ordering_mode "${ORDERING_MODE}" \
    --use_wrist_ema True \
    --use_fixed_ema False

done
```
