#!/bin/bash

for LAYER in {25..31}; do
  TAG=$(printf "L%02d" "$LAYER")
  LOG_FILE="full_run_${TAG}.log"

  echo "============================================================"
  echo "Running attention_layer_ids=[${LAYER}]"
  echo "TAG=${TAG}"
  echo "LOG=${LOG_FILE}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=.:/data2/sjsh/LIBERO \
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /data2/sjsh/checkpoints/openvla-7b-oft-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True \
    --num_trials_per_task 1 \
    --seed 0 \
    --memo "stopword_filtered_text_attn_debug_${TAG}" \
    --keyframe_interval 10 \
    --disable_kv_cache_reuse False \
    --use_cosine_similarity True \
    --include_warp_holes False \
    --attention_layer_ids "[${LAYER}]" \
    --critical_zscore_k 0.12 \
    --fixed_critical_zscore_k 0.12 \
    --wrist_critical_zscore_k 0.12 \
    --use_grasp_aware_critical False \
    --grasp_wrist_dilation_radius 1 \
    --grasp_eef_speed_thr 0.005 \
    --grasp_gripper_close_thr 0.0005 \
    --grasp_gripper_open_thr 0.070 \
    --progressive_pruning_layers "[4,10,14]" \
    --progressive_drop_ratios "[0.2,0.4,0.5]" \
    --use_dynam_cache True \
    --reuse_mode "both" \
    --use_attention_ema True \
    --attention_ema_alpha 0.5 \
    --reset_attention_ema_on_keyframe True \
    --debug_progressive_drop True \
    --candidate_ordering_mode "normattn_global" \
    --use_wrist_ema True \
    --use_fixed_ema False \
    --fixed_warp_similarity_threshold 0.4 \
    --wrist_warp_similarity_threshold 0.4 \
    --use_edge_aware_critical False \
    --edge_boost_weight 0.3 \
    --use_static_boundary_critical False \
    --static_boundary_dilation_radius 1 \
    --use_text_aware_critical False \
    --use_text_aware_debug True \
    --use_stopword_filtered_debug True \
    --use_action_to_text_debug False \
    2>&1 | tee "${LOG_FILE}" | grep --line-buffered -E "ACTION->TEXT DBG|STOPWORD DBG|TEXT-AWARE 3WAY CHECK|TEXT-AWARE 3WAY VIS"

  echo "Finished ${TAG}"
  echo ""
done