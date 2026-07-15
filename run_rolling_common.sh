#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-5}"
STALE="${STALE:-8}"
PRUNING_LAYERS="[2,6,10]"
DROP_RATIOS="[0.20,0.45,0.60]"

# 너무 오래 걸리면 여기서 원하는 task만 남기면 됨
TASKS=(  
  # "libero_goal"
  # "libero_object"
  # "libero_spatial"
  "libero_10"
)

EXPERIMENTS=(
  "mixed_wrist_ema"
  "content_no_wrist_ema"
  "content_wrist_ema"
  "mixed_no_wrist_ema"

)

for TASK_SUITE in "${TASKS[@]}"; do
  case "${TASK_SUITE}" in
    
    # libero_goal)
    #   CHECKPOINT="/data2/sjsh/checkpoints/openvla-7b-oft-finetuned-libero-goal"
    #   ;;
    # libero_object)
    #   CHECKPOINT="/data2/sjsh/checkpoints/openvla-7b-oft-finetuned-libero-object"
    #   ;;
    # libero_spatial)
    #   CHECKPOINT="/data2/sjsh/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
    #   ;;    
    libero_10)
      CHECKPOINT="/data2/sjsh/checkpoints/openvla-7b-oft-finetuned-libero-10"
      ;;
    *)
      echo "Unknown TASK_SUITE: ${TASK_SUITE}"
      exit 1
      ;;
  esac

  for EXP in "${EXPERIMENTS[@]}"; do
    case "${EXP}" in
      mixed_no_wrist_ema)
        CRITICAL_ATTENTION_MODE="mixed"
        USE_WRIST_EMA="False"
        MEMO="rolling_anchor_${TASK_SUITE}_ablate_mixed_no_wristema_2_6_10_drop020_045_060_recencycap${STALE}"
        ;;

      mixed_wrist_ema)
        CRITICAL_ATTENTION_MODE="mixed"
        USE_WRIST_EMA="True"
        MEMO="rolling_anchor_${TASK_SUITE}_ablate_mixed_wristema_2_6_10_drop020_045_060_recencycap${STALE}"
        ;;

      content_no_wrist_ema)
        CRITICAL_ATTENTION_MODE="content_words"
        USE_WRIST_EMA="False"
        MEMO="rolling_anchor_${TASK_SUITE}_ablate_content_no_wristema_2_6_10_drop020_045_060_recencycap${STALE}"
        ;;

      content_wrist_ema)
        CRITICAL_ATTENTION_MODE="content_words"
        USE_WRIST_EMA="True"
        MEMO="rolling_anchor_${TASK_SUITE}_ablate_content_wristema_2_6_10_drop020_045_060_recencycap${STALE}"
        ;;

      *)
        echo "Unknown experiment: ${EXP}"
        exit 1
        ;;
    esac

    echo "============================================================"
    echo "Running task      : ${TASK_SUITE}"
    echo "Experiment        : ${EXP}"
    echo "GPU               : ${GPU}"
    echo "Checkpoint        : ${CHECKPOINT}"
    echo "Critical mode     : ${CRITICAL_ATTENTION_MODE}"
    echo "Use wrist EMA     : ${USE_WRIST_EMA}"
    echo "Pruning layers    : ${PRUNING_LAYERS}"
    echo "Drop ratios       : ${DROP_RATIOS}"
    echo "Stale threshold   : ${STALE}"
    echo "Memo              : ${MEMO}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=.:/data2/sjsh/LIBERO \
    python -u experiments/robot/libero/run_libero_eval.py \
      --model_family openvla \
      --pretrained_checkpoint "${CHECKPOINT}" \
      --task_suite_name "${TASK_SUITE}" \
      --center_crop True \
      --num_trials_per_task 10 \
      --seed 0 \
      --memo "${MEMO}" \
      --use_rolling_anchor True \
      --keyframe_interval 0 \
      --warping_mode homography \
      --disable_kv_cache_reuse False \
      --use_cosine_similarity True \
      --include_warp_holes False \
      --attention_layer_ids "[1]" \
      --critical_zscore_k 0.12 \
      --fixed_critical_zscore_k 0.20 \
      --wrist_critical_zscore_k 0.12 \
      --use_grasp_aware_critical False \
      --grasp_wrist_dilation_radius 1 \
      --grasp_eef_speed_thr 0.005 \
      --grasp_gripper_close_thr 0.0005 \
      --grasp_gripper_open_thr 0.070 \
      --progressive_pruning_layers "${PRUNING_LAYERS}" \
      --progressive_drop_ratios "${DROP_RATIOS}" \
      --use_dynam_cache True \
      --reuse_mode both \
      --use_attention_ema True \
      --attention_ema_alpha 0.5 \
      --reset_attention_ema_on_keyframe True \
      --debug_progressive_drop True \
      --candidate_ordering_mode normattn_global \
      --use_wrist_ema "${USE_WRIST_EMA}" \
      --use_fixed_ema False \
      --fixed_warp_similarity_threshold 0.4 \
      --wrist_warp_similarity_threshold 0.4 \
      --use_edge_aware_critical False \
      --edge_boost_weight 0.3 \
      --use_static_boundary_critical False \
      --static_boundary_dilation_radius 1 \
      --use_text_aware_critical False \
      --use_text_aware_debug False \
      --use_stopword_filtered_debug False \
      --use_action_to_text_debug False \
      --use_query_mode_map_debug False \
      --critical_attention_mode "${CRITICAL_ATTENTION_MODE}" \
      --use_recency_cap True \
      --stale_force_threshold "${STALE}" \
      --reuse_hole_using_attention False
  done
done