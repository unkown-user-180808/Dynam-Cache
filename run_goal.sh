#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
TASK_SUITE="libero_goal"
CHECKPOINT="/data2/sjsh/checkpoints/openvla-7b-oft-finetuned-libero-goal"
STALE="${STALE:-8}"
PRUNING_LAYERS="[2,6,10]"

EXPERIMENTS=(
  "goal_015_040_100"
  "goal_015_045_100"
  "goal_025_045_090"
  "goal_020_045_095"
  "goal_015_035_100"
  "goal_010_055_100"
)

for EXP in "${EXPERIMENTS[@]}"; do
  case "${EXP}" in

    goal_015_040_100)
      DROP_RATIOS="[0.15,0.40,1.0]"
      MEMO="rolling_anchor_goal_2_6_10_drop_015_040_100_recencycap${STALE}"
      ;;

    goal_015_045_100)
      DROP_RATIOS="[0.15,0.45,1.0]"
      MEMO="rolling_anchor_goal_2_6_10_drop_015_045_100_recencycap${STALE}"
      ;;

    goal_025_045_090)
      DROP_RATIOS="[0.25,0.45,0.90]"
      MEMO="rolling_anchor_goal_2_6_10_drop_025_045_090_recencycap${STALE}"
      ;;

    goal_020_045_095)
      DROP_RATIOS="[0.25,045.,0.95]"
      MEMO="rolling_anchor_goal_2_6_10_drop_020_045_095_recencycap${STALE}"
      ;;

    goal_015_035_100)
      DROP_RATIOS="[0.30,0.55,0.90]"
      MEMO="rolling_anchor_goal_2_6_10_drop_015_035_100_recencycap${STALE}"
      ;;

    goal_010_055_100)
      DROP_RATIOS="[0.30,0.55,0.95]"
      MEMO="rolling_anchor_goal_2_6_10_drop_010_055_100_recencycap${STALE}"
      ;;

    *)
      echo "Unknown experiment: ${EXP}"
      exit 1
      ;;
  esac

  echo "============================================================"
  echo "Running ${TASK_SUITE} | ${EXP}"
  echo "GPU=${GPU}"
  echo "PRUNING_LAYERS=${PRUNING_LAYERS}"
  echo "DROP_RATIOS=${DROP_RATIOS}"
  echo "MEMO=${MEMO}"
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
    --use_wrist_ema True \
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
    --critical_attention_mode content_words \
    --use_recency_cap True \
    --stale_force_threshold "${STALE}" \
    --reuse_hole_using_attention False
done