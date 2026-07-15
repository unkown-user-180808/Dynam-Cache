"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor


json_numpy.patch()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.constants import (
ACTION_DIM,
ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

from experiments.robot.libero.attention_utils import (
token_attention_merge,
spatial_scores_to_map,
update_attention_ema,
get_content_word_row_groups,
token_attention_merge_word_groups,
)


DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224


np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

def clean_attentions(attns):
    if attns is None:
        return None

    if torch.is_tensor(attns):
        if attns.dim() == 4 and torch.is_floating_point(attns):
            return (attns,)
        return None

    cleaned = []
    has_valid = False

    for x in attns:
        if torch.is_tensor(x) and x.dim() == 4 and torch.is_floating_point(x):
            cleaned.append(x)
            has_valid = True
        else:
            cleaned.append(None)

    return tuple(cleaned) if has_valid else None

def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False

def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    if path1.stat().st_size != path2.stat().st_size:
        return False

    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.


    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.


    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]

def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.


    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_vla(cfg: Any) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    """
    Apply FiLM (Feature-wise Linear Modulation) to the VLA vision backbone.


    Args:
        vla: The VLA model
        cfg: Configuration object with model parameters

    Returns:
        torch.nn.Module: VLA model with FiLM applied
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)

    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.


    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.


    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    """
    Get proprioception projector for the VLA model.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model
        proprio_dim: Dimension of proprioception data

    Returns:
        ProprioProjector: The initialized proprio projector
    """
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "proprio_projector--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "proprio_projector--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
    """
    Get noisy action projector for diffusion-based action prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        NoisyActionProjector: The initialized noisy action projector
    """
    noisy_action_projector = NoisyActionProjector(
        llm_dim=llm_dim,
    ).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()

    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "noisy_action_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    noisy_action_projector.load_state_dict(state_dict)

    return noisy_action_projector

def get_action_head(cfg: Any, llm_dim: int) -> Union[L1RegressionActionHead, DiffusionActionHead]:
    """
    Get action head for continuous value prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        Union[L1RegressionActionHead, DiffusionActionHead]: The initialized action head

    Raises:
        AssertionError: If both L1 regression and diffusion are specified
    """
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot use both L1 regression and diffusion action head!"

    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_diffusion_steps_train=cfg.num_diffusion_steps_train
        )
        action_head.noise_scheduler.set_timesteps(cfg.num_diffusion_steps_inference)
    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "action_head--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "action_head--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_action_head_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

    return action_head

def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()

def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = crop_and_resize(image, crop_scale, batch_size)

    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    return Image.fromarray(image.numpy()).convert("RGB")

def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        check_image_format(image)

        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        pil_image = Image.fromarray(image).convert("RGB")

        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    last_caches: Optional[dict] = None,
    warped_cache=None,
) -> List[np.ndarray]:
    """Generate one OpenVLA-OFT action chunk with the minimum cache side-channel."""

    def _predict(inputs, proprio):
        common = dict(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
        )
        if action_head is None:
            return vla.predict_action(**common)
        return vla.predict_action(
            **common,
            proprio=proprio,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            action_head=action_head,
            use_film=use_film,
        )

    with torch.inference_mode():
        images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            images.extend(obs[k] for k in obs if "wrist" in k)
        images = prepare_images_for_vla(images, cfg)

        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)

        if len(images) > 1:
            wrist_pixel_values = [
                processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)["pixel_values"]
                for image in images[1:]
            ]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], *wrist_pixel_values], dim=1
            )

        proprio = None
        if cfg.use_proprio:
            proprio_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(obs["state"], proprio_stats)

        if not getattr(cfg, "use_dynam_cache", True):
            action, _ = _predict(inputs, proprio)
            return [action[i] for i in range(len(action))]

        incoming_caches = last_caches if last_caches is not None else {}
        if cfg.disable_kv_cache_reuse:
            warped_cache = None

        lm_config = vla.language_model.config
        llama_model = vla.language_model.model
        attention_layer_ids = tuple(
            int(i) for i in getattr(cfg, "attention_layer_ids", (1,))
        )

        old_state = {
            "warped_past_key_values": getattr(lm_config, "warped_past_key_values", None),
            "force_cache_output": getattr(lm_config, "force_cache_output", False),
            "force_attention_output": getattr(lm_config, "force_attention_output", False),
            "force_hidden_states_output": getattr(lm_config, "force_hidden_states_output", False),
            "collect_attn_layers": getattr(lm_config, "collect_attn_layers", None),
        }

        lm_config.warped_past_key_values = warped_cache
        lm_config.force_cache_output = True
        lm_config.force_attention_output = True
        lm_config.force_hidden_states_output = False
        lm_config.collect_attn_layers = list(attention_layer_ids)

        try:
            action, _ = _predict(inputs, proprio)
            new_cache = getattr(llama_model, "last_forward_cache", None)
            raw_attentions = getattr(llama_model, "last_forward_attentions", None)
            kept_query_positions = getattr(
                llama_model, "last_forward_kept_query_positions", None
            )
        finally:
            for name, value in old_state.items():
                setattr(lm_config, name, value)

        prev_fixed_ema = incoming_caches.get("ema_fixed_spatial_map")
        prev_wrist_ema = incoming_caches.get("ema_wrist_spatial_map")
        prev_ema_step = int(incoming_caches.get("ema_attention_step", 0))
        cached_content_word_groups = incoming_caches.get("content_word_groups")

        incoming_caches.clear()
        incoming_caches["past_key_values"] = new_cache
        incoming_caches["kept_query_positions"] = kept_query_positions

        attns = clean_attentions(raw_attentions)
        if attns is not None:
            spatial_layer_ids = [
                layer_id for layer_id in attention_layer_ids
                if 0 <= layer_id < len(attns) and attns[layer_id] is not None
            ]
            if not spatial_layer_ids:
                raise RuntimeError(
                    f"No requested attention layer was collected: {attention_layer_ids}"
                )

            num_patches_per_image = 256
            fixed_token_start = 1
            wrist_token_start = fixed_token_start + num_patches_per_image
            num_image_tokens = num_patches_per_image * int(cfg.num_images_in_input)
            query_token_start = (
                fixed_token_start
                + num_image_tokens
                + (1 if cfg.use_proprio else 0)
            )

            attention_kept_positions = kept_query_positions
            pruning_layers = getattr(lm_config, "progressive_pruning_layers", None)
            if pruning_layers:
                progressive_start_layer = min(int(x) for x in pruning_layers)
                if all(layer_id < progressive_start_layer for layer_id in spatial_layer_ids):
                    attention_kept_positions = None

            prompt_token_count = int(inputs["input_ids"].shape[1])
            text_query_end = query_token_start + max(prompt_token_count - 1, 0)
            mode = str(getattr(cfg, "critical_attention_mode", "mixed"))

            merge_kwargs = dict(
                multihead_attention=attns,
                layer_ids=spatial_layer_ids,
                kept_query_positions=attention_kept_positions,
                num_key_tokens=num_patches_per_image,
            )

            if mode == "content_words":
                word_groups = cached_content_word_groups
                if word_groups is None:
                    text_token_ids = inputs["input_ids"][0, 1:prompt_token_count].tolist()
                    text_token_strs = processor.tokenizer.convert_ids_to_tokens(text_token_ids)
                    word_groups = get_content_word_row_groups(
                        text_token_strs=text_token_strs,
                        text_token_start=query_token_start,
                    )
                incoming_caches["content_word_groups"] = word_groups
                fixed_scores = token_attention_merge_word_groups(
                    **merge_kwargs,
                    word_groups=word_groups,
                    key_token_start=fixed_token_start,
                )
                wrist_scores = token_attention_merge_word_groups(
                    **merge_kwargs,
                    word_groups=word_groups,
                    key_token_start=wrist_token_start,
                )
            else:
                if mode == "mixed":
                    q_start, q_end = query_token_start, None
                elif mode == "text_only":
                    q_start, q_end = query_token_start, text_query_end
                elif mode == "status_only":
                    if not cfg.use_proprio:
                        raise ValueError("status_only requires cfg.use_proprio=True")
                    q_start, q_end = query_token_start - 1, query_token_start
                elif mode == "action_only":
                    q_start, q_end = text_query_end, None
                else:
                    raise ValueError(f"Unknown critical_attention_mode: {mode}")

                fixed_scores = token_attention_merge(
                    **merge_kwargs,
                    key_token_start=fixed_token_start,
                    query_token_start=q_start,
                    query_token_end=q_end,
                )
                wrist_scores = token_attention_merge(
                    **merge_kwargs,
                    key_token_start=wrist_token_start,
                    query_token_start=q_start,
                    query_token_end=q_end,
                )

            fixed_map = spatial_scores_to_map(fixed_scores, device=DEVICE).detach().cpu()
            wrist_map = spatial_scores_to_map(wrist_scores, device=DEVICE).detach().cpu()
            incoming_caches["latest_fixed_spatial_map"] = fixed_map
            incoming_caches["latest_wrist_spatial_map"] = wrist_map

            if getattr(cfg, "use_attention_ema", False):
                alpha = float(np.clip(getattr(cfg, "attention_ema_alpha", 0.35), 0.0, 1.0))
                incoming_caches["ema_fixed_spatial_map"] = (
                    update_attention_ema(prev_fixed_ema, fixed_map, alpha)
                    if getattr(cfg, "use_fixed_ema", True)
                    else fixed_map
                )
                incoming_caches["ema_wrist_spatial_map"] = (
                    update_attention_ema(prev_wrist_ema, wrist_map, alpha)
                    if getattr(cfg, "use_wrist_ema", True)
                    else wrist_map
                )
                incoming_caches["ema_attention_step"] = prev_ema_step + 1

        return [action[i] for i in range(len(action))]

def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
    ) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()