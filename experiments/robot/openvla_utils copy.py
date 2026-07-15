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
import torch.nn.functional as F

# Apply JSON numpy patch for serialization

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
get_layer_mask_schedule,
# compute_hidden_sim_maps,
compute_hidden_norm_maps,
update_attention_ema,
get_content_word_row_groups,
token_attention_merge_word_groups,
)

# Initialize important constants

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings

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
    # If the API call below runs without error, the model is on the hub
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

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
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

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
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
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
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

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
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

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
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

    # If loading a locally stored pretrained checkpoint, check whether config or model files
    # need to be synced so that any changes the user makes to the VLA modeling code will
    # actually go into effect
    # If loading a pretrained checkpoint from Hugging Face Hub, we just assume that the policy
    # will be used as is, with its original modeling logic
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    # Load the model
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action normalization
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

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    # Create and apply FiLMed vision backbone
    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    # Load vision backbone checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    # Use the model component instead of wrapper and convert to bfloat16
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
        # Download dataset stats directly from HF Hub
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
    # Initialize projector and move to device
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
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
        # Download proprio projector directly from HF Hub
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
    # Initialize projector and move to device
    noisy_action_projector = NoisyActionProjector(
        llm_dim=llm_dim,
    ).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()

    # Find and load checkpoint
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

    # Initialize appropriate action head based on configuration
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_diffusion_steps_train=cfg.num_diffusion_steps_train
        )
        # Set number of diffusion steps for inference
        action_head.noise_scheduler.set_timesteps(cfg.num_diffusion_steps_inference)
    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
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
        # Download proprio projector directly from HF Hub
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

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
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
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
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

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
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

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
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
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
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
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
            #all_images.append(obs["wrist_image"])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)

        # Build VLA prompt
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process primary image
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            # Concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        # Generate action
        use_dynam_cache = getattr(cfg, "use_dynam_cache", True)
        if use_dynam_cache:
            incoming_caches = last_caches if last_caches is not None else {}

            if cfg.disable_kv_cache_reuse:
                warped_cache = None

            lm_config = vla.language_model.config
            llama_model = vla.language_model.model

            old_warped_cache = getattr(lm_config, "warped_past_key_values", None)
            old_force_cache = getattr(lm_config, "force_cache_output", False)
            old_force_attn = getattr(lm_config, "force_attention_output", False)
            old_collect_attn_layers = getattr(lm_config, "collect_attn_layers", None)

            num_hidden_layers = getattr(lm_config, "num_hidden_layers", None)
            if num_hidden_layers is None and hasattr(llama_model, "layers"):
                num_hidden_layers = len(llama_model.layers)

            last_layer_id = None if num_hidden_layers is None else num_hidden_layers - 1

            attention_layer_ids = [
                int(i) for i in getattr(cfg, "attention_layer_ids", (1,))
            ]

            # collect_attn_layers = attention_layer_ids
            # if last_layer_id is not None:
            #     collect_attn_layers = sorted(set(collect_attn_layers + [last_layer_id]))

            if cfg.disable_kv_cache_reuse:
                collect_attn_layers = None  # 전체 레이어
            else:
                collect_attn_layers = attention_layer_ids
                if last_layer_id is not None:
                    collect_attn_layers = sorted(set(collect_attn_layers + [last_layer_id]))

            lm_config.warped_past_key_values = warped_cache
            lm_config.force_cache_output = True
            lm_config.force_attention_output = True
            lm_config.force_hidden_states_output = True
            lm_config.collect_attn_layers = collect_attn_layers

            lm_config.force_hidden_states_output = True # extract last hidden states

            try:
                if action_head is None:
                    action, _ = vla.predict_action(
                        **inputs,
                        unnorm_key=cfg.unnorm_key,
                        do_sample=False,
                    )
                else:
                    action, _ = vla.predict_action(
                        **inputs,
                        unnorm_key=cfg.unnorm_key,
                        do_sample=False,
                        proprio=proprio,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        action_head=action_head,
                        use_film=use_film,
                    )

                # get_vla_action 안, predict_action 호출 직전이나 직후에
                print(f"[TOKEN DEBUG] input_ids shape: {inputs['input_ids'].shape}")
                print(f"[TOKEN DEBUG] task_label: '{task_label}'")
                print(f"[TOKEN DEBUG] tokenized text only: {processor.tokenizer(task_label, return_tensors='pt')['input_ids'].shape}")

                print(f"[TOKEN DEBUG] final inputs_embeds seq_len 추정용:")
                print(f"  input_ids: {inputs['input_ids']}")  # 실제 토큰 ID 시퀀스 출력 (숫자들)
                print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
                new_caches = {
                    "past_key_values": getattr(llama_model, "last_forward_cache", None),
                    "attentions": getattr(llama_model, "last_forward_attentions", None),
                    "kept_query_positions": getattr(llama_model, "last_forward_kept_query_positions", None),
                    "final_hidden_states": getattr(llama_model, "last_forward_final_hidden_states", None), # extract last hidden states
                }

                # 임시 확인 코드
                # fh = new_caches.get("final_hidden_states")
                # if fh is not None:
                #     print(f"[DEBUG] final_hidden_states shape: {fh.shape}")
                #     print(f"[DEBUG] final_hidden_states dtype: {fh.dtype}")
                #     print(f"[DEBUG] final_hidden_states min/max: {fh.min():.4f} / {fh.max():.4f}")
                # else:
                #     print("[DEBUG] final_hidden_states is None ← 문제!")
            finally:
                lm_config.warped_past_key_values = old_warped_cache
                lm_config.force_cache_output = old_force_cache
                lm_config.force_attention_output = old_force_attn
                lm_config.collect_attn_layers = old_collect_attn_layers

            # ------ [EMA state] ------------
            prev_fixed_ema = incoming_caches.get("ema_fixed_spatial_map")
            prev_wrist_ema = incoming_caches.get("ema_wrist_spatial_map")
            prev_ema_step = incoming_caches.get("ema_attention_step", 0)
            # -------[EMA state] -----------

            incoming_caches.clear()
            incoming_caches.update(new_caches)

            if prev_fixed_ema is not None:
                incoming_caches["prev_ema_fixed_spatial_map"] = prev_fixed_ema
            if prev_wrist_ema is not None:
                incoming_caches["prev_ema_wrist_spatial_map"] = prev_wrist_ema
            incoming_caches["prev_ema_attention_step"] = prev_ema_step

            last_caches = incoming_caches

            attns = clean_attentions(last_caches.get("attentions"))
            kept_query_positions = last_caches.get("kept_query_positions", None)

            attns = clean_attentions(last_caches.get("attentions"))
            # print(f"[DEBUG3] attentions raw: {last_caches.get('attentions') is not None}")
            # print(f"[DEBUG3] attns after clean: {attns is not None}")

            if attns is not None:

                # print(f"[DEBUG] len(attns)={len(attns)}, last_layer_id={last_layer_id}")
                # print(f"[DEBUG] attns[last_layer_id] is None: {attns[last_layer_id] is None}")

                last_caches["attentions"] = attns

                # pruning_layers = getattr(vla.language_model.config, "progressive_pruning_layers", None)
                # if len(attns) > 1 and pruning_layers is not None:
                #     all_layer_ratios = get_layer_mask_schedule(attns)
                #     if all_layer_ratios is not None:
                #         selected_ratios = [
                #             float(all_layer_ratios[min(int(layer_id), len(all_layer_ratios) - 1)])
                #             for layer_id in pruning_layers
                #         ]
                #         vla.language_model.config.progressive_drop_ratios = selected_ratios

                spatial_layer_ids = [
                    int(i) for i in getattr(cfg, "attention_layer_ids", (1,))
                    if int(i) < len(attns)
                ]
                if len(spatial_layer_ids) == 0:
                    spatial_layer_ids = list(range(len(attns)))

                num_patches_per_image = 256
                fixed_token_start = 1
                wrist_token_start = fixed_token_start + num_patches_per_image

                num_image_tokens = num_patches_per_image * int(cfg.num_images_in_input)
                num_extra_projected_tokens = 1 if cfg.use_proprio else 0
                query_token_start = fixed_token_start + num_image_tokens + num_extra_projected_tokens

                attention_kept_query_positions = kept_query_positions

                pruning_layers = getattr(vla.language_model.config, "progressive_pruning_layers", None)
                if pruning_layers is not None and len(pruning_layers) > 0:
                    progressive_start_layer = min(int(x) for x in pruning_layers)

                    # If all collected attention layers are before the first pruning layer,
                    # their attention maps still use the full unpruned query layout.
                    if all(int(layer_id) < progressive_start_layer for layer_id in spatial_layer_ids):
                        attention_kept_query_positions = None

                # ----------------- text-aware / query-mode attention -----------------
                use_text_aware_critical = getattr(cfg, "use_text_aware_critical", False)
                use_text_aware_debug = getattr(cfg, "use_text_aware_debug", False)

                critical_attention_mode = getattr(cfg, "critical_attention_mode", "mixed")
                use_query_mode_map_debug = getattr(cfg, "use_query_mode_map_debug", False)

                need_query_mode_scores = (
                    use_query_mode_map_debug
                    or critical_attention_mode != "mixed"
                )

                need_text_query_end = (
                    use_text_aware_critical
                    or use_text_aware_debug
                    or getattr(cfg, "use_action_to_text_debug", False)
                    or getattr(cfg, "use_stopword_filtered_debug", False)
                    or need_query_mode_scores
                )

                text_query_end = None
                if need_text_query_end:
                    prompt_token_count = inputs["input_ids"].shape[1]
                    text_query_end = query_token_start + (prompt_token_count - 1)

                # 기존 mixed map에서 text-aware critical만 쓰고 싶을 때만 text end 적용
                main_query_token_end = text_query_end if use_text_aware_critical else None
                # ----------------- text-aware / query-mode attention -----------------

                fixed_spatial_scores = token_attention_merge(
                    multihead_attention=attns,
                    layer_ids=spatial_layer_ids,
                    kept_query_positions=attention_kept_query_positions,
                    key_token_start=fixed_token_start,
                    num_key_tokens=num_patches_per_image,
                    query_token_start=query_token_start,
                    query_token_end=main_query_token_end,
                )

                wrist_spatial_scores = token_attention_merge(
                    multihead_attention=attns,
                    layer_ids=spatial_layer_ids,
                    kept_query_positions=attention_kept_query_positions,
                    key_token_start=wrist_token_start,
                    num_key_tokens=num_patches_per_image,
                    query_token_start=query_token_start,
                    query_token_end=main_query_token_end,
                )

                last_caches["latest_fixed_spatial_map"] = spatial_scores_to_map(
                    fixed_spatial_scores,
                    device=DEVICE,
                ).detach().cpu()

                last_caches["latest_wrist_spatial_map"] = spatial_scores_to_map(
                    wrist_spatial_scores,
                    device=DEVICE,
                ).detach().cpu()

                if use_text_aware_debug and text_query_end is not None:
                    fixed_text_only_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=fixed_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=query_token_start,
                        query_token_end=text_query_end,
                    )
                    wrist_text_only_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=wrist_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=query_token_start,
                        query_token_end=text_query_end,
                    )
                    last_caches["latest_fixed_text_only_map"] = spatial_scores_to_map(
                        fixed_text_only_scores, device=DEVICE,
                    ).detach().cpu()
                    last_caches["latest_wrist_text_only_map"] = spatial_scores_to_map(
                        wrist_text_only_scores, device=DEVICE,
                    ).detach().cpu()

                # ----------------- action→text 중요도 1차 검증 (로그 전용) ----------------------
                if getattr(cfg, "use_action_to_text_debug", False) and text_query_end is not None:
                    prompt_token_count = inputs["input_ids"].shape[1]
                    action_token_start = text_query_end  # = query_token_start + (prompt_token_count - 1)
                    text_token_count = prompt_token_count - 1  # text_query_end - query_token_start와 동일

                    # action 토큰들이 각 text 토큰에 얼마나 주목하는지 (key=text 구간)
                    action_to_text_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=query_token_start,   # text 구간 시작 (=514)
                        num_key_tokens=text_token_count,      # text 토큰 개수
                        query_token_start=action_token_start, # action 토큰부터
                        query_token_end=None,                 # 끝까지(=stop 토큰까지)
                    )

                    # 토큰 문자열과 점수를 같이 출력
                    text_token_ids = inputs["input_ids"][0, 1:1+text_token_count].tolist()  # BOS 제외, text 구간만
                    text_token_strs = processor.tokenizer.convert_ids_to_tokens(text_token_ids)
                    scores_list = action_to_text_scores.tolist()

                    log_lines = [
                        f"{tok!r}: {score:.4f}"
                        for tok, score in zip(text_token_strs, scores_list)
                    ]
                    print(f"[ACTION->TEXT DBG] task='{task_label}'", flush=True)
                    print(f"[ACTION->TEXT DBG] " + " | ".join(log_lines), flush=True)
                # ----------------- action→text 중요도 1차 검증 (로그 전용) ----------------------

                # ----------------- stopword 제외 text attention map 검증 ----------------------
                if getattr(cfg, "use_stopword_filtered_debug", False) and text_query_end is not None:
                    prompt_token_count = inputs["input_ids"].shape[1]

                    # BOS 제외한 text token들
                    text_token_count = prompt_token_count - 1
                    text_token_ids = inputs["input_ids"][0, 1:1 + text_token_count].tolist()
                    text_token_strs = processor.tokenizer.convert_ids_to_tokens(text_token_ids)

                    # task span 안의 content word group 추출
                    content_word_groups = get_content_word_row_groups(
                        text_token_strs=text_token_strs,
                        text_token_start=query_token_start,
                    )

                    fixed_stopword_scores = token_attention_merge_word_groups(
                        multihead_attention=attns,
                        word_groups=content_word_groups,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=fixed_token_start,
                        num_key_tokens=num_patches_per_image,
                    )

                    wrist_stopword_scores = token_attention_merge_word_groups(
                        multihead_attention=attns,
                        word_groups=content_word_groups,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=wrist_token_start,
                        num_key_tokens=num_patches_per_image,
                    )

                    fixed_stopword_map = spatial_scores_to_map(
                        fixed_stopword_scores,
                        device=DEVICE,
                    ).detach().cpu()

                    wrist_stopword_map = spatial_scores_to_map(
                        wrist_stopword_scores,
                        device=DEVICE,
                    ).detach().cpu()

                    last_caches["latest_fixed_stopword_filtered_map"] = fixed_stopword_map
                    last_caches["latest_wrist_stopword_filtered_map"] = wrist_stopword_map

                    content_words = [group["word"] for group in content_word_groups]
                    content_token_groups = [group["tokens"] for group in content_word_groups]

                    print(
                        f"[STOPWORD DBG] task='{task_label}' "
                        f"content_words={content_words} "
                        f"content_token_groups={content_token_groups}",
                        flush=True,
                    )
                # ----------------- stopword 제외 text attention map 검증 ----------------------

                # ----------------- query-mode attention map debug / critical source ----------------------
                # mixed / text-only / content-only / status-only / action-only map을 각각 저장
                # critical_attention_mode가 mixed가 아니면, visualization을 꺼도 score 계산은 해야 함
                if need_query_mode_scores and text_query_end is not None:
                    prompt_token_count = inputs["input_ids"].shape[1]
                    text_token_count = prompt_token_count - 1

                    status_token_start = query_token_start - 1 if cfg.use_proprio else None
                    status_token_end = query_token_start if cfg.use_proprio else None
                    action_token_start = text_query_end

                    # A. text-only map
                    fixed_text_only_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=fixed_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=query_token_start,
                        query_token_end=text_query_end,
                    )

                    wrist_text_only_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=wrist_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=query_token_start,
                        query_token_end=text_query_end,
                    )

                    last_caches["latest_fixed_text_only_map"] = spatial_scores_to_map(
                        fixed_text_only_scores, device=DEVICE,
                    ).detach().cpu()
                    last_caches["latest_wrist_text_only_map"] = spatial_scores_to_map(
                        wrist_text_only_scores, device=DEVICE,
                    ).detach().cpu()

                    # B. content-words-only map
                    text_token_ids = inputs["input_ids"][0, 1:1 + text_token_count].tolist()
                    text_token_strs = processor.tokenizer.convert_ids_to_tokens(text_token_ids)

                    content_word_groups = get_content_word_row_groups(
                        text_token_strs=text_token_strs,
                        text_token_start=query_token_start,
                    )

                    fixed_content_scores = token_attention_merge_word_groups(
                        multihead_attention=attns,
                        word_groups=content_word_groups,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=fixed_token_start,
                        num_key_tokens=num_patches_per_image,
                    )

                    wrist_content_scores = token_attention_merge_word_groups(
                        multihead_attention=attns,
                        word_groups=content_word_groups,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=wrist_token_start,
                        num_key_tokens=num_patches_per_image,
                    )

                    last_caches["latest_fixed_content_words_map"] = spatial_scores_to_map(
                        fixed_content_scores, device=DEVICE,
                    ).detach().cpu()
                    last_caches["latest_wrist_content_words_map"] = spatial_scores_to_map(
                        wrist_content_scores, device=DEVICE,
                    ).detach().cpu()

                    last_caches["latest_fixed_stopword_filtered_map"] = last_caches["latest_fixed_content_words_map"]
                    last_caches["latest_wrist_stopword_filtered_map"] = last_caches["latest_wrist_content_words_map"]

                    # C. status/proprio-only map
                    if cfg.use_proprio and status_token_start is not None:
                        fixed_status_scores = token_attention_merge(
                            multihead_attention=attns,
                            layer_ids=spatial_layer_ids,
                            kept_query_positions=attention_kept_query_positions,
                            key_token_start=fixed_token_start,
                            num_key_tokens=num_patches_per_image,
                            query_token_start=status_token_start,
                            query_token_end=status_token_end,
                        )

                        wrist_status_scores = token_attention_merge(
                            multihead_attention=attns,
                            layer_ids=spatial_layer_ids,
                            kept_query_positions=attention_kept_query_positions,
                            key_token_start=wrist_token_start,
                            num_key_tokens=num_patches_per_image,
                            query_token_start=status_token_start,
                            query_token_end=status_token_end,
                        )

                        last_caches["latest_fixed_status_only_map"] = spatial_scores_to_map(
                            fixed_status_scores, device=DEVICE,
                        ).detach().cpu()
                        last_caches["latest_wrist_status_only_map"] = spatial_scores_to_map(
                            wrist_status_scores, device=DEVICE,
                        ).detach().cpu()
                    else:
                        fixed_status_scores = None
                        wrist_status_scores = None

                        last_caches["latest_fixed_status_only_map"] = None
                        last_caches["latest_wrist_status_only_map"] = None

                    # D. action-only map
                    fixed_action_only_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=fixed_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=action_token_start,
                        query_token_end=None,
                    )

                    wrist_action_only_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=spatial_layer_ids,
                        kept_query_positions=attention_kept_query_positions,
                        key_token_start=wrist_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=action_token_start,
                        query_token_end=None,
                    )

                    last_caches["latest_fixed_action_only_map"] = spatial_scores_to_map(
                        fixed_action_only_scores, device=DEVICE,
                    ).detach().cpu()
                    last_caches["latest_wrist_action_only_map"] = spatial_scores_to_map(
                        wrist_action_only_scores, device=DEVICE,
                    ).detach().cpu()

                    if use_query_mode_map_debug:
                        content_words = [group["word"] for group in content_word_groups]
                        print(
                            f"[QUERY-MODE MAP DBG] task='{task_label}' "
                            f"layers={spatial_layer_ids} "
                            f"status=({status_token_start},{status_token_end}) "
                            f"text=({query_token_start},{text_query_end}) "
                            f"action=({action_token_start},end) "
                            f"content_words={content_words}",
                            flush=True,
                        )
                # ----------------- query-mode attention map debug / critical source ----------------------

                # ----------------- choose critical attention source -----------------
                critical_attention_mode = getattr(cfg, "critical_attention_mode", "mixed")

                # 기본값: 기존 mixed map
                selected_fixed_scores = fixed_spatial_scores
                selected_wrist_scores = wrist_spatial_scores

                if critical_attention_mode == "text_only":
                    selected_fixed_scores = fixed_text_only_scores
                    selected_wrist_scores = wrist_text_only_scores

                elif critical_attention_mode == "content_words":
                    selected_fixed_scores = fixed_content_scores
                    selected_wrist_scores = wrist_content_scores

                elif critical_attention_mode == "status_only":
                    if fixed_status_scores is not None and wrist_status_scores is not None:
                        selected_fixed_scores = fixed_status_scores
                        selected_wrist_scores = wrist_status_scores
                    else:
                        selected_fixed_scores = fixed_spatial_scores
                        selected_wrist_scores = wrist_spatial_scores

                elif critical_attention_mode == "action_only":
                    selected_fixed_scores = fixed_action_only_scores
                    selected_wrist_scores = wrist_action_only_scores

                elif critical_attention_mode == "mixed":
                    selected_fixed_scores = fixed_spatial_scores
                    selected_wrist_scores = wrist_spatial_scores

                else:
                    raise ValueError(f"Unknown critical_attention_mode: {critical_attention_mode}")

                # critical patch selection에서 실제로 쓰일 map을 이걸로 업데이트
                fixed_selected_map = spatial_scores_to_map(
                    selected_fixed_scores,
                    device=DEVICE,
                ).detach().cpu()

                wrist_selected_map = spatial_scores_to_map(
                    selected_wrist_scores,
                    device=DEVICE,
                ).detach().cpu()

                last_caches["latest_fixed_spatial_map"] = fixed_selected_map
                last_caches["latest_wrist_spatial_map"] = wrist_selected_map

                print(
                    f"[CRITICAL ATTN MODE] mode={critical_attention_mode} "
                    f"fixed_sum={float(torch.as_tensor(selected_fixed_scores).sum()):.4f} "
                    f"wrist_sum={float(torch.as_tensor(selected_wrist_scores).sum()):.4f}",
                    flush=True,
                )
                # ----------------- choose critical attention source -----------------

                if getattr(cfg, "use_attention_ema", False):
                    alpha = float(getattr(cfg, "attention_ema_alpha", 0.35))
                    alpha = max(0.0, min(1.0, alpha))

                    use_fixed_ema = getattr(cfg, "use_fixed_ema", True)
                    use_wrist_ema = getattr(cfg, "use_wrist_ema", True)

                    # fixed: EMA 적용 여부에 따라 분기
                    if use_fixed_ema:
                        last_caches["ema_fixed_spatial_map"] = update_attention_ema(
                            last_caches.get("prev_ema_fixed_spatial_map"),
                            last_caches["latest_fixed_spatial_map"],
                            alpha,
                        )
                    else:
                        last_caches["ema_fixed_spatial_map"] = last_caches["latest_fixed_spatial_map"]

                    # wrist: EMA 적용 여부에 따라 분기
                    if use_wrist_ema:
                        last_caches["ema_wrist_spatial_map"] = update_attention_ema(
                            last_caches.get("prev_ema_wrist_spatial_map"),
                            last_caches["latest_wrist_spatial_map"],
                            alpha,
                        )
                    else:
                        last_caches["ema_wrist_spatial_map"] = last_caches["latest_wrist_spatial_map"]

                    last_caches["ema_attention_step"] = int(last_caches.get("prev_ema_attention_step", 0)) + 1

                    last_caches.pop("prev_ema_fixed_spatial_map", None)
                    last_caches.pop("prev_ema_wrist_spatial_map", None)
                    last_caches.pop("prev_ema_attention_step", None)

                if last_layer_id is not None and last_layer_id < len(attns) and attns[last_layer_id] is not None:
                    final_fixed_spatial_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=[last_layer_id],
                        kept_query_positions=kept_query_positions,
                        key_token_start=fixed_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=query_token_start,
                    )

                    final_wrist_spatial_scores = token_attention_merge(
                        multihead_attention=attns,
                        layer_ids=[last_layer_id],
                        kept_query_positions=kept_query_positions,
                        key_token_start=wrist_token_start,
                        num_key_tokens=num_patches_per_image,
                        query_token_start=query_token_start,
                    )

                    last_caches["latest_fixed_final_layer_spatial_map"] = spatial_scores_to_map(
                        final_fixed_spatial_scores,
                        device=DEVICE,
                    ).detach().cpu()

                    last_caches["latest_wrist_final_layer_spatial_map"] = spatial_scores_to_map(
                        final_wrist_spatial_scores,
                        device=DEVICE,
                    ).detach().cpu()

                    last_caches["latest_final_attention_layer_id"] = last_layer_id

                    # entropy 로깅
                    # if attns[last_layer_id] is not None:
                    #     attn = attns[last_layer_id].to(torch.float32).mean(dim=1)[0]  # (q_len, k_len)

                    #     # action token rows만
                    #     action_attn = attn[query_token_start:, :]  # (n_action, k_len)

                    #     # fixed camera patches attention만
                    #     fixed_attn = action_attn[:, fixed_token_start:fixed_token_start + num_patches_per_image]
                    #     fixed_attn = fixed_attn / (fixed_attn.sum(dim=-1, keepdim=True) + 1e-10)
                    #     fixed_entropy = -torch.sum(fixed_attn * torch.log(fixed_attn + 1e-10), dim=-1).mean().item()
                    #     fixed_entropy_max = float(torch.log(torch.tensor(num_patches_per_image, dtype=torch.float32)))
                    #     fixed_entropy_ratio = fixed_entropy / (fixed_entropy_max + 1e-10)

                    #     # wrist camera patches attention만
                    #     wrist_attn = action_attn[:, wrist_token_start:wrist_token_start + num_patches_per_image]
                    #     wrist_attn = wrist_attn / (wrist_attn.sum(dim=-1, keepdim=True) + 1e-10)
                    #     wrist_entropy = -torch.sum(wrist_attn * torch.log(wrist_attn + 1e-10), dim=-1).mean().item()
                    #     wrist_entropy_max = float(torch.log(torch.tensor(num_patches_per_image, dtype=torch.float32)))
                    #     wrist_entropy_ratio = wrist_entropy / (wrist_entropy_max + 1e-10)

                    #     last_caches["latest_fixed_entropy"] = fixed_entropy_ratio
                    #     last_caches["latest_wrist_entropy"] = wrist_entropy_ratio

                # -------------------- [ENTROPY 구간] -----------------------------
                # 모든 레이어 entropy 계산
                # print(f"[DEBUG2] attns is not None: {attns is not None}")
                # print(f"[DEBUG2] len(attns): {len(attns) if attns is not None else 'N/A'}")
                # print(f"[DEBUG2] non-None attns count: {sum(1 for a in attns if a is not None) if attns is not None else 'N/A'}")
                # layer_entropies = {}
                # for layer_id, attn_map in enumerate(attns):
                #     if attn_map is None or not torch.is_tensor(attn_map):
                #         continue
                #     attn = attn_map.to(torch.float32).mean(dim=1)[0]
                #     action_attn = attn[query_token_start:, :]

                #     # fixed
                #     fixed_attn = action_attn[:, fixed_token_start:fixed_token_start + num_patches_per_image]
                #     fixed_attn = fixed_attn / (fixed_attn.sum(dim=-1, keepdim=True) + 1e-10)
                #     fixed_ent = -torch.sum(fixed_attn * torch.log(fixed_attn + 1e-10), dim=-1).mean().item()
                #     fixed_ent_ratio = fixed_ent / float(torch.log(torch.tensor(num_patches_per_image, dtype=torch.float32)))

                #     # wrist
                #     wrist_attn = action_attn[:, wrist_token_start:wrist_token_start + num_patches_per_image]
                #     wrist_attn = wrist_attn / (wrist_attn.sum(dim=-1, keepdim=True) + 1e-10)
                #     wrist_ent = -torch.sum(wrist_attn * torch.log(wrist_attn + 1e-10), dim=-1).mean().item()
                #     wrist_ent_ratio = wrist_ent / float(torch.log(torch.tensor(num_patches_per_image, dtype=torch.float32)))

                #     layer_entropies[layer_id] = {
                #         "fixed": round(fixed_ent_ratio, 4),
                #         "wrist": round(wrist_ent_ratio, 4),
                #     }
                # print(f"[DEBUG2] layer_entropies keys: {list(layer_entropies.keys())}")
                # last_caches["layer_entropies"] = layer_entropies
                # -------------------- [ENTROPY 구간] -----------------------------

                    # 모든 레이어 개별 heatmap 저장

                    # progressive_start_layer 안전하게 정의
                    # _progressive_start_layer = 0
                    # _pruning_layers = getattr(vla.language_model.config, "progressive_pruning_layers", None)
                    # if _pruning_layers is not None and len(_pruning_layers) > 0:
                    #     _progressive_start_layer = min(int(x) for x in _pruning_layers)

                    # for _viz_layer_id in range(len(attns)):
                    #     if attns[_viz_layer_id] is None:
                    #         continue
                    #     _fixed_scores = token_attention_merge(
                    #         multihead_attention=attns,
                    #         layer_ids=[_viz_layer_id],
                    #         kept_query_positions=kept_query_positions if _viz_layer_id >= _progressive_start_layer else None,
                    #         key_token_start=fixed_token_start,
                    #         num_key_tokens=num_patches_per_image,
                    #         query_token_start=query_token_start,
                    #     )
                    #     _wrist_scores = token_attention_merge(
                    #         multihead_attention=attns,
                    #         layer_ids=[_viz_layer_id],
                    #         kept_query_positions=kept_query_positions if _viz_layer_id >= _progressive_start_layer else None,
                    #         key_token_start=wrist_token_start,
                    #         num_key_tokens=num_patches_per_image,
                    #         query_token_start=query_token_start,
                    #     )
                    #     last_caches[f"layer_{_viz_layer_id:02d}_fixed_map"] = spatial_scores_to_map(_fixed_scores, device=DEVICE).detach().cpu()
                    #     last_caches[f"layer_{_viz_layer_id:02d}_wrist_map"] = spatial_scores_to_map(_wrist_scores, device=DEVICE).detach().cpu()

                    # 디버그
                    # fh = last_caches.get("final_hidden_states")
                    # kqp = kept_query_positions
                    # print(f"[DEBUG NORM] final_hidden shape: {fh.shape if fh is not None else None}")
                    # print(f"[DEBUG NORM] kept_query_positions: {kqp.shape if kqp is not None else None}")
                    # if kqp is not None:
                    #     print(f"[DEBUG NORM] kqp min/max: {kqp.min()} / {kqp.max()}")
                    #     fixed_mask = (kqp >= fixed_token_start) & (kqp < fixed_token_start + num_patches_per_image)
                    #     wrist_mask = (kqp >= wrist_token_start) & (kqp < wrist_token_start + num_patches_per_image)
                    #     print(f"[DEBUG NORM] fixed patch 남은 수: {fixed_mask.sum()}")
                    #     print(f"[DEBUG NORM] wrist patch 남은 수: {wrist_mask.sum()}")

                    # fixed_norm_map, wrist_norm_map = compute_hidden_norm_maps(
                    #     final_hidden_states=last_caches.get("final_hidden_states"),
                    #     fixed_token_start=fixed_token_start,
                    #     wrist_token_start=wrist_token_start,
                    #     num_patches_per_image=num_patches_per_image,
                    #     kept_query_positions=kept_query_positions,
                    #     device=DEVICE,
                    # )
                    # if fixed_norm_map is not None:
                    #     last_caches["latest_fixed_hidden_norm_map"] = fixed_norm_map
                    #     last_caches["latest_wrist_hidden_norm_map"] = wrist_norm_map

                    # hidden state similarity map 계산
                    # fixed_cos_map, wrist_cos_map, fixed_dot_map, wrist_dot_map = compute_hidden_sim_maps(
                    #     final_hidden_states=last_caches.get("final_hidden_states"),
                    #     fixed_token_start=fixed_token_start,
                    #     wrist_token_start=wrist_token_start,
                    #     num_patches_per_image=num_patches_per_image,
                    #     query_token_start=query_token_start,
                    #     device=DEVICE,
                    # )
                    # if fixed_cos_map is not None:
                    #     last_caches["latest_fixed_hidden_cos_map"] = fixed_cos_map
                    #     last_caches["latest_wrist_hidden_cos_map"] = wrist_cos_map
                    #     last_caches["latest_fixed_hidden_dot_map"] = fixed_dot_map
                    #     last_caches["latest_wrist_hidden_dot_map"] = wrist_dot_map

        else:
            if action_head is None:
                action, _ = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                )
            else:
                action, _ = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    use_film=use_film,
                )
    # Return action chunk as list of actions
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