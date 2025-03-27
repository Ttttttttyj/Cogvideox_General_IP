# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from datetime import timedelta
from einops import rearrange
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler,CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
from pre_transformer_ip import GeneralIP_CogVideoXTransformer3DModel
from models.pipeline_ip import GeneralIP_Pipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available,load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import re
from PIL import Image
import random
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import numpy as np
from functools import partial


if is_wandb_available():
    import wandb

os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--ref_image_column",
        type=str,
        default="ref_image",
        help="The column of the dataset containing ref_images. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to ref_image data.",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_ref_images",
        type=str,
        default=None,
        help="One or more ref_image(s) that is used during validation to verify that the model is learning. Multiple validation ref_images should be separated by the '--validation_ref_image_seperator' string.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--validation_ref_image_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation ref_images",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--enable_mask_loss", type=bool, default=False, help="Whether or not to mask loss")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", default=False,help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    return parser.parse_args()


class VideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        ref_image_column: str = "ref_image",
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.ref_image_column = ref_image_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        self.instance_prompts, self.instance_video_paths,self.instance_ref_image_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts) or self.num_instance_videos != len(self.instance_ref_image_paths):
            raise ValueError(
                f"Expected length of instance prompts, videos and ref_images to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=} and {len(self.instance_ref_image_paths)}. Please ensure that the number of caption prompts, videos and ref_images match in your dataset."
            )

        # self.instance_videos = self._preprocess_data()
        self.train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # 将 PIL 图像转换为张量，并将像素值从 [0, 255] 转换到 [0, 1]
            ]
        )

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            # "instance_video": self.instance_videos[index],
            "instance_video": self.process_video_data(self.instance_video_paths[index]),
            "instance_ref_image": self.process_ref_image_data(self.instance_ref_image_paths[index]),
        }

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompts_path = self.instance_data_root.joinpath(self.caption_column)
        videos_path = self.instance_data_root.joinpath(self.video_column)
        ref_images_path = self.instance_data_root.joinpath(self.ref_image_column)

        if not prompts_path.exists() or not prompts_path.is_dir():
            raise ValueError(
                "Expected `--caption_column` to be path to a dir in `--instance_data_root` containing text prompt files."
            )
        if not videos_path.exists() or not videos_path.is_dir():
            raise ValueError(
                "Expected `--video_column` to be path to a dir in `--instance_data_root` containing video data"
            )
        if not ref_images_path.exists() or not ref_images_path.is_dir():
            raise ValueError(
                "Expected `--ref_images_column` to be path to a dir in `--instance_data_root` containing ref_images data"
            )

        video_paths = []
        captions = []
        ref_image_paths = []
        for root, dirnames, filenames in os.walk(videos_path):# 加载数据集路径
            for filename in filenames:
                if filename.endswith(".mp4"):
                    video_path = os.path.join(root, filename)
                    video_paths.append(video_path) 

                caption_path = video_path.replace(".mp4", ".txt").replace("videos", "labels")
                if os.path.exists(caption_path):
                    caption = open(caption_path, "r").read().splitlines()[0]
                else:
                    caption = ""
                captions.append(caption)

                # ref_images_dir_path = root.replace("videos","ref_images")
                # idx = re.findall(r'\d+', video_path)
                # ref_image_dir_path = os.path.join(ref_images_dir_path,idx[0])
                # ref_image_path = os.path.join(ref_image_dir_path,idx[0]+"_masked.jpg")
                # if not os.path.exists(ref_image_path):
                #     ref_image_path = ""
                # ref_image_paths.append(ref_image_path)

                ref_images_dir_path = root.replace("videos","random_ref_images")
                idx = re.findall(r'\d+', video_path)
                ref_image_dir_path = os.path.join(ref_images_dir_path,idx[0])
                ref_masked_image_dir = os.path.join(ref_image_dir_path,"masked_images")
                ref_image_paths.append(ref_masked_image_dir)

        return captions,video_paths,ref_image_paths

    def process_video_data(self, path):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(uri=Path(path).as_posix(), width=self.width, height=self.height)
        video_num_frames = len(video_reader) 

        start_frame = min(self.skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - self.skip_frames_end)
        if end_frame <= start_frame:
            frames = video_reader.get_batch([start_frame])
        elif end_frame - start_frame <= self.max_num_frames:
            frames = video_reader.get_batch(list(range(start_frame, end_frame)))
        else:
            indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
            frames = video_reader.get_batch(indices)

        # Ensure that we don't go over the limit
        frames = frames[: self.max_num_frames]
        selected_num_frames = frames.shape[0]

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        remainder = (3 + (selected_num_frames % 4)) % 4
        if remainder != 0:
            frames = frames[:-remainder]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        frames = frames.float()
        frames = torch.stack([self.train_transforms(frame) for frame in frames], dim=0)
        frames = frames.permute(0, 3, 1, 2).contiguous() # [F, C, H, W]
        return frames
    
    def resize_for_rectangle_crop(self,arr, height, width, reshape_mode="random"):
        if arr.shape[3] / arr.shape[2] > width / height:
            arr = resize(
                arr,
                size=[height, int(arr.shape[3] * height / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * width / arr.shape[3]), width],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - height
        delta_w = w - width

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = transforms.functional.crop(arr, top=top, left=left, height=height, width=width)
        return arr
    
    def process_ref_image_data(self,path):
        ref_images = []
        ref_masked_image_dir = path
        masked_images_list = os.listdir(ref_masked_image_dir)
        masked_images_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        ref_image_choice = random.choice(masked_images_list[:5])
        ref_image_path = os.path.join(ref_masked_image_dir,ref_image_choice)
        ref_image = Image.open(ref_image_path).convert("RGB")
        # ref_image = Image.open(path).convert("RGB")
        ref_image = self.image_transform(ref_image).unsqueeze(0) #[1,3,h,w]
        ref_image = self.resize_for_rectangle_crop(ref_image, self.height, self.width, reshape_mode="center") # [3,h,w]
        ref_image = ref_image * 2.0 - 1.0
        ref_image = ref_image.unsqueeze(0) # [F,C,H,W] [1,C,H,W]
        ref_images.append(ref_image)

        mask_image_path = ref_image_path.replace("masked_images","mask_images").replace("maked","mask")
        mask_image = Image.open(mask_image_path)
        mask_array = np.array(mask_image)
        mask_idx = np.where(mask_array > 200)
        left = np.min(mask_idx[1])  # 最左边的列
        right = np.max(mask_idx[1])  # 最右边的列
        top = np.min(mask_idx[0])   # 最上边的行
        bottom = np.max(mask_idx[0])  # 最下边的行
        binary_mask = np.zeros_like(mask_array, dtype=np.uint8)
        binary_mask[top:bottom+1, left:right+1] = 1
        binary_mask = torch.tensor(binary_mask).unsqueeze(0).unsqueeze(0).float()
        binary_mask = F.interpolate(
            binary_mask,
            size=(self.height, self.width),
            mode='bilinear',
            align_corners=False
        )
        binary_mask = binary_mask.squeeze(0).squeeze(0)
        ref_images.append(binary_mask)

        return ref_images


def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"video_{i}.mp4"}}
            )

    model_description = f"""
# CogVideoX LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained using the [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

Was LoRA for the text encoder enabled? No.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import CogVideoXPipeline_incontext
import torch

pipe = CogVideoXPipeline_incontext.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name=["cogvideox-lora"])

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

video = pipe("{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) and [here](https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type
        
    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B and `CogVideoXDPMScheduler` for CogVideoX-5B.
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, **scheduler_args)

    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{epoch+1}_{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    free_memory()

    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # 创建日志目录
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 创建项目配置
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # 创建分布式数据并行参数
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # 创建加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    # transformer = CogVideoXTransformer3DModel.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="transformer",
    #     torch_dtype=load_dtype,
    #     revision=args.revision,
    #     variant=args.variant,
    # )
    transformer = GeneralIP_CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
        low_cpu_mem_usage=False,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    # scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler = CogVideoXDDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(True)

    # transformer.requires_grad_(False)

    # 将 pre_transformer 模块的参数的 requires_grad 属性设置为 True
    for param in transformer.pre_transformer.parameters():
        param.requires_grad = True

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(accelerator: Accelerator,model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    # model: CogVideoXTransformer3DModel
                    model: GeneralIP_CogVideoXTransformer3DModel
                    model = unwrap_model(accelerator, model)
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()


    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            with init_empty_weights():
                # transformer_ = CogVideoXTransformer3DModel.from_config(
                #     args.pretrained_model_name_or_path, subfolder="transformer"
                # )
                transformer_ = GeneralIP_CogVideoXTransformer3DModel.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                )
                init_under_meta = True

        # load_model = CogVideoXTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
        load_model = GeneralIP_CogVideoXTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))


    with open("train_params_sft+pre_transformer.txt","w") as file:
        for name, param in transformer.named_parameters():
            if param.requires_grad:
                file.write(f" Train Parameter name: {name}\n")

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset = VideoDataset(
        instance_data_root=args.instance_data_root,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column,
        video_column=args.video_column,
        ref_image_column=args.ref_image_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    # def encode_video(video):
    #     # latent_dist_list = []
    #     video = video.to(accelerator.device, dtype=vae.dtype)
    #     latent_dist = vae.encode(video).latent_dist            
    #     return latent_dist

    def encode_video(videos):
        # latent_dist_list = []
        videos = videos.to(accelerator.device, dtype=vae.dtype)
        latent_dist = vae.encode(videos).latent_dist  
        video_latents = latent_dist.sample() * vae.config.scaling_factor
        video_latents = video_latents.to(memory_format=torch.contiguous_format).float()
        video_latents = video_latents.permute(0, 2, 1, 3, 4)
        return video_latents

    def add_noise_to_first_frame(image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(accelerator.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image_noise = image_noise
        image = image + image_noise
        return image
    
    def process_image(image):
        image = image.permute(0, 2, 1, 3, 4)
        # sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(accelerator.device)
        # sigma = torch.exp(sigma).to(image.dtype)
        # image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        # image_noise = image_noise
        # image = image + image_noise
        image_latent_dist = encode_video(image)
        return image_latent_dist

    def resize_mask(mask, latent):
        latent_size = latent.size()
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        return resized_mask

    # train_dataset.instance_videos = [encode_video(video) for video in train_dataset.instance_videos]

    def collate_fn(examples,noised_image_input):
        prompts = [example["instance_prompt"] for example in examples]
        videos = [example["instance_video"].unsqueeze(0) for example in examples]
        videos = torch.cat(videos)  #[batch_size,frame,channel,height,weight]
        videos = videos.permute(0, 2, 1, 3, 4) #[batch_size,channel,frame,height,weight]
        video_latents = encode_video(videos)
        # videos = []
        # for example in examples:
        #     video = example["instance_video"]
            
        #     latents = encode_video(video) 
        #     latents = latents.sample() * vae.config.scaling_factor
        #     latents = rearrange(latents, "F C 1 H W -> 1 F C H W")
        #     videos.append(latents)
        # videos = torch.cat(videos)
        video_latents = video_latents.sample() * vae.config.scaling_factor
        video_latents = video_latents.to(memory_format=torch.contiguous_format).float()
        video_latents = video_latents.permute(0, 2, 1, 3, 4) #[batch_size,frame,channel,height,weight]

        ref_images = [example["instance_ref_image"].unsqueeze(0) for example in examples]
        ref_images = torch.cat(ref_images).to(accelerator.device) #[batch_size,1,channel,height,weight]
        if noised_image_input:
            ref_images = ref_images.permute(0, 2, 1, 3, 4)
            ref_images = add_noise_to_first_frame(ref_images)
            ref_images = encode_video(ref_images)
            ref_images = ref_images.sample() * vae.config.scaling_factor
            ref_images = ref_images.permute(0, 2, 1, 3, 4)
        ref_images = ref_images.to(memory_format=torch.contiguous_format).float()


        return {
            "videos": video_latents,
            "prompts": prompts,
            "ref_images": ref_images,
        }

    # custom_collate_fn = partial(collate_fn,noised_image_input = args.noised_image_input)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        # collate_fn=custom_collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            warmup_num_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-sft"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    output_loss_file = os.path.join(args.output_dir, "epoch_mean_loss.txt")
    with open(output_loss_file, "w") as loss_file:
        for epoch in range(first_epoch, args.num_train_epochs):
            transformer.train()
            epoch_loss_sum = 0.0
            for step, batch in enumerate(train_dataloader):
                models_to_accumulate = [transformer]

                with accelerator.accumulate(models_to_accumulate):
                    videos = batch["instance_video"]  # [B, F, C, H, W],[2,49,3,480,720]
                    prompts = batch["instance_prompt"]
                    ref_images = batch["instance_ref_image"][0] #[2,1,3,480,720]
                    masks = batch["instance_ref_image"][1]
                    # model_input = batch["videos"].to(dtype=weight_dtype)  # [B, F, C, H, W]
                    # prompts = batch["prompts"]
                    # ref_images = batch["ref_images"].to(dtype=weight_dtype)
                    videos = videos.permute(0, 2, 1, 3, 4)
                    model_input = encode_video(videos).to(dtype=weight_dtype) #[2,13,16,60,90]
                    ref_images = process_image(ref_images).to(dtype=weight_dtype) #[2,1,16,60,90]

                    masks = masks.unsqueeze(1).to(dtype=weight_dtype) #[b,1,h,w]
                    masks = masks.repeat(1, model_input.shape[1], 1, 1).unsqueeze(1) #[b,1,f,h,w]
                    video_latent = model_input.clone().permute(0, 2, 1, 3, 4) #[b,f,c,h,w] -> [b,c,f,h,w]
                    masks = resize_mask(masks, video_latent) #[b,1,f,60,90]
                    masks = masks.repeat(1, video_latent.shape[1], 1, 1, 1).permute(0, 2, 1, 3, 4).float()  # B C F H W -> B F C H W
                    masks = masks.reshape(video_latent.shape[0], -1)

                    # encode prompts
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )

                    # Sample noise that will be added to the latents
                    noise = torch.randn_like(model_input)
                    batch_size, num_frames, num_channels, height, width = model_input.shape

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Prepare rotary embeds
                    image_rotary_emb = (
                        prepare_rotary_positional_embeddings(
                            height=args.height,
                            width=args.width,
                            num_frames=num_frames,
                            vae_scale_factor_spatial=vae_scale_factor_spatial,
                            patch_size=model_config.patch_size,
                            attention_head_dim=model_config.attention_head_dim,
                            device=accelerator.device,
                        )
                        if model_config.use_rotary_positional_embeddings
                        else None
                    )

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                    # Predict the noise residual
                    model_output = transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        ref_image_hidden_states=ref_images,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                    model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                    alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                    weights = 1 / (1 - alphas_cumprod)
                    while len(weights.shape) < len(model_pred.shape):
                        weights = weights.unsqueeze(-1)

                    target = model_input

                    # loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                    # loss = loss.mean()

                    loss = (weights * (model_pred - target) ** 2).reshape(batch_size, -1)

                    if args.enable_mask_loss:
                        loss = (loss * masks).sum() / masks.sum()
                    else:
                        loss = torch.mean(loss, dim=1).mean()
                    accelerator.backward(loss)



                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    if accelerator.state.deepspeed_plugin is None:
                        optimizer.step()
                        optimizer.zero_grad()

                    lr_scheduler.step()

                epoch_loss_sum += loss.item()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
            
            epoch_loss_mean = epoch_loss_sum / num_update_steps_per_epoch
            loss_file.write(f"Epoch {epoch + 1}, Mean Loss: {epoch_loss_mean:.6f}\n")
            loss_file.flush()
            # 打印当前 epoch 的 loss 均值
            print(f"Epoch {epoch + 1}, Mean Loss: {epoch_loss_mean:.6f}")

            if accelerator.is_main_process:
                if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:
                    print("Validating...")
                    # Create pipeline
                    pipe = GeneralIP_Pipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=unwrap_model(accelerator, transformer),
                        text_encoder=unwrap_model(accelerator, text_encoder),
                        vae=unwrap_model(accelerator, vae),
                        scheduler=scheduler,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )

                    validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                    validation_ref_images = args.validation_ref_images.split(args.validation_ref_image_separator)
                    idx = 0
                    for validation_prompt in validation_prompts:
                        validation_image = load_image(image=validation_ref_images[idx])
                        pipeline_args = {
                            "image" : validation_image,
                            "prompt": validation_prompt,
                            "guidance_scale": args.guidance_scale,
                            "use_dynamic_cfg": args.use_dynamic_cfg,
                            "height": args.height,
                            "width": args.width,
                        }

                        validation_outputs = log_validation(
                            pipe=pipe,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                        )
                        idx += 1

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(accelerator, transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)

        transformer.save_pretrained(
            os.path.join(args.output_dir, "transformer"),
            safe_serialization=True,
            max_shard_size="5GB",
        )
        # Final test inference
        pipe = GeneralIP_Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()

        # Run inference
        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_ref_images = args.validation_ref_images.split(args.validation_ref_image_separator)
            idx = 0
            for validation_prompt in validation_prompts:
                validation_image = load_image(image=validation_ref_images[idx])
                pipeline_args = {
                    "image" : validation_image,
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                }

                video = log_validation(
                    pipe=pipe,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    is_final_validation=True,
                )
                idx += 1
                validation_outputs.extend(video)

        if args.push_to_hub:
            print(f"Saving model to {repo_id}...")
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
