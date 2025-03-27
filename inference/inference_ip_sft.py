"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import argparse
from typing import Literal
from torchvision import transforms
import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    AutoencoderKLCogVideoX
)
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
import sys
sys.path.insert(0, "/home/tuyijing/CogVideoX/finetune/models")
# print(sys.path)
from i2v_ip_transformer import GeneralIP_CogVideoXTransformer3DModel
from i2v_ip_pipeline import GeneralIP_Pipeline
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.utils.torch_utils import is_compiled_module
import os

def unwrap_model(model):
    model = model._orig_mod if is_compiled_module(model) else model
    return model
    
def generate_video(
    prompt: str,
    model_path: str,
    transformer_path: str,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    num_frames: int = 49,
    width: int = 720, 
    height: int = 480,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.
    image = None
    video = None
    transformer = GeneralIP_CogVideoXTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    
    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder").to(dtype=dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae").to(dtype=dtype)

    # scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler") #5b
    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        # pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe = GeneralIP_Pipeline.from_pretrained(
            model_path,
            transformer=unwrap_model(transformer),
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            scheduler=scheduler,
            torch_dtype=dtype,
        )
        image = load_image(image=image_or_video_path)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing") # 5b

    pipe.to("cuda")
    
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
    if generate_type == "i2v":
        video_generate = pipe(
            prompt=prompt,
            image=image,  # The path of the image to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=False,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=False,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
        # video_generate = pipe(
        #     height=height,
        #     width=width,
        #     prompt=prompt,
        #     num_videos_per_prompt=num_videos_per_prompt,
        #     num_inference_steps=num_inference_steps,
        #     num_frames=num_frames,
        #     use_dynamic_cfg=False,
        #     guidance_scale=guidance_scale,
        #     generator=torch.Generator().manual_seed(seed),
        # ).frames[0]
    else:
        video_generate = pipe(
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            # num_frames=49,
            use_dynamic_cfg=False,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    print(output_path)
    
    export_to_video(video_generate, output_path, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="../CogVideoX-2b", help="The path of the pre-trained model to be used"    # change if use 5b-i2v
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--output_mp4_file", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--transformer_path", type=str, help="The path of Transformer"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')" # change if use 5b-i2v
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"   # change if use 5b-i2v
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_mp4_file)
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        transformer_path = args.transformer_path,
        output_path=output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height
    )
