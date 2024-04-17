import torch
from diffusers.models import AutoencoderKLTemporalDecoder
from video_editing.models.unet_spatial_temporal_model import UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import load_image

from video_editing.pipeline.svd_pipeline import StableVideoDiffusionPipeline

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from video_editing.pipeline.svd_pipeline import StableVideoDiffusionPipeline

pretrained_model_path = "stabilityai/stable-video-diffusion-img2vid-xt"
# Load scheduler, tokenizer and models.
noise_scheduler = EulerDiscreteScheduler.from_pretrained(
    pretrained_model_path, subfolder="scheduler")
feature_extractor = CLIPImageProcessor.from_pretrained(
    pretrained_model_path, subfolder="feature_extractor", revision=None
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    pretrained_model_path, subfolder="image_encoder", revision=None, variant="fp16"
)
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    pretrained_model_path, subfolder="vae", revision=None, variant="fp16").to("cuda")
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "/scratch/nua3jz/outputs/checkpoint-126500/unet",
    low_cpu_mem_usage=False,
)

image_encoder.to("cuda")
vae.to("cuda")
unet.to("cuda")

pipeline = StableVideoDiffusionPipeline.from_pretrained(
                            pretrained_model_path,
                            unet=unet,
                            image_encoder=image_encoder,
                            vae=vae,
                            revision=None,
                            torch_dtype=torch.float16,
                        )
pipeline.to("cuda")

from main import load_video_frames, export_to_gif
from diffusers.utils import load_image
import numpy as np
import os
validation_data = os.listdir("/home/nua3jz/Code/Video_Editing_New/validation_data/video_frames")

for v_data in validation_data:
    video_frames = pipeline(
                    load_image("/home/nua3jz/Code/Video_Editing_New/validation_data/object_images/"+v_data+'.jpg').resize((1024, 576)),
                    torch.tensor(np.load('/home/nua3jz/Code/Video_Editing_New/validation_data/bboxes/'+v_data+'.npy')[0:1, :]),
                    load_video_frames('/home/nua3jz/Code/Video_Editing_New/validation_data/video_frames/'+v_data, 8),
                    height=576,
                    width=1024,
                    num_frames=8,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.02,
                    # generator=generator,
                ).frames[0]
    out_file=f"/home/nua3jz/Code/Video_Editing_New/outputs/{v_data}.mp4"
    for i in range(len(video_frames)):
        img = video_frames[i]
        video_frames[i] = np.array(img)
    export_to_gif(video_frames, out_file, 8)