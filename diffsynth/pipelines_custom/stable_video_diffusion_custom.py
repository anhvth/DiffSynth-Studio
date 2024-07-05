from ..models import ModelManager, SVDImageEncoder, SVDUNet, SVDVAEEncoder, SVDVAEDecoder
from ..schedulers import ContinuousODEScheduler
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange, repeat
from diffsynth.pipelines.stable_video_diffusion import SVDVideoPipeline

class SVDVideoPipelineCustom(SVDVideoPipeline):
    @staticmethod
    def from_model_manager(model_manager: ModelManager, **kwargs):
        pipe = SVDVideoPipelineCustom(device=model_manager.device, torch_dtype=model_manager.torch_dtype)
        pipe.fetch_main_models(model_manager)
        return pipe

    def encode_video_with_vae(self, video, preprocess=True):
        if preprocess:
            video = torch.concat([self.preprocess_image(frame) for frame in video], dim=0)
            video = rearrange(video, "T C H W -> 1 C T H W")
        else:
            video = video.unsqueeze(0)
        video = video.to(device=self.device, dtype=self.torch_dtype)
        latents = self.vae_encoder.encode_video(video)
        latents = rearrange(latents[0], "C T H W -> T C H W")
        return latents
        
    @torch.no_grad()
    def __call__(
        self,
        input_image=None,
        input_video=None,
        mask_frames=[],
        mask_frame_ids=[],
        min_cfg_scale=1.0,
        max_cfg_scale=3.0,
        denoising_strength=1.0,
        num_frames=25,
        height=576,
        width=1024,
        fps=7,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        num_inference_steps=20,
        post_normalize=True,
        contrast_enhance_scale=1.2,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        control_frames = None,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)

        # Prepare latent tensors
        noise = torch.randn((num_frames, 4, height//8, width//8), device="cpu", dtype=self.torch_dtype).to(self.device)
        if denoising_strength == 1.0:
            latents = noise.clone()
        else:
            latents = self.encode_video_with_vae(input_video)
            latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])

        # Prepare mask frames
        if len(mask_frames) > 0:
            mask_latents = self.encode_video_with_vae(mask_frames)

        # Encode image
        image_emb_clip_posi = self.encode_image_with_clip(input_image)
        image_emb_clip_nega = torch.zeros_like(image_emb_clip_posi)
        # import ipdb; ipdb.set_trace()

        # Update latent with pose embedings
        if control_frames is not None:
            image_emb_vae_first = self.encode_image_with_vae(input_image, noise_aug_strength)
            control_latents = self.encode_video_with_vae(control_frames, preprocess=False)
            image_emb_vae_posi = torch.cat([image_emb_vae_first, control_latents[1:]])
        #---
        else:
            image_emb_vae_posi = repeat(self.encode_image_with_vae(input_image, noise_aug_strength), "B C H W -> (B T) C H W", T=num_frames)
        image_emb_vae_nega = torch.zeros_like(image_emb_vae_posi)

        # Prepare classifier-free guidance
        cfg_scales = torch.linspace(min_cfg_scale, max_cfg_scale, num_frames)
        cfg_scales = cfg_scales.reshape(num_frames, 1, 1, 1).to(device=self.device, dtype=self.torch_dtype)
        
        # Prepare positional id
        add_time_id = torch.tensor([[fps-1, motion_bucket_id, noise_aug_strength]], device=self.device)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):

            # Mask frames
            for frame_id, mask_frame_id in enumerate(mask_frame_ids):
                latents[mask_frame_id] = self.scheduler.add_noise(mask_latents[frame_id], noise[mask_frame_id], timestep)

            # Fetch model output
            noise_pred = self.calculate_noise_pred(
                latents, timestep, add_time_id, cfg_scales,
                image_emb_vae_posi, image_emb_clip_posi, image_emb_vae_nega, image_emb_clip_nega
            )

            # Forward Euler
            latents = self.scheduler.step(noise_pred, timestep, latents)
            
            # Update progress bar
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        # Decode image
        latents = self.post_process_latents(latents, post_normalize=post_normalize, contrast_enhance_scale=contrast_enhance_scale)
        video = self.vae_decoder.decode_video(latents, progress_bar=progress_bar_cmd)
        video = self.tensor2video(video)

        return video

