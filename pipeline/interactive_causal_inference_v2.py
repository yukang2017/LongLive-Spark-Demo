# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG
import numpy as np
from einops import rearrange
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("demo")

class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        self.global_prompts = []
        self.cached_output = None

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        log.info(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
            )
        
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompt: str,
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert batch_size == 1, "inference_stream 目前仅支持 batch_size == 1 用于前端播放"

        self.num_frame_per_block = 3
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # encode all prompts
        log.info(text_prompt)
        # TODO: build history cond_list
        cond_in_use = self.text_encoder(text_prompts=text_prompt)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        log.info(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        if len(self.global_prompts) % 6 == 0:
            self.global_prompts = [text_prompt]
            self.cached_output = None

            self._initialize_kv_cache(
                batch_size,
                dtype=noise.dtype,
                device=noise.device,
                kv_cache_size_override=kv_cache_size
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            switch_frame_indices = []
        else:
            self.global_prompts.append(text_prompt)
            switch_frame_indices = [0]
            output = torch.cat([self.cached_output, output], dim=1)

        start_frame = 42 * (len(self.global_prompts) - 1)
        current_start_frame = start_frame
        self.generator.model.local_attn_size = self.local_attn_size
        log.info(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)


        yield_times = 0
        finished_recache = False
        for current_num_frames in all_num_frames:
            moved_frames = current_start_frame - start_frame
            log.info("current_start_frame: %d", current_start_frame)
            if len(self.global_prompts) > 1 and not finished_recache:
                self._recache_after_switch(output, current_start_frame, cond_in_use)
                finished_recache = True

            noisy_input = noise[ :, moved_frames : moved_frames + current_num_frames]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                log.info("noisy_input.shape: %s", noisy_input.shape)
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # rerun with clean context to update cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # denoised_pred: (B, t, C, H, W)  这里 t = self.num_frame_per_block
            decoded_block = self.vae.decode_to_pixel(denoised_pred.to(noise.device), use_cache=False)
            decoded_block = (decoded_block * 0.5 + 0.5).clamp(0, 1)  # (B,t,C,H,W)
            block_thwc = rearrange(decoded_block[0],
                                   "t c h w -> t h w c").detach().cpu().numpy()  # (t,H,W,C), float32 in [0,1]

            # ✅ 逐帧 yield（而不是每个 block 只 yield 一次）
            log.info("block_thwc.shape: %s", block_thwc.shape)
            for f in range(block_thwc.shape[0]):
                frame_u8 = (block_thwc[f] * 255.0).round().astype("uint8")
                yield frame_u8
                print("yield_times", yield_times)
                yield_times += 1

            current_start_frame += current_num_frames

            '''
            with torch.autocast(device_type=noise.device.type, dtype=torch.bfloat16, enabled=True):
                decoded_block = self.vae.decode_to_pixel(
                    denoised_pred.to(noise.device),
                    use_cache=False
                )
            decoded_block = (decoded_block * 0.5 + 0.5).clamp(0, 1)  # B,t,C,H,W in [0,1]

            # 只取第 0 个样本（batch_size==1）
            block_thwc = rearrange(decoded_block[0], "t c h w -> t h w c").detach().cpu()
            block_u8 = (block_thwc * 255.0).round().to(torch.uint8).numpy()  # (t, H, W, C)

            # 逐帧送出：每次 yield 一个 (H,W,C) uint8
            for f in range(block_u8.shape[0]):
                yield block_u8[f]

            for f in range(block_thwc.shape[0]):
                frame_u8 = (block_thwc[f] * 255.0).round().astype("uint8")
                yield frame_u8
            # Update frame pointer
            current_start_frame += current_num_frames
            '''

        log.info("output.shape: %s", output.shape)

        self.cached_output = output

        '''
        # Standard decoding
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
        '''