import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from torchvision import transforms  # noqa: F401  # 保留与原脚本一致的导入
from einops import rearrange

from utils.misc import set_seed
from utils.distributed import barrier  # 使用项目中的barrier函数
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline.interactive_causal_inference_v2 import InteractiveCausalInferencePipeline
from utils.dataset import MultiTextDataset
import gradio

import time
import tempfile
import imageio.v3 as iio  # pip install imageio
import cv2
import base64
import numpy as np

# ----------------------------- Argument 解析 -----------------------------
parser = argparse.ArgumentParser("Prompt-multiple-switch inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ----------------------------- Distributed 设置 -----------------------------
if "LOCAL_RANK" in os.environ:
    # 设置NCCL环境变量以避免hang
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    
    # 先设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 初始化process group时指定backend和timeout
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout
        )
    
    set_seed(config.seed + local_rank)
    print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    print(f"Single GPU mode on device {device}")

low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

pipeline = InteractiveCausalInferencePipeline(config, device=device)

if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]

    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if local_rank == 0:
            if missing:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if unexpected:
                print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
# 应用与加载 LoRA（仅在提供 config.adapter 时启用）。
try:
    from utils.lora_utils import configure_lora_for_model
    import peft
except Exception as e:
    configure_lora_for_model = None
    peft = None
    if getattr(config, "adapter", None):
        if local_rank == 0:
            print(f"[Warning] LoRA requested but dependencies unavailable: {e}")

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # 在加载基础权重后，对 generator 的 transformer 模型应用 LoRA 包装
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # 加载 LoRA 权重（如果提供了 lora_ckpt）
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # 兼容包含 `generator_lora` 键或直接是 LoRA state dict 两种格式
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ----------------------------- 构建数据集 -----------------------------
# 解析 switch_frame_indices
switch_frame_indices: List[int] = [int(x) for x in config.switch_frame_indices.split(",") if x.strip()]


# 创建输出目录
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

global_prompts = []

@torch.no_grad()
def synthesize_video_from_prompt(prompt):
    sampled_noise = torch.randn(
        [
            config.num_samples,
            42,
            16,
            60,
            104,
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    global_prompts.append(prompt)
    switch_frame_indices = [int(i) for i in torch.arange(1, len(global_prompts)) * 40]

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompt=prompt,
        switch_frame_indices=[switch_frame_indices[-1]] if len(global_prompts) > 1 else [],
        return_latents=False,
    )

    current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Determine model type for filename
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "lora"
    elif getattr(config, 'use_ema', False):
        model_type = "ema"
    else:
        model_type = "regular"

    for seed_idx in range(config.num_samples):
        if config.save_with_index:
            output_path = os.path.join(config.output_folder, f"rank{rank}-{seed_idx}_{model_type}.mp4")
        else:
            # 取第一段 prompt 作为文件名前缀，避免过长
            short_name = prompts_list[0][0][:100].replace("/", "_")
            output_path = os.path.join(config.output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")
        write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)

    return output_path
    # if config.inference_iter != -1 and i >= config.inference_iter:
    #     break

@torch.no_grad()
def on_generate_stream(prompt, seed):
    import time, tempfile, os
    import numpy as np
    import imageio.v3 as iio
    from einops import rearrange

    torch.manual_seed(int(seed))
    prompt = (prompt or "").strip()

    # 采样初噪（保持你的形状）
    sampled_noise = torch.randn(
        [config.num_samples, 42, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16,
    )

    # 成片输出（临时文件，避免并发覆盖）
    os.makedirs(config.output_folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="longlive_", suffix=".mp4",
                                     delete=False, dir=config.output_folder) as tmpf:
        out_path = tmpf.name

    # 起始状态
    yield None, "渲染中：准备推理..."

    # 获取逐帧生成器（你已逐帧 yield HWC uint8）
    frame_iter = pipeline.inference(
        noise=sampled_noise,
        text_prompt=prompt,
        switch_frame_indices=[],
        return_latents=False,
    )

    frames = []
    t0 = time.time()
    last_text_update = t0
    frames_count = 0

    for frame in frame_iter:
        # 若是 torch.Tensor，转成 uint8 HWC numpy
        if isinstance(frame, torch.Tensor):
            t = frame.detach().to("cpu")
            if t.dtype.is_floating_point:
                t = (t * 255.0).clamp(0, 255).to(torch.uint8)
            if t.ndim == 3 and t.shape[0] in (1, 3):  # (C,H,W)->(H,W,C)
                t = rearrange(t, "c h w -> h w c")
            frame = t.numpy()
        elif isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        frames.append(frame)
        frames_count += 1

        # 每 ~0.5s 刷一次文字，减少前端负担
        now = time.time()
        if now - last_text_update > 0.5:
            yield None, f"渲染中：已完成 {frames_count} 帧..."
            last_text_update = now

    # 封装 mp4（阻塞一小会儿）
    yield None, f"封装视频中（共 {frames_count} 帧）..."
    iio.imwrite(out_path, frames, fps=16)

    dt = time.time() - t0
    # 最终只输出成片路径 + 状态（视频组件会自动加载）
    yield out_path, f"生成完成（{frames_count} 帧，用时 {dt:.1f}s）。"

with gradio.Blocks(title="Prompt → Video (Streaming Preview)") as demo:
    gradio.Markdown("# Prompt → Video 示例\n上方输入 prompt，点击 Generate，下方会显示生成的视频。")

    with gradio.Column():
        seed = gradio.Textbox(label="Seed (-1 for random)", value="42")
        p1 = gradio.Textbox(label="Prompt", lines=2)
        generate_btn = gradio.Button("Generate")

    # 成片：等最终写完 mp4 后展示
    video_output = gradio.Video(label="完整视频", autoplay=True, elem_id="final_video")
    status = gradio.Textbox(label="状态", interactive=False)

    generate_btn.click(fn=on_generate_stream, inputs=[p1, seed],
                       outputs=[video_output, status])

if __name__ == '__main__':
    demo.queue(max_size=16, default_concurrency_limit=1)
    demo.launch(share=True)
