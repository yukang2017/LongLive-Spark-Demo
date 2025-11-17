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
from gradio.themes.utils.colors import Color as ColorTemplate
from nvidia import Nvidia
from gradio.themes.utils import colors, fonts, sizes

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
def concat_history_videos(history_paths):
    """将同一会话中的历史视频按顺序拼接在一起，最多拼接 6 段。

    参数
    ----
    history_paths : List[str]
        当前会话内已经生成的所有成片路径（按时间顺序保存）。

    返回
    ----
    str 或 None
        拼接后的视频路径；若没有可用视频则返回 None。
    """
    import os
    import tempfile
    import numpy as np
    import imageio.v3 as iio

    if not history_paths:
        # 没有历史视频，前端 File 组件会显示为空
        return None

    # 只取最近的 6 段，超过 6 段视为“重新开始拼接”
    selected_paths = list(history_paths)[-6:]

    all_frames = []
    for p in selected_paths:
        if not os.path.exists(p):
            continue
        # 读取整段视频的所有帧，保持原有分辨率
        vid = iio.imread(p)   # 形状一般为 (T, H, W, C)
        if vid.ndim == 3:
            # 万一缺通道维，简单扩展
            vid = vid[..., None]
        all_frames.append(vid)

    if not all_frames:
        return None

    concat_frames = np.concatenate(all_frames, axis=0)

    os.makedirs(config.output_folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="longlive_concat_",
        suffix=".mp4",
        delete=False,
        dir=config.output_folder,
    ) as tmpf:
        out_path = tmpf.name

    # 与单段视频一致，使用 16fps
    iio.imwrite(out_path, concat_frames, fps=16)

    return out_path


@torch.no_grad()
def on_generate_stream(prompt, seed, history_paths):
    import time, tempfile, os
    import numpy as np
    import imageio.v3 as iio
    from einops import rearrange

    # 初始化或恢复历史列表（当前会话内的所有成片路径）
    if history_paths is None:
        history_paths = []
    # 简单保证是 list 类型
    history_paths = list(history_paths)

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
    with tempfile.NamedTemporaryFile(
        prefix="longlive_",
        suffix=".mp4",
        delete=False,
        dir=config.output_folder,
    ) as tmpf:
        out_path = tmpf.name

    # 获取逐帧生成器（逐帧产出 HWC uint8）
    frame_iter = pipeline.inference(
        noise=sampled_noise,
        text_prompt=prompt,
        switch_frame_indices=[],
        return_latents=False,
    )

    frames = []
    t0 = time.time()

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

    # 封装 mp4
    iio.imwrite(out_path, frames, fps=16)

    dt = time.time() - t0  # 这里目前只是保留时长信息，如有需要可用于日志

    # 更新历史：追加当前成片，并限制长度避免无限增长（这里保留最近 100 段）
    history_paths.append(out_path)
    if len(history_paths) > 100:
        history_paths = history_paths[-100:]

    # 最终输出成片路径 + 历史列表（前端视频组件会自动加载）
    return out_path, history_paths

nv_green = ColorTemplate(
    name="nv_green",
    c50="#f2f9f3",  # Light shade of nv green
    c100="#e5f3eb",  # Lighter shade of nv green
    c200="#c7e6d9",  # Mid-tone nv green
    c300="#76b900",  # nv green
    c400="#48873a",  # Darker shade of nv green
    c500="#30692c",  # Dark shade of nv green
    c600="#245121",  # Very dark shade of nv green
    c700="#1b3a17",  # Very dark shade of nv green
    c800="#12280e",  # Very dark shade of nv green
    c900="#0a1909",  # Very dark shade of nv green
    c950="#081407",  # Very dark shade of nv green
)

# 创建 Gradio 界面
with gradio.Blocks(
    theme=Nvidia(
        primary_hue=nv_green,
        secondary_hue=nv_green,
        neutral_hue=colors.gray,
    ),
    title="LongLive Playground",

    js="async () => {" + open("./app_script.js", "r", encoding="utf-8").read() + "}",
    css="./app_style.css",) as demo:
    # 顶部标题
    with gradio.Column(scale=1, min_width=400):
        #gradio.Image("images/logo.png", show_label=False, elem_id="logo")
        gradio.Markdown(
            """
            <h1 style="text-align: center; font-size: 35px; margin-top: 1px;">
                🎬 LongLive: Real-time Interactive Long Video Generation
            </h1>
            """
        )
    gradio.Markdown(
        """
        <div style="text-align: center;">
            <strong style="font-size: 25px;">LongLive-1.3B</strong><br><br>
            <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                <a href='https://github.com/NVlabs/LongLive'>
                    <img src='https://img.shields.io/badge/GitHub-LongLive-blue' alt='GitHub'>
                </a>
                <a href='https://huggingface.co/Efficient-Large-Model/LongLive-1.3B'>
                    <img src='https://img.shields.io/badge/HF%20Model-LongLive-bron' alt='GitHub'>
                </a>
                <a href='https://arxiv.org/abs/2509.22622'>
                    <img src='https://img.shields.io/badge/ArXiv-Paper-red' alt='ArXiv'>
                </a>
                <a href='https://www.youtube.com/watch?v=CO1QC7BNvig'>
                    <img src='https://img.shields.io/badge/YouTube-Intro-yellow' alt='YouTube'>
                </a>
            </div>
        </div>
        <strong> </strong>
        <p>
            We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LongLive supports up to 240-second videos on a single H100 GPU. LongLive further supports INT8-quantized inference with only marginal quality loss.
        </p>
        """
    )

    gradio.Markdown("# Prompt → Video 示例\n上方输入 prompt，点击 Generate，下方会显示生成的视频。")

    with gradio.Column():
        seed = gradio.Textbox(label="Seed (-1 for random)", value="42")
        p1 = gradio.Textbox(label="Prompt", lines=2)
        generate_btn = gradio.Button("Generate")
        download_btn = gradio.Button("下载历史拼接视频")

    # 成片：等最终写完 mp4 后展示
    video_output = gradio.Video(label="新生成的视频", autoplay=True, elem_id="final_video")
    concat_file = gradio.File(label="历史拼接视频（点击下载）")
    history_state = gradio.State([])

    # 生成单段视频，同时更新当前会话的历史列表
    generate_btn.click(
        fn=on_generate_stream,
        inputs=[p1, seed, history_state],
        outputs=[video_output, history_state],
    )

    # 下载：将同一会话中的历史视频按顺序拼接（最多 6 段）
    download_btn.click(
        fn=concat_history_videos,
        inputs=[history_state],
        outputs=[concat_file],
    )

if __name__ == '__main__':
    demo.queue(max_size=16, default_concurrency_limit=1)
    demo.launch(share=True)
