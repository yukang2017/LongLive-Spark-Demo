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
def on_generate(prompt, seed):
    torch.manual_seed(int(seed))
    with open('/tmp/.unhold', 'w') as f:
        f.write('')

    prompt = prompt.strip()
    # if not prompt or prompt.strip() == "":
    #     return None, "请在上方输入 prompt 后再生成视频。"

    # You can add sanitization / prompt conditioning here
    video_path = synthesize_video_from_prompt(prompt)
    os.remove('/tmp/.unhold')
    return video_path, "生成完成：" + os.path.basename(video_path)

'''
@torch.no_grad()
def on_generate_stream(prompt, seed):
    torch.manual_seed(int(seed))
    with open('/tmp/.unhold', 'w') as f:
        f.write('')
    prompt = prompt.strip()

    # 采样初噪
    sampled_noise = torch.randn([config.num_samples, 42, 16, 60, 104], device=device, dtype=torch.bfloat16)

    # 输出视频路径（最终一次性写成片）
    if dist.is_initialized():
        rr = dist.get_rank()
    else:
        rr = 0
    model_type = "lora" if getattr(pipeline, "is_lora_enabled", False) else ("ema" if getattr(config, "use_ema", False) else "regular")
    out_path = os.path.join(config.output_folder, f"rank{rr}-stream-{model_type}.mp4")

    # 预览：逐帧 yield 给 gr.Image
    # 尝试使用真正的流式推理；否则 fallback 到一次性推理再逐帧播出
    if True:
        frame_iter = pipeline.inference(
            noise=sampled_noise,
            text_prompt=prompt,
            switch_frame_indices=[]
        )
        realtime = True
    else:
        # 回退：整体推理一次，再把结果逐帧吐出去（“伪流”）
        video_btchw = pipeline.inference(
            noise=sampled_noise,
            text_prompt=prompt,
            switch_frame_indices=[],
            return_latents=False,
        )  # (B,T,C,H,W) float in [0,1]
        video_hwcu8 = (rearrange(video_btchw, "b t c h w -> t h w c") * 255).clamp(0,255).to(torch.uint8).cpu().numpy()
        frame_iter = (frame for frame in video_hwcu8)  # 逐帧迭代
        realtime = False

    # 逐帧展示 + 同步收集帧，最后一次性写 mp4（mp4 不能边写边播，需 finalize）
    frames = []
    fps = 16
    t0 = time.time()
    for i, frame in enumerate(frame_iter):
        frames.append(frame)
        # 每帧都 yield：live_image 刷新；video_output 暂时 None；状态显示进度
        yield frame, None, f"生成中：第 {i+1} 帧"

    # 结束后写成片（一次性）
    iio.imwrite(out_path, frames, fps=fps)  # mp4 写入，浏览器可播放
    dt = time.time() - t0
    os.remove('/tmp/.unhold')

    # 最后一次 yield：把最终 mp4 给 gr.Video，status 完成
    # 同时保留最后一帧在预览窗口
    last_frame = frames[-1] if frames else None
    yield last_frame, out_path, f"生成完成（{len(frames)} 帧，{dt:.1f}s，{'实时流' if realtime else '伪流'}）"
'''

@torch.no_grad()
def on_generate_stream(prompt, seed):
    import time
    from PIL import Image
    import numpy as np

    torch.manual_seed(int(seed))
    prompt = (prompt or "").strip()

    # 采样初噪
    sampled_noise = torch.randn([config.num_samples, 42, 16, 60, 104],
                                device=device, dtype=torch.bfloat16)

    # 预览帧参数（可按需求调）
    TARGET_FPS = 8             # 先用 8~12 都可
    PERIOD = 1.0 / TARGET_FPS
    PREVIEW_W = 512             # 预览宽度（高按比例缩放）；想更快可设 512/384
    JPEG_QUALITY = 75           # 70~85 区间比较平衡

    def downsample_to_jpeg_numpy(frame_u8: np.ndarray) -> np.ndarray:
        """
        输入：frame_u8 (H, W, 3) uint8
        输出：仍是 (H,W,3) uint8 的小图（让 gr.Image 自己转发 jpeg；format 已设为 jpeg）
        """
        h, w = frame_u8.shape[:2]
        if w > PREVIEW_W:
            new_h = int(h * PREVIEW_W / w)
            # 用 Pillow 做高效缩放
            img = Image.fromarray(frame_u8, mode="RGB").resize((PREVIEW_W, new_h), Image.BILINEAR)
            return np.asarray(img, dtype=np.uint8)
        return frame_u8

    # 真·流：你的 inference 已逐帧 yield (H,W,C) uint8
    frame_iter = pipeline.inference(
        noise=sampled_noise,
        text_prompt=prompt,
        switch_frame_indices=[],   # 你的内部已处理切换
    )

    frames_for_video = []   # 用于最终写 mp4
    last_t = 0.0
    shown = 0

    t0 = time.time()
    for i, frame in enumerate(frame_iter):
        frames_for_video.append(frame)

        # 限帧：达到周期才刷新 UI（否则跳过这次 UI 更新）
        now = time.time()
        if now - last_t < PERIOD:
            continue
        last_t = now

        # 预览降采样（大图会拖慢前端）
        preview = downsample_to_jpeg_numpy(frame)

        shown += 1
        # ✅ 只在需要时刷新，且用“批量更新”的 dict 形式更稳
        yield {
            live_image: preview,                 # gr.Image(format=jpeg) 接 numpy 就行
            status: f"生成中：第 {i+1} 帧（预览已显示 {shown} 帧 @~{TARGET_FPS}fps）",
            video_output: None                   # 流期间不更新视频成片
        }

    # 结束后把完整 mp4 一次性写出（保持你原逻辑）
    fps = 16
    out_path = os.path.join(config.output_folder, "stream-final.mp4")
    iio.imwrite(out_path, frames_for_video, fps=fps)
    dt = time.time() - t0

    # 收尾：展示成片，同时保留最后一帧预览
    if frames_for_video:
        last_frame = frames_for_video[-1]
        preview_last = downsample_to_jpeg_numpy(last_frame)
    else:
        preview_last = None

    yield {
        live_image: preview_last,
        video_output: out_path,
        status: f"生成完成（{len(frames_for_video)} 帧，{dt:.1f}s，预览约 {TARGET_FPS} fps）"
    }

with gradio.Blocks(title="Prompt → Video (Streaming Preview)") as demo:
    gradio.Markdown("# Prompt → Video 示例\n上方输入 prompt，点击 Generate，下方会显示生成的视频。")

    with gradio.Column():
        seed = gradio.Textbox(label="Seed (-1 for random)", value="42")
        p1 = gradio.Textbox(label="Prompt", lines=2)
        generate_btn = gradio.Button("Generate")

    # 新增：实时预览（逐帧刷）
    #live_image = gradio.Image(label="实时预览（逐帧）", interactive=False)
    live_image = gradio.Image(
        label="实时预览（逐帧）",
        interactive=False,
        format="jpeg",  # 关键：用 JPEG
        height=360,  # 你也可以用 width=640,height="auto"
    )

    # 成片：等最终写完 mp4 后展示
    video_output = gradio.Video(label="完整视频")
    status = gradio.Textbox(label="状态", interactive=False)

    generate_btn.click(fn=on_generate_stream, inputs=[p1, seed],
                       outputs=[live_image, video_output, status])

if __name__ == '__main__':
    demo.queue(max_size=16, default_concurrency_limit=1)
    demo.launch(share=True)
