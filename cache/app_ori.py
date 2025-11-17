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
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline.causal_multiple_switch_inference import (
    CausalMultipleSwitchInferencePipeline,
)
from utils.dataset import MultiTextDataset
import gradio



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

pipeline = CausalMultipleSwitchInferencePipeline(config, device=device)

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

prompts = [
    "Golden-hour beach scene, a graceful female dancer in a flowing white dress stands barefoot on wet sand, sunset sky glowing orange and pink behind her. She begins with a calm standing pose, arms gently outstretched, hair softly swaying in the breeze. Cinematic lighting, realistic textures, warm romantic atmosphere.",
    "The same dancer, same beach and sunset lighting. She slowly raises one arm upward and starts a gentle side step, her dress catching the breeze. Subtle ripples in the shallow water reflect the sky colors. Consistent character design, cinematic realism.",
    "The dancer continues, spinning lightly on one foot, her dress flowing in a circular motion. Sunset colors deepen, with orange hues blending into purple near the horizon. Water reflections shimmer. Maintain the same character, environment, and cinematic style.",
    "The dancer completes her spin and transitions into a low lunge, one hand grazing the sand, hair flowing naturally. The sea breeze slightly lifts the hem of her dress. The warm, fading sunlight casts long shadows across the beach. Consistent dancer and beach scene.",
    "She rises gracefully, stepping forward toward the viewer, performing a gentle leap with both arms extended. The horizon now glows with deep purples and pinks. Small splashes form around her feet as she lands lightly on the wet sand. Maintain the same dancer, dress, and lighting style.",
    "The dancer finishes with a serene standing pose facing the horizon, both arms open wide as if embracing the sunset. The last light of the day paints the sky with soft lavender and amber. Gentle waves roll in the background, preserving the cinematic, realistic atmosphere and consistent character."
]

@torch.no_grad()
def synthesize_video_from_prompt(prompts_list):
    sampled_noise = torch.randn(
        [
            config.num_samples,
            config.num_output_frames,
            16,
            60,
            104,
        ],
        device=device,
        dtype=torch.bfloat16,
    )

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts_list=prompts_list,
        switch_frame_indices=switch_frame_indices,
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
def on_generate(p1, p2, p3, p4, p5, p6, seed):
    torch.manual_seed(int(seed))
    with open('/tmp/.unhold', 'w') as f:
        f.write('')
    prompts = [p1, p2, p3, p4, p5, p6]
    prompts = [p.strip() for p in prompts if p.strip()]
    # if not prompt or prompt.strip() == "":
    #     return None, "请在上方输入 prompt 后再生成视频。"

    # You can add sanitization / prompt conditioning here
    video_path = synthesize_video_from_prompt(prompts)
    os.remove('/tmp/.unhold')
    return video_path, "生成完成：" + os.path.basename(video_path)


with gradio.Blocks() as demo:
    gradio.Markdown("# Prompt → Video 示例\n上方输入 prompt，点击 Generate，下方会显示生成的视频。")

    with gradio.Column():
        seed = gradio.Textbox(label="Seed (-1 for random)", value="42")
        p1 = gradio.Textbox(label="Prompt 1", lines=2)
        p2 = gradio.Textbox(label="Prompt 2", lines=2)
        p3 = gradio.Textbox(label="Prompt 3", lines=2)
        p4 = gradio.Textbox(label="Prompt 4", lines=2)
        p5 = gradio.Textbox(label="Prompt 5", lines=2)
        p6 = gradio.Textbox(label="Prompt 6", lines=2)
        generate_btn = gradio.Button("Generate")

    # 视频展示区域
    video_output = gradio.Video(label="输出视频")
    status = gradio.Textbox(label="状态", interactive=False)

    prompts = [p1, p2, p3, p4, p5, p6]
    # prompts = [p.strip() for p in prompts if p.strip()]
    generate_btn.click(fn=on_generate, inputs=[p1,p2,p3,p4,p5,p6,seed], outputs=[video_output, status])
# ----------------------------- 推理循环 -----------------------------

if __name__ == '__main__':
    demo.launch(share=True)
