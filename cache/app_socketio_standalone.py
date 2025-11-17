
"""
Standalone Flask‑SocketIO streaming demo for your video generator.
- Does NOT import/reuse your Gradio app.py
- Initializes config/device/pipeline by itself
- Streams JPEG frames over WebSocket to a tiny HTML client
Run:
  python app_socketio_standalone.py --config_path path/to/config.yaml --port 5000
Or production:
  gunicorn -k eventlet -w 1 app_socketio_standalone:app --bind 0.0.0.0:5000 --env CONFIG_PATH=path/to/config.yaml
"""
import os
import io
import base64
import time
import queue
import threading
import argparse
from typing import List, Optional

import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# --- Torch / project imports ---
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from einops import rearrange

# Your project utilities (must be importable in PYTHONPATH)
from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from pipeline.interactive_causal_inference_v2 import InteractiveCausalInferencePipeline
#import threading
#PIPE_LOCK = threading.Lock()

# ============== Config & initialization ==============

def build_pipeline(config_path: str):
    """
    Build everything needed for inference without touching the old app.py.
    Returns: (config, device, pipeline, local_rank)
    """
    # Load config
    config = OmegaConf.load(config_path)

    # Distributed/basic env (minimal safe init; you can expand to your full logic)
    if "LOCAL_RANK" in os.environ:
        os.environ["NCCL_CROSS_NIC"] = "1"
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout
            )
        set_seed(int(getattr(config, "seed", 42)) + local_rank)
        if local_rank == 0:
            print(f"[Rank {rank}] DDP initialized on {device}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(int(getattr(config, "seed", 42)))
        print(f"[Single] device={device}")

    # Memory hints
    low_memory = get_cuda_free_memory_gb(device) < 40 if device.type == "cuda" else False

    # Build pipeline
    pipe = InteractiveCausalInferencePipeline(config, device=device)

    # (Optional) load generator ckpt
    gen_ckpt = getattr(config, "generator_ckpt", None)
    use_ema = bool(getattr(config, "use_ema", False))
    if gen_ckpt and os.path.exists(gen_ckpt):
        state_dict = torch.load(gen_ckpt, map_location="cpu")
        raw_gen_state = state_dict["generator_ema" if use_ema else "generator"]
        if use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")
            cleaned = {_clean_key(k): v for k, v in raw_gen_state.items()}
            missing, unexpected = pipe.generator.load_state_dict(cleaned, strict=False)
            if local_rank == 0:
                if missing:
                    print(f"[ckpt] missing={len(missing)} (show first 8): {missing[:8]}")
                if unexpected:
                    print(f"[ckpt] unexpected={len(unexpected)} (show first 8): {unexpected[:8]}")
        else:
            pipe.generator.load_state_dict(raw_gen_state)
        if local_rank == 0:
            print(f"[ckpt] loaded from {gen_ckpt} (use_ema={use_ema})")

    # (Optional) LoRA
    pipe.is_lora_enabled = False
    try:
        from utils.lora_utils import configure_lora_for_model
        import peft
    except Exception as e:
        configure_lora_for_model = None
        peft = None
        if getattr(config, "adapter", None) and local_rank == 0:
            print(f"[Warning] LoRA requested but deps unavailable: {e}")

    if getattr(config, "adapter", None) and configure_lora_for_model is not None:
        if local_rank == 0:
            print(f"[LoRA] enabling: {config.adapter}")
        pipe.generator.model = configure_lora_for_model(
            pipe.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=(local_rank == 0),
        )
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path and os.path.exists(lora_ckpt_path):
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipe.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipe.generator.model, lora_checkpoint)
            if local_rank == 0:
                print("[LoRA] weights loaded")
        else:
            if local_rank == 0:
                print("[LoRA] no lora_ckpt specified; using initialized adapters")
        pipe.is_lora_enabled = True

    # dtype / device moves
    pipe = pipe.to(dtype=torch.bfloat16)
    if low_memory and hasattr(DynamicSwapInstaller, "install_model"):
        DynamicSwapInstaller.install_model(pipe.text_encoder, device=device)
    pipe.generator.to(device=device)
    pipe.vae.to(device=device)

    return config, device, pipe, local_rank

# ============== Flask‑SocketIO app ==============

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "devkey")
#socketio = SocketIO(app, cors_allowed_origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global singletons set after init()
CFG = None
DEVICE = None
PIPE = None
LOCAL_RANK = 0

# A small queue and a steady sender thread to avoid "PPT effect"
FRAME_Q: "queue.Queue[tuple[str,int]]" = queue.Queue(maxsize=128)

def to_jpeg_base64(frame_u8: np.ndarray, resize_ratio: float = 1.0, quality: int = 80) -> str:
    img = Image.fromarray(frame_u8, mode="RGB")
    if resize_ratio != 1.0:
        w, h = img.size
        img = img.resize((int(w * resize_ratio), int(h * resize_ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")

'''
def sender_worker(target_fps: int = 12):
    period = 1.0 / max(1, target_fps)
    last = 0.0
    while True:
        item = FRAME_Q.get()
        if item is None:
            # Stream ended
            #socketio.emit("eos", {})
            FRAME_Q.task_done()
            continue
        b64, idx = item
        # throttle to steady FPS
        now = time.time()
        wait = period - (now - last)
        if wait > 0:
            time.sleep(wait)
        socketio.emit("frame", {"data": b64, "index": idx})
        last = time.time()
        FRAME_Q.task_done()

def sender_worker(target_fps: int = 12):
    period = 1.0 / max(1, target_fps)
    last = 0.0
    while True:
        item = FRAME_Q.get()
        try:
            if item is None:
                # 可选：这是一条“全局”EOS；如果你统一用了 (sid,None,-1) 就不需要这段
                FRAME_Q.task_done()
                continue

            sid, b64, idx = item
            if b64 is None:
                socketio.emit("eos", {}, namespace="/", to=sid)
                FRAME_Q.task_done()
                continue

            # 控帧率
            now = time.time()
            wait = period - (now - last)
            if wait > 0:
                time.sleep(wait)

            socketio.emit("frame", {"data": b64, "index": idx}, namespace="/", to=sid)
            last = time.time()
        finally:
            FRAME_Q.task_done()
'''

def sender_worker(target_fps: int = 12):
    period = 1.0 / max(1, target_fps)
    last = 0.0
    while True:
        item = FRAME_Q.get()
        try:
            # 队列协议：item 必须是三元组 (sid, b64, idx)
            if item is None:
                # 如果你还有别处 put(None)，这里直接丢弃并继续
                continue

            sid, b64, idx = item

            if b64 is None:
                # 结束该 sid 的流
                socketio.emit("eos", {}, namespace="/", to=sid)
                continue

            # 控帧率
            now = time.time()
            wait = period - (now - last)
            if wait > 0:
                time.sleep(wait)

            socketio.emit("frame", {"data": b64, "index": idx}, namespace="/", to=sid)
            last = time.time()

        finally:
            # ⚠️ 每个 get() 恰好对应一次 task_done()
            FRAME_Q.task_done()

_sender = threading.Thread(target=sender_worker, kwargs=dict(target_fps=int(os.environ.get("STREAM_FPS", "12"))), daemon=True)
_sender.start()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("start")
def on_start(data):
    sid = request.sid
    prompt = (data.get("prompt") or "").strip()
    seed = int(data.get("seed") or 42)
    socketio.start_background_task(target=generate_stream_job, sid=sid, prompt=prompt, seed=seed)


'''
def generate_stream_job(prompt: str, seed: int):
    global CFG, DEVICE, PIPE, LOCAL_RANK
    torch.set_grad_enabled(False)
    torch.manual_seed(int(seed))

    # Sample noise (shape must match your model)
    sampled_noise = torch.randn(
        [CFG.num_samples, 42, 16, 60, 104],
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    # "pseudo streaming": generate full video, then push frames
    # If you later implement PIPE.inference_stream(...), replace below with true streaming.
    video_btchw = PIPE.inference(
        noise=sampled_noise,
        text_prompt=prompt,
        switch_frame_indices=[],
        return_latents=False,
    )  # (B,T,C,H,W) in [0,1]

    video_thwc = (rearrange(video_btchw[0], "t c h w -> t h w c") * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    for i, frame in enumerate(video_thwc):
        try:
            b64 = to_jpeg_base64(frame, resize_ratio=float(os.environ.get("STREAM_RESIZE", "1.0")), quality=int(os.environ.get("STREAM_JPEG_QUALITY", "80")))
            FRAME_Q.put((b64, int(i)), timeout=5)
        except queue.Full:
            # drop when saturated
            pass

    # EOS signals
    FRAME_Q.put((None, -1))   # mark for client
    FRAME_Q.put(None)         # notify sender to emit eos
'''

def generate_stream_job(sid, prompt: str, seed: int):
    if True:
        import queue, os, torch
        torch.set_grad_enabled(False)
        torch.manual_seed(int(seed))

        # 采样噪声（按你的模型要求，保持和原先一致）
        sampled_noise = torch.randn(
            [CFG.num_samples, 42, 16, 60, 104],
            device=DEVICE,
            dtype=torch.bfloat16,
        )

        # ✅ 现在 inference 是生成器：逐帧产出 HWC uint8
        frame_iter = PIPE.inference(
            noise=sampled_noise,
            text_prompt=prompt,
            switch_frame_indices=[],   # 你内部已根据 self.global_prompts 处理
            return_latents=False,
        )

        idx = 0
        for frame_u8 in frame_iter:
            # frame_u8: (H, W, C) uint8 —— 你在 inference 里已这样 yield
            try:
                b64 = to_jpeg_base64(
                    frame_u8,
                    resize_ratio=float(os.environ.get("STREAM_RESIZE", "1.0")),
                    quality=int(os.environ.get("STREAM_JPEG_QUALITY", "80")),
                )
                FRAME_Q.put((sid, b64, idx), timeout=5)
                idx += 1
            except queue.Full:
                # 队列满了就丢帧，保持实时性
                pass
            except Exception:
                # 任何异常都结束流，避免 sender 卡住
                FRAME_Q.put((sid, None, -1))
                #FRAME_Q.put(None)
                raise

        # 收尾：通知前端结束
        FRAME_Q.put((sid, None, -1))   # 给前端的“逻辑 EOS”
        #FRAME_Q.put(None)         # 让 sender_worker 发 'eos'

def init_server(config_path: str):
    global CFG, DEVICE, PIPE, LOCAL_RANK
    CFG, DEVICE, PIPE, LOCAL_RANK = build_pipeline(config_path)
    # Ensure output folder exists if needed by your pipeline
    out_dir = getattr(CFG, "output_folder", "./outputs")
    os.makedirs(out_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", ""),
                        help="Path to your OmegaConf YAML config")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if not args.config_path or not os.path.exists(args.config_path):
        raise FileNotFoundError(f"--config_path not found: {args.config_path}")

    init_server(args.config_path)
    print(f"[Server] Ready on http://{args.host}:{args.port}")
    print(f"[Server] Use Cloudflare Tunnel: cloudflared tunnel --url http://127.0.0.1:{args.port}")
    # eventlet recommended for production websocket performance
    socketio.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
