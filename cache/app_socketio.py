
import os
import io
import base64
import threading
import queue
import time

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np
import torch
from einops import rearrange

# --- Import your existing initialization (config, device, pipeline, etc.) ---
# We import /mnt/data/app.py as a module to reuse its already-set pipeline/config without launching Gradio.
import importlib.util, types, sys, pathlib

APP_PY_PATH = pathlib.Path(__file__).with_name("app.py")
spec = importlib.util.spec_from_file_location("gradio_app", str(APP_PY_PATH))
gradio_app = importlib.util.module_from_spec(spec)
sys.modules["gradio_app"] = gradio_app
spec.loader.exec_module(gradio_app)  # This will run all top-level setup but not launch (guarded by __main__)

# Grab the objects we need
config = gradio_app.config
device = gradio_app.device
pipeline = gradio_app.pipeline
dist = gradio_app.dist
torch = gradio_app.torch

# --- Flask-SocketIO App ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "devkey")
socketio = SocketIO(app, cors_allowed_origins="*")

# Simple in-memory job queue and a sender thread
frame_queue = queue.Queue(maxsize=64)
latest_status = {"text": ""}

def to_jpeg_base64(frame_u8: np.ndarray, resize_ratio: float = 1.0, quality: int = 80) -> str:
    """frame_u8: HxWxC uint8"""
    img = Image.fromarray(frame_u8, mode="RGB")
    if resize_ratio != 1.0:
        w, h = img.size
        img = img.resize((int(w*resize_ratio), int(h*resize_ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def sender_worker():
    """Continuously send frames to clients at a steady FPS, using the latest queued frames."""
    target_fps = 12
    frame_period = 1.0 / target_fps
    last_emit = 0.0
    while True:
        item = frame_queue.get()
        if item is None:
            # Signal end of a stream
            socketio.emit("eos", {})
            frame_queue.task_done()
            continue
        # Item is already base64 jpeg
        b64, idx = item
        # throttle to target FPS
        now = time.time()
        wait = frame_period - (now - last_emit)
        if wait > 0:
            time.sleep(wait)
        socketio.emit("frame", {"data": b64, "index": idx})
        last_emit = time.time()
        frame_queue.task_done()

sender_thread = threading.Thread(target=sender_worker, daemon=True)
sender_thread.start()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("start")
def on_start(data):
    """
    data = {"prompt": "...", "seed": 42}
    """
    prompt = (data.get("prompt") or "").strip()
    seed = int(data.get("seed") or 42)

    # Kick off background task for generation
    socketio.start_background_task(target=generate_job, prompt=prompt, seed=seed)

def generate_job(prompt: str, seed: int):
    # set seed
    torch.manual_seed(int(seed))

    # allocate noise
    sampled_noise = torch.randn(
        [config.num_samples, 42, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16,
    )

    # Run model once to get the full video (pseudo-stream); if you later add pipeline.inference_stream,
    # replace the section below with true block/step streaming.
    video_btchw = pipeline.inference(
        noise=sampled_noise,
        text_prompt=prompt,
        switch_frame_indices=[],
        return_latents=False,
    )  # (B,T,C,H,W) in [0,1], torch

    video_thwc = (rearrange(video_btchw[0], "t c h w -> t h w c") * 255.0).clamp(0,255).to(torch.uint8).cpu().numpy()
    # enqueue frames
    for i, frame in enumerate(video_thwc):
        b64 = to_jpeg_base64(frame, resize_ratio=1.0, quality=80)
        try:
            frame_queue.put((b64, int(i)), timeout=5)
        except queue.Full:
            # drop if queue is full
            pass

    # Signal end of stream
    frame_queue.put((None, -1))
    frame_queue.put(None)  # special None item for sender to emit eos

if __name__ == "__main__":
    # Run with eventlet for proper WebSocket performance
    # Example: python app_socketio.py -- then visit http://localhost:5000
    # For production, consider: gunicorn -k eventlet -w 1 app_socketio:app --bind 0.0.0.0:5000
    socketio.run(app, host="0.0.0.0", port=5000)
