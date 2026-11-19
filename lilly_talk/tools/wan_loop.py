#!/usr/bin/env python3
"""Nahtloser Video-Loop aus einem Standbild mit Wan 2.2 I2V (Apache 2.0) auf dem ComfyUI-PC.

Erstes und letztes Bild = Standbild (WanFirstLastFrameToVideo), Lightning-LoRAs (4 Schritte),
Einzelbilder werden geladen und lokal mit ffmpeg zu einem stummen H.264-MP4 zusammengesetzt.

  python3 wan_loop.py bild.png "was sich bewegt" ausgabe.mp4 [--seed 7] [--w 480 --h 640 --frames 81]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

NEG = ("static, frozen, blurry, low quality, distorted face, deformed hands, extra fingers, text, watermark, "
       "subtitles, jitter, flicker, sudden camera movement, zoom, cut")


def endpoint() -> str:
    base = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000").rstrip("/")
    host, _, port = base.split("//", 1)[-1].partition(":")
    try:
        socket.create_connection((host, int(port or 80)), timeout=3).close()
    except OSError:
        sys.exit(f"ComfyUI nicht erreichbar ({base}); COMFYUI_URL setzen")
    return base


def post_json(base, path, data):
    req = urllib.request.Request(base + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def upload(base, image: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{image.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + image.read_bytes() + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req, timeout=120))["name"]


def workflow(image_name, prompt, seed, w, h, frames, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 5.0}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": 5.0}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": NEG}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "12": {"class_type": "WanFirstLastFrameToVideo", "inputs": {
            "positive": ["8", 0], "negative": ["9", 0], "vae": ["10", 0], "width": w, "height": h,
            "length": frames, "batch_size": 1, "start_image": ["11", 0], "end_image": ["11", 0]}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["5", 0], "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["12", 0], "negative": ["12", 1],
            "latent_image": ["12", 2], "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable"}},
        "14": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["6", 0], "add_noise": "disable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["12", 0], "negative": ["12", 1],
            "latent_image": ["13", 0], "start_at_step": 2, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["10", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("prompt")
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--fps", type=int, default=16)
    a = ap.parse_args()
    base = endpoint()
    name = upload(base, a.image)
    prefix = f"lilly_loop/{a.out.stem}_{uuid.uuid4().hex[:6]}"
    pid = post_json(base, "/prompt", {"prompt": workflow(name, a.prompt, a.seed, a.w, a.h, a.frames, prefix)})["prompt_id"]
    print(f"gestartet {pid} auf {base}", flush=True)
    t0 = time.time()
    while True:
        time.sleep(5)
        try:
            hist = json.load(urllib.request.urlopen(f"{base}/history/{pid}", timeout=30))
        except OSError:
            continue  # PC unter Last antwortet manchmal nicht
        if pid in hist:
            h = hist[pid]
            if h.get("status", {}).get("status_str") == "error":
                sys.exit("ComfyUI-Fehler: " + json.dumps(h["status"].get("messages", []))[:800])
            images = h["outputs"]["16"]["images"]
            break
        if time.time() - t0 > 3600:
            sys.exit("Zeitüberschreitung")
    with tempfile.TemporaryDirectory() as tmp:
        for i, im in enumerate(images):
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            Path(tmp, f"f{i:04d}.png").write_bytes(urllib.request.urlopen(f"{base}/view?{q}", timeout=60).read())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        # letztes Bild weglassen (= erstes), damit der Loop nicht doppelt steht
        n = len(images) - 1
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", f"{tmp}/f%04d.png",
                        "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        "-movflags", "+faststart", "-an", str(a.out)], check=True)
    print(f"fertig {a.out} ({n} Bilder, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
