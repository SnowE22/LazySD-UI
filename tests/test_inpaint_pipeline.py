# -*- coding: utf-8 -*-
"""
LazySD-UI Inpaint Test Runner
Directly tests the end-to-end backend inpainting pipeline:
Generates a test init image + binary circle mask, passes them to SDCppWorker,
runs sd-cli.exe with flux3 Q4_K_M, and verifies that the output image is created.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, r"D:\LazySD-UI")

from backend.settings import get_settings
from backend.worker import GenerateRequest
from backend.sd_cpp import SDCppWorker
from backend.registry.loader import get_registry
from backend.gpu import detect_gpu

def run_test():
    print("=== [TEST] LazySD-UI Inpaint Integration Test ===")
    settings = get_settings()
    
    # 1. Detect GPU
    gpu_info = detect_gpu()
    print(f"[*] Detected GPU: {gpu_info.name} ({gpu_info.vram_gb:.1f} GB VRAM)")

    # 2. Prepare test directory
    uploads_dir = settings.output_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    init_path = uploads_dir / "test_inpaint_init.png"
    mask_path = uploads_dir / "test_inpaint_mask.png"

    # Create 512x512 test RGB image (e.g. green background with red central square)
    w, h = 512, 512
    img_data = np.full((h, w, 3), [40, 140, 60], dtype=np.uint8)
    img_data[156:356, 156:356] = [220, 50, 50]
    Image.fromarray(img_data).save(init_path)
    print(f"[*] Created test init image: {init_path}")

    # Create 512x512 8-bit grayscale mask (white circle in center = inpaint area)
    mask_data = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    radius = 90
    y, x = np.ogrid[:h, :w]
    mask_data[(x - cx)**2 + (y - cy)**2 <= radius**2] = 255
    Image.fromarray(mask_data, mode="L").save(mask_path)
    print(f"[*] Created test inpaint mask: {mask_path}")

    # 3. Load Model Specs
    registry = get_registry()
    model = registry.get("flux_3_image_klein_gguf")
    variant_id = "q4_k_m"

    print(f"[*] Model: {model.id}, Variant: {variant_id}")

    # 4. Construct GenerateRequest
    request = GenerateRequest(
        model_id=model.id,
        variant_id=variant_id,
        prompt="a glowing golden pocket watch, intricate brass gears, highly detailed",
        negative_prompt="blurry, distorted, low quality",
        width=512,
        height=512,
        steps=16,
        cfg=1.0,
        noise=0.0,
        seed=12345,
        backend="sd_cpp",
        init_image_path=init_path,
        init_image_strength=0.75,
        mask_image_path=mask_path,
        mask_blur=4
    )

    print("[*] Starting SDCppWorker generation...")
    def on_progress(event):
        s = getattr(event, 'step', None)
        t = getattr(event, 'total_steps', None)
        pct = getattr(event, 'percentage', None)
        msg = getattr(event, 'message', '') or ''
        if s is not None and t is not None:
            print(f"    -> Progress: step {s}/{t} ({pct}%) {msg}")
        elif msg:
            print(f"    -> Status: {event.status} - {msg}")

    worker = SDCppWorker()
    res = worker.generate(request, model, gpu_info, progress_callback=on_progress)
    
    print("\n=== [RESULT] Generation Completed! ===")
    print(f"Status: {res.status}")
    print(f"Output Image: {res.image_path}")
    print(f"Duration: {res.duration_seconds:.2f}s")
    if res.error:
        print(f"Error: {res.error}")

if __name__ == "__main__":
    run_test()
