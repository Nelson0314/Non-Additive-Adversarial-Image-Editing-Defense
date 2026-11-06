"""非 SD 系編輯器測速（trial non_sd_editor_speed）：512²、主種子 20260812、Table 1 的四句 add 句型，
影像 man_01 原圖與 CAPGuard 2.5 倍防禦圖；暖機 1 次後計時每次編輯（含 VAE 編解碼、不含載入），記錄峰值 VRAM。
用法：python speed_editor.py <klein|klein_base|vibe|vibe_distilled> <輸出目錄>"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SEED = 20260812
PROMPTS = ["Add sunglasses to the person", "Add a police suit to the person", "Add a helmet to the person", "Add a bowtie to the person"]
BASE = Path.home() / "image-immunization"
IMAGES = {
    "orig": BASE / "baseline/data/pexels_portraits/man/man_01.png",
    "capguard": BASE / "color/trials/lut3d_defense_and_naturalness/artifacts_pexels/defenses/color3d_scale2p5_pexels/man_01__color3d_scale2p5_pexels__def.png",
}
which, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
device = "cuda"
t0 = time.time()
if which.startswith("klein"):
    from diffusers import Flux2KleinPipeline
    repo = {"klein": "black-forest-labs/FLUX.2-klein-4B", "klein_base": "black-forest-labs/FLUX.2-klein-base-4B"}[which]
    settings = {"klein": dict(num_inference_steps=4, guidance_scale=1.0), "klein_base": dict(num_inference_steps=50, guidance_scale=4.0)}[which]
    pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device)
    vae = pipe.vae

    def edit(img, prompt):
        g = torch.Generator(device).manual_seed(SEED)
        return pipe(prompt=prompt, image=img, height=512, width=512, generator=g, **settings).images[0]
else:
    from huggingface_hub import snapshot_download
    from vibe.editor import ImageEditor
    repo = {"vibe": "iitolstykh/VIBE-Image-Edit", "vibe_distilled": "iitolstykh/VIBE-Image-Edit-DistilledCFG"}[which]
    path = snapshot_download(repo_id=repo)
    settings = {"vibe": dict(image_guidance_scale=1.2, guidance_scale=4.5, num_inference_steps=20),
                "vibe_distilled": dict(num_inference_steps=20)}[which]
    editor = ImageEditor(checkpoint_path=path, device=device, **settings)
    vae = getattr(getattr(editor, "pipe", None) or getattr(editor, "pipeline", None), "vae", None)

    def edit(img, prompt):
        torch.manual_seed(SEED)
        res = editor.generate_edited_image(instruction=prompt, conditioning_image=img, num_images_per_prompt=1)
        return res[0] if isinstance(res, list) else res
torch.cuda.synchronize()
load_s = time.time() - t0
imgs = {k: Image.open(v).convert("RGB").resize((512, 512), Image.LANCZOS) for k, v in IMAGES.items()}
edit(imgs["orig"], PROMPTS[0])  # 暖機
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for name, img in imgs.items():
    for k, prompt in enumerate(PROMPTS):
        torch.cuda.synchronize(); t = time.time()
        y = edit(img, prompt)
        torch.cuda.synchronize(); times.append(time.time() - t)
        y.save(out / f"{which}__man_01_{name}__p{k}.png")
vae_info = {}
if vae is not None:
    cfg = dict(vae.config)
    vae_info = {"class": type(vae).__name__, "latent_channels": cfg.get("latent_channels"),
                "block_out_channels": cfg.get("block_out_channels"), "encoder_block_out_channels": cfg.get("encoder_block_out_channels"),
                "downsample_hint": cfg.get("downsample_block_type") or cfg.get("down_block_types")}
rec = {"model": which, "repo": repo, "settings": settings, "load_s": round(load_s, 1), "edits": len(times),
       "sec_per_edit_mean": round(sum(times) / len(times), 3), "sec_per_edit": [round(t, 3) for t in times],
       "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2), "gpu": torch.cuda.get_device_name(), "vae": vae_info}
(out / f"{which}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("[RESULT]", json.dumps(rec, ensure_ascii=False, default=str), flush=True)
