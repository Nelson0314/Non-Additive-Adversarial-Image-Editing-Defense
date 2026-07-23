"""SDEdit 編輯（img2img）— SPEC.md §2.8、STRUCTURE.md §2.2。

使用 diffusers 之 img2img（即 SDEdit 之實作）。
生成超參數依 SPEC §2.2：height=512, width=512, guidance_scale=7.5,
num_inference_steps=100, eta=1。
strength 由 config 提供（SPEC §2.8 標記待確認，向 DAYN 作者索取）。
"""
