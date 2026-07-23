"""AdverseCleaner（= BF+GF）— SPEC.md §5.1。

cv2 bilateral filter + cv2.ximgproc.guidedFilter（需 opencv-contrib-python）。
實作兩變體：僅 bilateral filter（bf_only）、完整 BF+GF（bf_gf），
以觀察 guided filter 是否把對抗噪聲帶回而削弱淨化效果。
"""
