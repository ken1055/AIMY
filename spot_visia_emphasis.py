"""
VISIA 流のシミ強調（マーカー描画ではなくトーンマッピング）
- 検出された spot 位置を中心にソフトな強度マップを作成
- v7 セピア画像のシミ位置を「暗く・赤茶寄りに沈める」
- 周囲となだらかに繋がる自然な強調
"""
import cv2
import numpy as np
from spot_detection import detect_spots, render_v7_sepia


def build_spot_strength_map(spots, shape, base_blur_sigma=4):
    """個別 spot を中心にソフトな強度マップを生成"""
    h, w = shape[:2]
    strength = np.zeros((h, w), dtype=np.float32)

    for spot in spots:
        x, y, area = spot['x'], spot['y'], spot['area']
        sev = spot['severity']
        weight = {'small': 0.7, 'medium': 0.85, 'large': 1.0}[sev]
        r = max(3, int(np.sqrt(area / np.pi) * 1.2))
        cv2.circle(strength, (x, y), r, weight, -1, cv2.LINE_AA)

    strength = cv2.GaussianBlur(strength, (0, 0), sigmaX=base_blur_sigma)
    return np.clip(strength, 0, 1)


def render_visia_emphasis(bgr_sepia, spots,
                          darken_amount=0.30,
                          red_shift=0.10,
                          blur_sigma=3):
    """
    VISIA 流: セピア画像のシミ位置を暗く＆赤茶寄りに沈める。
    デフォルトは soft プリセット（確定値）。
    """
    strength = build_spot_strength_map(spots, bgr_sepia.shape, blur_sigma)
    s = strength[..., None]

    out_f = bgr_sepia.astype(np.float32)
    B, G, R = out_f[..., 0:1], out_f[..., 1:2], out_f[..., 2:3]

    decay_B = 1.0 - s * (darken_amount + red_shift * 1.2)
    decay_G = 1.0 - s * (darken_amount + red_shift * 0.5)
    decay_R = 1.0 - s * (darken_amount * 0.7)
    new_B = B * decay_B.clip(0, 1)
    new_G = G * decay_G.clip(0, 1)
    new_R = R * decay_R.clip(0, 1)

    out = np.concatenate([new_B, new_G, new_R], axis=2)
    return out.clip(0, 255).astype(np.uint8)
