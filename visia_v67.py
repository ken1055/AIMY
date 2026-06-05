"""
v6/v7: VISIA 寄り強調アップ
"""
import cv2
import numpy as np
from visia_brown_v4 import (detect_landmarks, build_skin_soft_mask,
                             melanin_boost_gray, visia_brown_lut)


def unsharp_multi(gray, soft_mask, sigmas, amounts):
    g = gray.astype(np.float32)
    boost = g.copy()
    for sigma, amount in zip(sigmas, amounts):
        blur = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)
        high = g - blur
        boost += high * amount * soft_mask
    return boost.clip(0, 255).astype(np.uint8)


def render_with_intensity(bgr, mel_strength, amounts, gamma):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    pts = detect_landmarks(rgb)
    feather = int(min(h, w) * 0.040)
    skin_soft = build_skin_soft_mask(rgb, pts, feather)

    gray = melanin_boost_gray(rgb, mel_strength=mel_strength)

    s = min(h, w) / 1000.0
    sigmas = (max(2, int(3 * s)),
              max(4, int(10 * s)),
              max(8, int(25 * s)))
    boosted = unsharp_multi(gray, skin_soft, sigmas, amounts)

    if gamma != 1.0:
        boosted = (np.power(boosted / 255.0, 1 / gamma) * 255).astype(np.uint8)

    sepia = visia_brown_lut()[boosted]
    return cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)
