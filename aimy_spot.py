"""
AIMY シミ解析 — 統合 API（最終版）
=====================================

シワ検出と同じインターフェース。VISIA Brown Spots 風レンダリングを返す。

使い方:
    from aimy_spot import analyze_spots

    bgr = cv2.imread('face.jpg')
    result = analyze_spots(bgr)
    cv2.imwrite('out.jpg', result['visia_map'])

確定パラメータ:
    - レンダリング: v7 (mel=0.7, unsharp=(1.6,1.9,0.9), gamma=0.78)
    - 検出感度: very_high (n=140-170 程度)
    - 強調: soft (darken=0.30, red_shift=0.10, blur_sigma=3)
"""
import cv2
import numpy as np

from spot_detection import detect_spots, render_v7_sepia
from spot_visia_emphasis import render_visia_emphasis


# ---------- AIMY 統合 API ----------
def analyze_spots(bgr, sensitivity='very_high', return_map=True):
    """
    AIMY シミ解析メイン関数。

    Args:
        bgr: 入力画像 (BGR uint8, 任意サイズ)
        sensitivity: 'low' / 'medium' / 'high' / 'very_high' / 'max'
        return_map: True で VISIA 風マップ画像も生成

    Returns:
        dict with keys:
            'spots': list of {x, y, area, severity, score, bbox}
            'counts': {'small': N, 'medium': N, 'large': N, 'total': N}
            'visia_map': BGR image (return_map=True 時のみ)
            'spot_mask': バイナリマスク (H×W uint8)
    """
    h, w = bgr.shape[:2]
    scale = 1500 / max(h, w)
    if scale < 1.0:
        bgr_proc = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
    else:
        bgr_proc = bgr

    spots, mask = detect_spots(bgr_proc, sensitivity=sensitivity)

    counts = {'small': 0, 'medium': 0, 'large': 0}
    for s in spots:
        counts[s['severity']] += 1
    counts['total'] = len(spots)

    result = {
        'spots': spots,
        'counts': counts,
        'spot_mask': mask,
    }

    if return_map:
        sepia = render_v7_sepia(bgr_proc)
        visia_map = render_visia_emphasis(sepia, spots)
        result['visia_map'] = visia_map
        result['sepia_base'] = sepia

    return result
