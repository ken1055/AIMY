"""
AIMY: AI Skin Analysis - Pore Visualization Module
====================================================
毛穴可視化のエントリーポイント。

肌の凹みである毛穴を Black-hat morphology で抽出し、
各色チャンネルから減算することで「肌色は保ったまま毛穴が黒い点として浮き出る」
表現を生成する。

Usage:
    from aimy_pore import visualize_pores
    result_bgr = visualize_pores(bgr)

Parameters (固定):
    手法:    BlackHatColor (各チャンネルに Black-hat morphology)
    ksize:   7   (毛穴サイズの構造要素)
    gain:    4.5 (黒さ強調倍率)
"""
import cv2
import numpy as np
import mediapipe as mp


# ─── ランドマーク定義 ───────────────────────────────────────
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
             400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,
             54,103,67,109]
LEFT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE  = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
LEFT_BROW  = [70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW = [300,293,334,296,336,285,295,282,283,276]
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,
              37,39,40,185]

# ─── 確定パラメータ ─────────────────────────────────────────
PORE_PARAMS = {
    'ksize': 7,
    'gain':  4.5,
    'resize_max': 1500,
}


def _build_skin_mask(bgr, P):
    """肌マスクを生成（顔輪郭 − 目 − 眉 − 唇）"""
    H, W = bgr.shape[:2]
    skin = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(skin, cv2.convexHull(P[FACE_OVAL]), 255)

    for idx in [LEFT_EYE, RIGHT_EYE]:
        pts = P[idx]; cx, cy = pts.mean(axis=0)
        pts_exp = np.array([(int(cx+(x-cx)*1.40),
                             int(cy+(y-cy)*(2.5 if y < cy else 1.3)))
                             for x, y in pts])
        cv2.fillConvexPoly(skin, cv2.convexHull(pts_exp), 0)

    for idx in [LEFT_BROW, RIGHT_BROW]:
        pts = P[idx]; cx, cy = pts.mean(axis=0)
        pts_exp = np.array([(int(cx+(x-cx)*1.20),
                             int(cy+(y-cy)*1.6 if y < cy else cy+(y-cy)*1.4))
                             for x, y in pts])
        cv2.fillConvexPoly(skin, cv2.convexHull(pts_exp), 0)

    pts = P[LIPS_OUTER]; cx, cy = pts.mean(axis=0)
    pts_exp = np.array([(int(cx+(x-cx)*1.18), int(cy+(y-cy)*1.3))
                         for x, y in pts])
    cv2.fillConvexPoly(skin, cv2.convexHull(pts_exp), 0)

    return skin


def _detect_landmarks(bgr):
    """MediaPipe で顔ランドマーク検出"""
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
            refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.5) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        raise RuntimeError('Face not detected')
    lm = res.multi_face_landmarks[0].landmark
    P = np.array([(int(p.x*W), int(p.y*H)) for p in lm])
    return P


def visualize_pores(bgr, params=None):
    """
    毛穴を強調した画像を生成（元画像と同サイズ、BGR 3ch）

    Args:
        bgr: 入力画像 (numpy.ndarray, BGR, uint8)
        params: パラメータの上書き（任意）

    Returns:
        np.ndarray (BGR, uint8): 毛穴強調済み画像
    """
    p = PORE_PARAMS.copy()
    if params: p.update(params)

    h, w = bgr.shape[:2]
    scale = p['resize_max'] / max(h, w)
    if scale < 1.0:
        bgr_proc = cv2.resize(bgr, (int(w*scale), int(h*scale)),
                              interpolation=cv2.INTER_AREA)
    else:
        bgr_proc = bgr.copy()

    P = _detect_landmarks(bgr_proc)
    skin = _build_skin_mask(bgr_proc, P)

    ksize = p['ksize']
    gain = p['gain']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    out = bgr_proc.copy().astype(np.float32)
    for c in range(3):
        bh = cv2.morphologyEx(bgr_proc[..., c], cv2.MORPH_BLACKHAT, kernel)
        out[..., c] = np.clip(out[..., c] - bh.astype(np.float32) * gain, 0, 255)
    out = out.astype(np.uint8)

    result = bgr_proc.copy()
    mask_3 = (skin > 0)[..., None]
    result = np.where(mask_3, out, result)

    return result
