"""
AIMY シミ検出 — 引き継ぎ仕様再構築版
- LAB の L + b デュアル Hessian ブロブ検出
- 形態学処理（small object removal, closing）
- 形状フィルタ（area, circularity, eccentricity）
- 大・中・小の3階層分類
- v7 セピア背景上に赤系オーバーレイで強調
"""
import cv2
import numpy as np

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label, regionprops

from visia_brown_v4 import (detect_landmarks, build_skin_soft_mask,
                             melanin_boost_gray, visia_brown_lut)
from visia_v67 import unsharp_multi


# ---------- Hessian ベース ブロブ検出 ----------
def hessian_blob_response(channel, sigma=2.0, dark=True):
    """
    Hessian の固有値による blob 検出。
    dark=True: 暗いブロブ（両固有値が正、その小さい方を採用）
    dark=False: 明るいブロブ（両固有値が負、その大きい方を反転）
    """
    img = channel.astype(np.float32)
    Hxx, Hxy, Hyy = hessian_matrix(img, sigma=sigma, order='rc')
    e1, e2 = hessian_matrix_eigvals((Hxx, Hxy, Hyy))

    if dark:
        resp = np.minimum(e1, e2)
        resp[resp < 0] = 0
    else:
        resp = -np.maximum(e1, e2)
        resp[resp < 0] = 0
    return resp


def detect_spots(bgr, sensitivity='medium', return_intermediate=False):
    """
    シミ検出メイン関数。

    Args:
        bgr: 入力画像 (BGR uint8)
        sensitivity: 'low' / 'medium' / 'high' （閾値パーセンタイル制御）
        return_intermediate: True で中間結果も返す

    Returns:
        spots: list of dict [{'x','y','area','severity','bbox'}]
        spot_mask: H×W バイナリマスク
        intermediate (optional): 中間処理結果
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 1. 肌マスク
    pts = detect_landmarks(rgb)
    if pts is None:
        raise RuntimeError('No face detected')
    feather = int(min(h, w) * 0.040)
    skin_soft = build_skin_soft_mask(rgb, pts, feather)
    skin_hard = (skin_soft > 0.6).astype(np.uint8) * 255

    # 2. LAB 変換
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    b = lab[..., 2]

    # 3. デュアル Hessian
    s = min(h, w) / 1000.0
    sigmas = [max(1.5, 1.5 * s),
              max(2.5, 2.5 * s),
              max(4.0, 4.0 * s)]

    resp_L = np.zeros_like(L, dtype=np.float32)
    resp_b = np.zeros_like(b, dtype=np.float32)
    for sig in sigmas:
        resp_L = np.maximum(resp_L, hessian_blob_response(L, sigma=sig, dark=True))
        resp_b = np.maximum(resp_b, hessian_blob_response(b, sigma=sig, dark=False))

    def norm01(x, mask):
        m = x[mask > 0]
        if len(m) == 0:
            return x
        p99 = np.percentile(m, 99)
        return np.clip(x / (p99 + 1e-6), 0, 1)

    resp_L_n = norm01(resp_L, skin_hard)
    resp_b_n = norm01(resp_b, skin_hard)

    # 4. 統合
    combined = np.sqrt(resp_L_n * resp_b_n) * 0.7 + np.maximum(resp_L_n * 0.3, resp_b_n * 0.3)
    combined *= skin_soft

    # 5. 閾値処理
    thr_pct = {
        'low': 98, 'medium': 96, 'high': 93,
        'very_high': 89, 'max': 85,
    }[sensitivity]
    skin_pixels = combined[skin_hard > 0]
    if len(skin_pixels) == 0:
        return [], np.zeros((h, w), np.uint8)
    threshold = np.percentile(skin_pixels, thr_pct)
    binary = combined > threshold

    # 6. 形態学
    min_area_base = {
        'low': 30, 'medium': 24, 'high': 20,
        'very_high': 15, 'max': 12,
    }[sensitivity]
    min_area_px = max(min_area_base, int(min_area_base * s * s))
    binary = remove_small_objects(binary, min_size=min_area_px)
    binary = binary_closing(binary, disk(max(1, int(2 * s))))

    # 7. 連結成分ごとの形状フィルタ
    lbl = label(binary)
    keep = np.zeros_like(binary)
    spots = []

    area_min = min_area_px
    area_max = int(300 * s * s)
    area_max_huge = int(1500 * s * s)

    skin_b_mean = b[skin_hard > 0].mean()

    for region in regionprops(lbl):
        area = region.area
        peri = region.perimeter
        circularity = 4 * np.pi * area / (peri ** 2 + 1e-6) if peri > 0 else 0
        ecc = region.eccentricity

        is_normal = (area_min <= area <= area_max
                     and circularity > 0.4
                     and ecc < 0.85)
        is_large = (area_max < area <= area_max_huge
                    and circularity > 0.3
                    and ecc < 0.9)

        if not (is_normal or is_large):
            continue

        b_threshold = {
            'low': 2.0, 'medium': 1.5, 'high': 1.0,
            'very_high': 0.5, 'max': 0.0,
        }[sensitivity]
        region_b = b[lbl == region.label].mean()
        if region_b - skin_b_mean < b_threshold:
            continue

        keep[lbl == region.label] = True

        cy, cx = region.centroid
        avg_resp = combined[lbl == region.label].mean()
        severity_score = area * 0.5 + avg_resp * 1000

        if area < area_min * 4:
            severity = 'small'
        elif area < area_max:
            severity = 'medium'
        else:
            severity = 'large'

        minr, minc, maxr, maxc = region.bbox
        spots.append({
            'x': int(cx), 'y': int(cy),
            'area': int(area),
            'circularity': float(circularity),
            'eccentricity': float(ecc),
            'severity': severity,
            'score': float(severity_score),
            'bbox': (int(minc), int(minr), int(maxc), int(maxr)),
        })

    spot_mask = keep.astype(np.uint8) * 255

    if return_intermediate:
        return spots, spot_mask, {
            'skin_soft': skin_soft,
            'L': L, 'b': b,
            'resp_L': resp_L_n, 'resp_b': resp_b_n,
            'combined': combined,
            'binary': binary,
        }
    return spots, spot_mask


# ---------- v7 セピア + 赤系3階層オーバーレイ ----------
def render_v7_sepia(bgr):
    """v7 セピア背景"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    pts = detect_landmarks(rgb)
    feather = int(min(h, w) * 0.040)
    skin_soft = build_skin_soft_mask(rgb, pts, feather)
    gray = melanin_boost_gray(rgb, mel_strength=0.7)
    s = min(h, w) / 1000.0
    sigmas = (max(2, int(3 * s)), max(4, int(10 * s)), max(8, int(25 * s)))
    boosted = unsharp_multi(gray, skin_soft, sigmas, (1.6, 1.9, 0.9))
    boosted = (np.power(boosted / 255.0, 1 / 0.78) * 255).astype(np.uint8)
    sepia = visia_brown_lut()[boosted]
    return cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)


def render_spots_overlay(bgr_sepia, spots, style='glow'):
    """
    シミ検出結果を 3階層の赤系オーバーレイで描画。
    style: 'circle' (中空円) / 'filled' (塗りつぶし) / 'glow' (光彩)
    """
    out = bgr_sepia.copy()
    overlay = np.zeros_like(out, dtype=np.float32)

    colors = {
        'small':  (60, 60, 220),
        'medium': (40, 40, 230),
        'large':  (20, 20, 255),
    }
    r_scale = {'small': 1.6, 'medium': 1.8, 'large': 2.0}

    for spot in spots:
        x, y = spot['x'], spot['y']
        area = spot['area']
        sev = spot['severity']
        color = colors[sev]
        base_r = max(4, int(np.sqrt(area / np.pi) * r_scale[sev]))

        if style == 'filled':
            cv2.circle(overlay, (x, y), base_r, color, -1, cv2.LINE_AA)
        elif style == 'glow':
            cv2.circle(overlay, (x, y), base_r, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (x, y), base_r, color, 2, cv2.LINE_AA)

    if style == 'glow':
        core = cv2.GaussianBlur(overlay, (0, 0), sigmaX=2)
        halo = cv2.GaussianBlur(overlay, (0, 0), sigmaX=10)
        overlay = np.clip(core * 1.0 + halo * 0.6, 0, 255)

    intensity = overlay.max(axis=2, keepdims=True) / 255.0
    alpha = intensity * (0.85 if style != 'circle' else 1.0)

    out_f = out.astype(np.float32)
    blended = out_f * (1 - alpha) + overlay * alpha / np.maximum(intensity, 1e-6)
    return blended.clip(0, 255).astype(np.uint8)
