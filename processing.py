# -*- coding: utf-8 -*-
"""
AIMY 画像処理モジュール
server.py（ローカル）と app.py（本番 Flask）の両方から import して使う。
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ── FaceMesh ランドマーク定数 ──
LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE  = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
MOUTH_OUT  = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
MOUTH_IN   = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282]


# ═══════════════════════════════════════════
#  共通ユーティリティ
# ═══════════════════════════════════════════

def decode_image(img_bytes: bytes):
    """bytes → (bgr, alpha_ch)。JPEG にはアルファ全 255 を付与。"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("画像のデコードに失敗しました")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        alpha_ch = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    else:
        alpha_ch = img[:, :, 3]
    return img[:, :, :3], alpha_ch


def encode_png(bgr: np.ndarray, alpha_ch: np.ndarray) -> bytes:
    result = np.dstack([bgr, alpha_ch])
    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise RuntimeError("PNG エンコードに失敗しました")
    return buf.tobytes()


def poly_mask(h, w, landmarks_xy, indices):
    pts = np.array([landmarks_xy[i] for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def build_face_exclude(bgr, h, w, dilate=10, forehead_px=8):
    """MediaPipe で目・口・眉・額の除外マスクを生成。"""
    exclude = np.zeros((h, w), dtype=np.uint8)
    if not _MP_AVAILABLE:
        exclude[:int(h * 0.30), :] = 255
        return exclude

    mp_face_mesh = mp.solutions.face_mesh
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    ) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        exclude[:int(h * 0.30), :] = 255
        return exclude

    lms = res.multi_face_landmarks[0].landmark
    xy = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    for idx in [LEFT_EYE, RIGHT_EYE, MOUTH_OUT, MOUTH_IN, LEFT_BROW, RIGHT_BROW]:
        exclude = cv2.bitwise_or(exclude, poly_mask(h, w, xy, idx))

    brow_top_y = min(xy[i][1] for i in (LEFT_BROW + RIGHT_BROW))
    exclude[:max(0, brow_top_y - forehead_px), :] = 255

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        exclude = cv2.dilate(exclude, k, iterations=1)

    return exclude


# ═══════════════════════════════════════════
#  顔マスク生成（全処理の前処理）
# ═══════════════════════════════════════════

# MediaPipe FaceMesh の顔輪郭ランドマーク（時計回り順）
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


def process_face_mask(img_bytes: bytes, **_) -> bytes:
    """
    FACE_OVAL 36点のポリゴンで顔輪郭マスクを生成。
    - fillPoly で実際の輪郭を正確にトレース
    - 画像サイズに比例したフェザリング
    - MediaPipe 未インストール時は元画像全体をそのまま RGBA で返す
    """
    bgr, _ = decode_image(img_bytes)
    h, w = bgr.shape[:2]

    if not _MP_AVAILABLE:
        alpha_full = np.full((h, w), 255, dtype=np.uint8)
        return encode_png(bgr, alpha_full)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    ) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        print("[AIMY] face_mask: 顔が検出できませんでした。元画像をそのまま使用します。")
        alpha_full = np.full((h, w), 255, dtype=np.uint8)
        return encode_png(bgr, alpha_full)

    lms = res.multi_face_landmarks[0].landmark
    pts_all = np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms], dtype=np.int32)

    # FACE_OVAL 36点のポリゴンで顔輪郭を正確にトレース（凸包より精度が高い）
    face_pts = pts_all[FACE_OVAL]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [face_pts], 255)

    # 輪郭を少し拡張して顔端が欠けないようにする
    dilate_k = max(3, int(min(h, w) * 0.012)) | 1
    mask = cv2.dilate(mask, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

    # 画像サイズに比例したフェザリング（輪郭をなめらかに）
    feather_sigma = max(3, int(min(h, w) * 0.010))
    alpha_ch = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_sigma)

    print(f"[AIMY] face_mask: 完了 {w}x{h} dilate={dilate_k} feather_sigma={feather_sigma}")
    return encode_png(bgr, alpha_ch)


# ═══════════════════════════════════════════
#  シミ画像生成（しみ.py）
# ═══════════════════════════════════════════

def process_shimi(img_bytes: bytes, severity: float = 1.0, **_) -> bytes:
    """
    VISIA Brown Spots 風シミ解析マップを生成する。
    - Hessian ブロブ検出でシミを検出
    - v7 セピアレンダリング + VISIA 強調トーンマッピング
    """
    bgr, alpha_ch = decode_image(img_bytes)
    h, w = bgr.shape[:2]

    try:
        from aimy_spot import analyze_spots
        result = analyze_spots(bgr, sensitivity='very_high')
        visia = result['visia_map']
        # 元のサイズに戻す（analyze_spots は内部でリサイズする場合がある）
        if visia.shape[:2] != (h, w):
            visia = cv2.resize(visia, (w, h), interpolation=cv2.INTER_LANCZOS4)
        counts = result['counts']
        print(f"[AIMY] shimi VISIA: spots={counts['total']} "
              f"(s={counts['small']} m={counts['medium']} l={counts['large']})")
        return encode_png(visia, alpha_ch)
    except Exception as e:
        import traceback
        print(f"[AIMY] shimi VISIA エラー: {e}")
        traceback.print_exc()
        return encode_png(bgr, alpha_ch)


# ═══════════════════════════════════════════
#  テクスチャー画像生成（テクスチャー.py）
# ═══════════════════════════════════════════

def process_texture(img_bytes: bytes, severity: float = 1.0, seed: int = 42) -> bytes:
    bgr, alpha_ch = decode_image(img_bytes)
    h, w = bgr.shape[:2]
    face_mask = (alpha_ch > 0).astype(np.uint8) * 255
    face_mask_bool = (alpha_ch > 0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 背景ピクセルをblurの歪み防止のため顔領域平均で埋める
    if np.any(face_mask_bool):
        face_mean_val = float(gray[face_mask_bool].mean())
    else:
        face_mean_val = 128.0
    gray_filled = gray.astype(np.float32).copy()
    gray_filled[~face_mask_bool] = face_mean_val

    DARKEN_GAIN = max(0.5, 0.7 + (1.0 - severity) * 0.2)
    blur  = cv2.GaussianBlur(gray_filled.astype(np.uint8), (31, 31), 0)
    diff  = gray_filled - blur.astype(np.float32)
    dark  = (diff < -5.0).astype(np.uint8)
    dens  = cv2.GaussianBlur(dark.astype(np.float32), (31, 31), 0)
    dense = ((dark & (dens >= 0.12).astype(np.uint8)) == 1) & (face_mask > 0)

    enhanced = bgr.astype(np.float32)
    for c in range(3):
        enhanced[:, :, c][dense] *= DARKEN_GAIN
    result_bgr = np.clip(enhanced, 0, 255).astype(np.uint8)

    exclude = build_face_exclude(bgr, h, w, dilate=13)

    # 外縁重み
    dist = cv2.distanceTransform(face_mask, cv2.DIST_L2, 5)
    outer_w = (1.0 - dist / max(dist.max(), 1e-6)) ** 2.5

    # 疎な暗点（diff は gray_filled ベースなのでそのまま利用）
    dark2  = (diff < -2.8).astype(np.uint8)
    dens2  = cv2.GaussianBlur(dark2.astype(np.float32), (15, 15), 0)
    sparse = (dark2 & (dens2 < 0.45).astype(np.uint8)).astype(np.uint8)
    cand   = sparse.astype(bool) & (face_mask > 0) & (exclude == 0)

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / max(w - 1, 1), yy / max(h - 1, 1)

    def gauss2(cx, cy, sx, sy):
        return np.exp(-0.5 * (((xn - cx) / sx) ** 2 + ((yn - cy) / sy) ** 2))

    cheek_w = np.clip(gauss2(0.22, 0.58, 0.13, 0.16) + gauss2(0.78, 0.58, 0.13, 0.16)
                      - 0.9 * gauss2(0.5, 0.58, 0.12, 0.18), 0, 1)
    fa = np.clip((yn - 0.34) / 0.06, 0.0, 1.0)

    if np.any(cand):
        st = np.clip(DARKEN_GAIN - diff, 0, None)
        p90 = np.percentile(st[cand], 90) + 1e-6
        st = np.clip(st / p90, 0, 1) * (0.70 + 0.30 * cheek_w)

        def overlay(rate, alpha_max, color_bgr, dk=5, bk=9):
            prob = np.clip(outer_w * (0.35 + 0.65 * cheek_w) * fa * rate, 0, 0.95)
            sel  = (rng.random((h, w)).astype(np.float32) < prob) & cand
            if not np.any(sel):
                return result_bgr.copy()
            am = sel.astype(np.float32) * np.clip(st, 0, 1) * alpha_max
            blob = cv2.GaussianBlur(
                cv2.dilate(sel.astype(np.uint8) * 255,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))).astype(np.float32),
                (bk, bk), 0) / 255.0
            am = np.clip(am * (0.55 + 0.45 * blob), 0, alpha_max)
            out = result_bgr.astype(np.float32)
            color = np.array(color_bgr, dtype=np.float32)
            return np.clip(out * (1 - am[..., None]) + color * am[..., None], 0, 255).astype(np.uint8)

        result_bgr = overlay(0.010, 0.72 * severity, (131, 0, 142))
        result_bgr = overlay(0.008, 0.95 * severity, (5, 130, 255))

    return encode_png(result_bgr, alpha_ch)


# ═══════════════════════════════════════════
#  毛穴画像生成（毛穴.py）
# ═══════════════════════════════════════════

def process_keana(img_bytes: bytes, severity: float = 1.0, **_) -> bytes:
    """
    Black-hat morphology で毛穴を強調した解析マップを生成する。
    - 各チャンネルの Black-hat 成分を減算して毛穴を暗く浮き上がらせる
    - 肌マスク領域（目・眉・唇を除く）のみに適用
    """
    bgr, alpha_ch = decode_image(img_bytes)
    h, w = bgr.shape[:2]

    try:
        from aimy_pore import visualize_pores
        result = visualize_pores(bgr)
        # 元のサイズに戻す（visualize_pores は内部でリサイズする場合がある）
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        print(f"[AIMY] keana BlackHat: 完了 {w}x{h}")
        return encode_png(result, alpha_ch)
    except Exception as e:
        print(f"[AIMY] keana BlackHat エラー ({e})、フォールバック実行")
        return encode_png(bgr, alpha_ch)


# ═══════════════════════════════════════════
#  赤み画像生成（赤み.py）
# ═══════════════════════════════════════════

def process_akami(img_bytes: bytes, severity: float = 1.0, seed: int = 42) -> bytes:
    bgr, alpha_ch = decode_image(img_bytes)
    h, w = bgr.shape[:2]
    face_mask = (alpha_ch > 0).astype(np.uint8) * 255

    GLOBAL_TH   = max(1.0, 4.0 - (severity - 0.5) * 4.0)
    ALPHA_MAX   = min(0.99, 0.70 + severity * 0.29)
    RED_BGR     = (19, 18, 160)  # #a01224 → BGR

    exclude = build_face_exclude(bgr, h, w, dilate=10)
    valid   = (face_mask > 0) & (exclude == 0)

    ycrcb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    red_ch   = ycrcb[:, :, 1].astype(np.float32)

    if np.count_nonzero(valid) == 0:
        return encode_png(bgr, alpha_ch)

    g_mean   = float(red_ch[valid].mean())
    strength = red_ch - g_mean
    cand     = valid & (strength > GLOBAL_TH)

    alpha_map = np.clip((strength - GLOBAL_TH) / 20.0, 0.0, 1.0) * ALPHA_MAX
    alpha_map[~cand] = 0.0

    out = bgr.copy().astype(np.float32)
    for c, tgt in enumerate(RED_BGR):
        out[:, :, c] = np.clip(out[:, :, c] * (1.0 - alpha_map) + tgt * alpha_map, 0, 255)

    print(f"[AIMY] 赤み: mean={g_mean:.2f} th={GLOBAL_TH:.2f} px={int(np.count_nonzero(cand))}")
    return encode_png(out.astype(np.uint8), alpha_ch)


# ═══════════════════════════════════════════
#  シワ画像生成（map_shiwa.py）
# ═══════════════════════════════════════════

def process_shiwa(img_bytes: bytes, severity: float = 1.0, seed: int = 42) -> bytes:
    """
    参照画像に合わせたシワ描画：
    - 額：4〜5本の横ウェーブライン（途中で分断あり）
    - 目尻：10本前後の放射線 + 下まぶた短線
    - ほうれい線：鼻翼→口角→顎まで延長した2本並行線
    - 顎ライン：黄緑色のアウトライン
    - 眉間：3本の縦線
    """
    bgr, alpha_ch = decode_image(img_bytes)
    h, w = bgr.shape[:2]

    if not _MP_AVAILABLE:
        return encode_png(bgr, alpha_ch)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    ) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        print("[AIMY] shiwa: 顔が検出できませんでした")
        return encode_png(bgr, alpha_ch)

    lms = res.multi_face_landmarks[0].landmark
    xy  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    rng    = np.random.default_rng(seed)
    result = bgr.copy()
    GREEN  = (30, 240, 80)   # ライムグリーン（BGR）
    YELLOW = (20, 230, 220)  # 黄緑（顎ライン用）
    lw     = max(2, int(min(h, w) * 0.005))

    def polyline(pts, color=GREEN, thickness=None):
        if len(pts) < 2:
            return
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [arr], False, color, thickness or lw)

    def bezier3(p0, cp1, cp2, p1, steps=40):
        """三次ベジェ曲線の点列を返す"""
        pts = []
        for j in range(steps):
            t = j / (steps - 1)
            x = int((1-t)**3*p0[0] + 3*(1-t)**2*t*cp1[0] + 3*(1-t)*t**2*cp2[0] + t**3*p1[0])
            y = int((1-t)**3*p0[1] + 3*(1-t)**2*t*cp1[1] + 3*(1-t)*t**2*cp2[1] + t**3*p1[1])
            pts.append([x, y])
        return pts

    # ── 1. 額の横ライン（4〜5本）──
    brow_top_y   = min(xy[i][1] for i in LEFT_BROW + RIGHT_BROW)
    brow_left_x  = min(xy[i][0] for i in LEFT_BROW)
    brow_right_x = max(xy[i][0] for i in RIGHT_BROW)
    face_top_y   = xy[10][1]
    brow_width   = brow_right_x - brow_left_x

    n_forehead = 5
    for i in range(n_forehead):
        t = (i + 0.5) / n_forehead
        line_y = int(brow_top_y - t * (brow_top_y - face_top_y) * 0.92)
        margin = int(brow_width * 0.06)
        x0, x1 = brow_left_x - margin, brow_right_x + margin
        amp   = 3 + int(rng.integers(0, 4))
        freq  = 0.08 + float(rng.uniform(0, 0.05))
        phase = float(rng.uniform(0, 2 * np.pi))
        pts = []
        for x in range(x0, x1, 3):
            wave = int(amp * np.sin(x * freq + phase))
            pts.append([x, line_y + wave])
        # 70%の確率で途中分断
        if rng.random() < 0.7 and len(pts) > 10:
            cut = rng.integers(int(len(pts)*0.3), int(len(pts)*0.7))
            gap = rng.integers(3, 10)
            polyline(pts[:cut - gap//2], thickness=lw)
            polyline(pts[cut + gap//2:], thickness=lw)
        else:
            polyline(pts, thickness=lw)

    # ── 2. 目尻のシワ（カラスの足跡）──
    # 左eye: outer=33, 右eye: outer=263
    # 下まぶた: 左=144/153/158, 右=373/380/385
    for side, outer_lm, lower_lms in [
        (-1, 33,  [144, 153, 158]),
        ( 1, 263, [373, 380, 385])
    ]:
        ox, oy = xy[outer_lm]
        n_crow = 10
        for k in range(n_crow):
            # -60° 〜 +60° の範囲で放射
            angle = np.radians(-60 + k * 120 / (n_crow - 1))
            length = int(min(h, w) * (0.035 + float(rng.uniform(0, 0.04))))
            ex = int(ox + side * np.cos(angle) * length)
            ey = int(oy + np.sin(angle) * length)
            n_pts = max(abs(ex - ox), abs(ey - oy), 5) + 1
            pts = []
            for j in range(n_pts):
                t = j / max(1, n_pts - 1)
                x = ox + t * (ex - ox)
                y = oy + t * (ey - oy)
                perp_x = -(ey - oy) / max(1, n_pts)
                perp_y =  (ex - ox) / max(1, n_pts)
                wave   = 1.8 * np.sin(j * 0.38)
                pts.append([int(x + perp_x * wave), int(y + perp_y * wave)])
            polyline(pts, thickness=max(1, lw - 1))

        # 下まぶた短線
        for lm_idx in lower_lms:
            lx, ly = xy[lm_idx]
            ex2 = int(lx + side * int(rng.integers(12, 28)))
            ey2 = int(ly + int(rng.integers(4, 12)))
            polyline([[lx, ly], [ex2, ey2]], thickness=max(1, lw - 1))

    # ── 3. ほうれい線（鼻翼→口角→顎まで延長、2本並行）──
    chin_x, chin_y = xy[152]
    for nose_lm, mouth_lm, side in [(49, 61, -1), (279, 291, 1)]:
        nx, ny = xy[nose_lm]
        mx, my = xy[mouth_lm]
        dx = abs(mx - nx)
        dy = my - ny
        end_x = int(chin_x + side * int(w * 0.04))
        end_y = int(chin_y + int(h * 0.015))

        # 鼻翼→口角 の制御点
        cp1 = (int(nx + side * dx * 0.55), int(ny + dy * 0.28))
        cp2 = (int(mx + side * dx * 0.32), int(my - dy * 0.15))

        for offset in [0, int(w * 0.011)]:
            ox_off = offset * side
            # 鼻翼→口角 ベジェ
            seg1 = bezier3((nx, ny), cp1, cp2, (mx, my), steps=30)
            # 口角→顎 直線補間
            seg2 = []
            for j in range(20):
                t = j / 19
                seg2.append([int(mx + t*(end_x - mx)), int(my + t*(end_y - my))])
            full = [[p[0] + ox_off, p[1]] for p in seg1 + seg2]
            polyline(full, thickness=lw)

    # ── 4. 顎ライン（黄緑・FACE_OVAL下半分）──
    JAW_OVAL = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
    jaw_base = [xy[i] for i in JAW_OVAL]
    for offset_y in [0, 5, 10]:
        pts = [[p[0], p[1] + offset_y] for p in jaw_base]
        polyline(pts, color=YELLOW, thickness=max(1, lw - 1))

    # ── 5. 眉間の縦線（3本）──
    gx, gy = xy[9]
    for k in range(3):
        offset_x = -8 + k * 8
        top_y = gy - int(h * 0.042)
        bot_y = gy + int(h * 0.018)
        pts = []
        for j in range(16):
            t = j / 15
            x = int(gx + offset_x * (1 - t * 0.35))
            y = int(top_y + t * (bot_y - top_y))
            pts.append([x + int(1.5 * np.sin(j * 0.55)), y])
        polyline(pts, thickness=lw)

    print(f"[AIMY] shiwa: severity={severity}")
    return encode_png(result, alpha_ch)
