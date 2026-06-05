"""
VISIA Brown Spots レンダリング v4
- CLAHE を廃止（境界が斑になる問題）
- 肌領域だけ Unsharp Mask で局所コントラスト微強調
- 全画像にシンプルな γ + セピア LUT
"""
import cv2
import numpy as np
import mediapipe as mp


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 113, 225, 224, 223, 222, 221]
RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 342, 445, 444, 443, 442, 441]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]


def detect_landmarks(rgb):
    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5)
    res = fm.process(rgb)
    fm.close()
    if not res.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    return np.array([(int(p.x * w), int(p.y * h)) for p in lm], np.int32)


def build_skin_soft_mask(rgb, pts, feather):
    """ソフト肌マスク（0-1 float）。境界は完全フェザリング。"""
    h, w = rgb.shape[:2]

    face = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face, [pts[FACE_OVAL]], 255)

    excl = np.zeros((h, w), np.uint8)
    base_k = max(3, int(min(h, w) * 0.012)) | 1
    for indices, scale in [
        (LEFT_EYE, 1.5), (RIGHT_EYE, 1.5),
        (LEFT_BROW, 2.5), (RIGHT_BROW, 2.5),
        (LIPS_OUTER, 1.0),
    ]:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts[indices]], 255)
        k = max(3, int(base_k * scale)) | 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
        excl = cv2.bitwise_or(excl, m)

    skin = cv2.bitwise_and(face, cv2.bitwise_not(excl))

    # 髪・極暗部は除外
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    V = hsv[..., 2]
    skin[V < 70] = 0

    # フェザリングして 0-1 float に
    k = feather * 2 + 1
    soft = cv2.GaussianBlur(skin.astype(np.float32), (k, k), 0) / 255.0
    return soft


def visia_brown_lut():
    """グレー → セピアの 256x3 LUT"""
    keys = np.array([
        (  0, 105,  65,  45),
        ( 50, 145,  95,  70),
        (110, 195, 135, 100),
        (170, 230, 175, 140),
        (220, 245, 205, 175),
        (255, 250, 220, 195),
    ], dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(np.arange(256), keys[:, 0], keys[:, c+1])
    return lut.clip(0, 255).astype(np.uint8)


def melanin_boost_gray(rgb, mel_strength=0.35):
    """L 値ベース + メラニン分布で軽くシミ強調したグレースケール"""
    R = rgb[..., 0].astype(np.float32) + 1
    B = rgb[..., 2].astype(np.float32) + 1
    mel = np.log(R) - np.log(B)  # メラニン指標
    mel_n = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32) / 255.0

    boosted = L * (1 - mel_n * mel_strength)
    return (boosted * 255).clip(0, 255).astype(np.uint8)


def unsharp_mask_local(gray, soft_mask, sigmas=(4, 12, 30), amounts=(0.8, 1.0, 0.6)):
    """肌領域だけ多段 unsharp mask（細かい〜大きい特徴を多周波数で強調）"""
    g = gray.astype(np.float32)
    boost = g.copy()
    for sigma, amount in zip(sigmas, amounts):
        blur = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)
        high = g - blur
        boost += high * amount * soft_mask
    return boost.clip(0, 255).astype(np.uint8)


def render(bgr, mel_strength=0.5, gamma=0.95):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    pts = detect_landmarks(rgb)
    if pts is None:
        raise RuntimeError('No face detected')

    feather = int(min(h, w) * 0.040)
    skin_soft = build_skin_soft_mask(rgb, pts, feather)

    gray = melanin_boost_gray(rgb, mel_strength=mel_strength)

    s = min(h, w) / 1000.0
    sigmas = (max(2, int(3 * s)),
              max(4, int(10 * s)),
              max(8, int(25 * s)))
    amounts = (0.9, 1.1, 0.5)
    boosted = unsharp_mask_local(gray, skin_soft, sigmas=sigmas, amounts=amounts)

    if gamma != 1.0:
        boosted = (np.power(boosted / 255.0, 1 / gamma) * 255).astype(np.uint8)

    sepia = visia_brown_lut()[boosted]
    return cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR), skin_soft, gray, boosted
