"""PrivateVoice_Ahee52 — 三个 logo / icon 方案（1024 PNG + ICO + 对比图）。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 1024
SS = 3                      # 超采样倍数
OUT = Path(r"C:\Local Repositories\PrivateVoice\results\logo")
OUT.mkdir(parents=True, exist_ok=True)

BG1 = (27, 20, 48)
BG2 = (46, 31, 79)
PURPLE = (139, 92, 246)
VIOLET = (167, 139, 250)
PINK = (236, 122, 214)
AMBER = (255, 184, 107)
MINT = (94, 234, 212)
WHITE = (255, 255, 255)


def gradient(size: int, c1=BG1, c2=BG2, angle: float = 135) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    a = math.radians(angle)
    t = np.clip(x * math.cos(a) + y * math.sin(a), 0, 1)
    t = t ** 1.05
    arr = np.zeros((size, size, 3), np.float32)
    for i in range(3):
        arr[..., i] = c1[i] + (c2[i] - c1[i]) * t
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def rounded_mask(size: int, radius_ratio: float = 0.235) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    r = int(size * radius_ratio)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    return m


def glow(img: Image.Image, center, radius, color, strength=1.0) -> Image.Image:
    layer = Image.new("RGB", (img.width, img.height), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = center
    steps = 26
    for i in range(steps, 0, -1):
        f = i / steps
        r = radius * f
        c = tuple(int(v * (1 - f) ** 2 * strength) for v in color)
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    layer = layer.filter(ImageFilter.GaussianBlur(radius * 0.16))
    return Image.fromarray(
        np.clip(np.asarray(img, np.int16) + np.asarray(layer, np.int16), 0, 255).astype(np.uint8))


def bar(d: ImageDraw.ImageDraw, cx: float, cy: float, w: float, h: float,
        color, radius: float | None = None) -> None:
    r = w / 2 if radius is None else radius
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=r, fill=color)


def compose(bg: Image.Image, art: Image.Image, size: int) -> Image.Image:
    mask = rounded_mask(size)
    return Image.composite(art, bg, mask)


# ---------------------------------------------------------------- 方案 A：声波微笑
def concept_a(size: int = S * SS) -> Image.Image:
    bg = gradient(size).resize((size, size), Image.LANCZOS)
    bg = glow(bg, (size * 0.5, size * 0.42), size * 0.46, (60, 30, 120), 1.1)
    art = bg.convert("RGB")
    d = ImageDraw.Draw(art, "RGBA")

    cx, cy = size * 0.5, size * 0.46
    # 外圈细环
    ring_r = size * 0.30
    d.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
              outline=(*VIOLET, 70), width=int(size * 0.006))

    # 五根声波柱，中间最高
    heights = [0.17, 0.30, 0.42, 0.30, 0.17]
    bw = size * 0.062
    gap = size * 0.083
    cols = [VIOLET, (196, 168, 255), WHITE, (196, 168, 255), VIOLET]
    for i, (h, c) in enumerate(zip(heights, cols)):
        x = cx + (i - 2) * gap
        bar(d, x, cy, bw, size * h, (*c, 255))
    # 微笑弧
    r = size * 0.175
    d.arc([cx - r, cy - r * 0.72, cx + r, cy + r * 1.30],
          start=25, end=155, fill=(*AMBER, 235), width=int(size * 0.022))
    return compose(gradient(size, (24, 18, 44), (58, 38, 96)), art, size)


# ---------------------------------------------------------------- 方案 B：对话气泡
def concept_b(size: int = S * SS) -> Image.Image:
    bg = gradient(size, (22, 17, 40), (52, 30, 88))
    art = bg.convert("RGB")
    d = ImageDraw.Draw(art, "RGBA")

    cx, cy = size * 0.5, size * 0.47
    w, h = size * 0.52, size * 0.40
    rect = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    d.rounded_rectangle(rect, radius=size * 0.11, fill=(*PURPLE, 255))
    # 气泡尾巴
    d.polygon([(cx - w * 0.12, cy + h / 2 - size * 0.01),
               (cx - w * 0.02, cy + h / 2 + size * 0.115),
               (cx + w * 0.16, cy + h / 2 - size * 0.02)], fill=(*PURPLE, 255))
    # 内部渐变高光
    gloss = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gloss)
    gd.ellipse([cx - w * 0.75, cy - h * 1.05, cx + w * 0.75, cy + h * 0.55],
               fill=(*PINK, 70))
    gloss = gloss.filter(ImageFilter.GaussianBlur(size * 0.05))
    art = Image.alpha_composite(art.convert("RGBA"), gloss).convert("RGB")
    d = ImageDraw.Draw(art, "RGBA")

    # 气泡内声波（挖空）
    heights = [0.10, 0.19, 0.26, 0.19, 0.10]
    bw = size * 0.045
    gap = size * 0.062
    for i, hh in enumerate(heights):
        bar(d, cx + (i - 2) * gap, cy - size * 0.005, bw, size * hh, (255, 255, 255, 245))

    # 气泡内底部两点（“正在回应”的暗示）
    for i, alpha in enumerate((235, 140)):
        rr = size * 0.018
        px = cx + (i - 0.5) * size * 0.062
        py = cy + h * 0.34
        d.ellipse([px - rr, py - rr, px + rr, py + rr],
                  fill=(255, 255, 255, alpha))
    return compose(gradient(size, (20, 16, 38), (54, 32, 90)), art, size)


# ---------------------------------------------------------------- 方案 C：双弧拥抱
def concept_c(size: int = S * SS) -> Image.Image:
    bg = gradient(size, (18, 16, 36), (40, 28, 74))
    art = bg.convert("RGB")
    art = glow(art, (size * 0.5, size * 0.5), size * 0.50, (36, 26, 84), 1.2)
    d = ImageDraw.Draw(art, "RGBA")
    cx, cy = size * 0.5, size * 0.5

    # 左右两条弧线（像两个人靠在一起）
    stroke = int(size * 0.062)
    r1 = size * 0.300
    d.arc([cx - r1 - size * 0.030, cy - r1, cx + r1 - size * 0.030, cy + r1],
          start=126, end=238, fill=(*VIOLET, 255), width=stroke)
    d.arc([cx - r1 + size * 0.030, cy - r1, cx + r1 + size * 0.030, cy + r1],
          start=-58, end=54, fill=(*MINT, 240), width=stroke)

    # 中央发声点 + 两层音圈
    for rr, alpha, wid in ((size * 0.145, 90, int(size * 0.012)),
                           (size * 0.205, 55, int(size * 0.010))):
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                  outline=(*WHITE, alpha), width=wid)
    dot = size * 0.088
    d.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=(*AMBER, 255))
    d.ellipse([cx - dot * 0.52, cy - dot * 0.52, cx + dot * 0.52, cy + dot * 0.52],
              fill=(*WHITE, 255))
    return compose(gradient(size, (18, 16, 36), (44, 30, 80)), art, size)


# ---------------------------------------------------------------- 输出
def save(art: Image.Image, name: str) -> None:
    big = art.resize((S, S), Image.LANCZOS)
    big.save(OUT / f"{name}.png")
    big.resize((512, 512), Image.LANCZOS).save(OUT / f"{name}_512.png")
    big.save(OUT / f"{name}.ico", sizes=[(256, 256), (128, 128), (64, 64),
                                         (48, 48), (32, 32), (16, 16)])
    print("saved", name)


def montage(images: list[tuple[str, Image.Image]]) -> None:
    pad, label_h = 40, 74
    cell = 460
    W = pad + (cell + pad) * len(images)
    H = pad * 2 + cell + label_h + 40
    sheet = Image.new("RGB", (W, H), (14, 11, 26))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 30)
        small = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 22)
    except Exception:
        font = small = ImageFont.load_default()
    for i, (title, im) in enumerate(images):
        x = pad + i * (cell + pad)
        thumb = im.resize((cell, cell), Image.LANCZOS)
        sheet.paste(thumb, (x, pad))
        d.text((x + 4, pad + cell + 14), title, font=font, fill=(238, 232, 255))
        d.text((x + 4, pad + cell + 52), "PrivateVoice_Ahee52", font=small,
               fill=(150, 138, 180))
    sheet.save(OUT / "concepts_1x3.png")

    # 小尺寸可读性检查
    row = Image.new("RGB", (pad + (96 + pad) * len(images), 200), (14, 11, 26))
    dr = ImageDraw.Draw(row)
    for i, (title, im) in enumerate(images):
        x = pad + i * (96 + pad)
        row.paste(im.resize((64, 64), Image.LANCZOS), (x, 56))
        row.paste(im.resize((32, 32), Image.LANCZOS), (x + 2, 140))
        dr.text((x, 24), chr(65 + i), font=font, fill=(238, 232, 255))
    row.save(OUT / "concepts_small.png")
    print("montage saved", sheet.size)


if __name__ == "__main__":
    a = concept_a()
    b = concept_b()
    c = concept_c()
    save(a, "concept_a_wave_smile")
    save(b, "concept_b_bubble")
    save(c, "concept_c_embrace")
    montage([
        ("A · 声波微笑", a),
        ("B · 对话气泡", b),
        ("C · 双弧拥抱", c),
    ])
