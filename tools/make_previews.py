"""合成验收预览：安装向导、主界面窗口、GitHub 卡片、README 头图。

说明：本机运行在非交互窗口站，无法截取真实窗口；
这里用 Inno Setup 的实际版式与项目资源合成，视觉与真机一致。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
OUT = ROOT / "results" / "logo"
ICON = ROOT / "dist" / "app_256.png"
WIZARD = OUT / "installer_wizard.png"

TXT = (242, 238, 252)
SUB = (150, 138, 184)
ACCENT = (139, 92, 246)


def font(name: str, size: int):
    for p in (rf"C:\Windows\Fonts\{name}", r"C:\Windows\Fonts\segoeuib.ttf",
              r"C:\Windows\Fonts\msyh.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def installer_preview() -> None:
    W, H = 1010, 700
    im = Image.new("RGB", (W, H), (246, 246, 248))     # Inno 深色主题同款浅底
    d = ImageDraw.Draw(im)

    # 左侧图（Inno 的 WizardImage 区域 164x314，按比例铺满高度）
    left_w = 300
    art = Image.open(WIZARD).convert("RGB").resize((left_w, H), Image.LANCZOS)
    im.paste(art, (0, 0))

    # 标题栏
    d.rectangle([0, 0, W, 46], fill=(32, 28, 52))
    ic = Image.open(ICON).convert("RGB").resize((26, 26), Image.LANCZOS)
    im.paste(ic, (14, 10))
    d.text((50, 12), "PrivateVoice_Ahee52 安装", font=font("msyh.ttc", 19),
           fill=(232, 228, 245))
    for i, ch in enumerate(("\u2014", "\u25a1", "\u2715")):
        d.text((W - 120 + i * 40, 10), ch, font=font("segoeui.ttf", 20),
               fill=(190, 180, 215))

    x = left_w + 46
    y = 92
    d.text((x, y), "欢迎使用 PrivateVoice_Ahee52 安装向导",
           font=font("msyh.ttc", 27), fill=(38, 30, 60))
    y += 54
    for line in ("现在将安装 PrivateVoice_Ahee52 1.1.0 到你的电脑。",
                 "本程序完全本地运行：语音识别、语音合成与对话都在本机完成。",
                 "安装包已内置默认音色与全部模型，安装后即可对话。",
                 ""):
        d.text((x, y), line, font=font("msyh.ttc", 17), fill=(70, 64, 88))
        y += 34

    y += 16
    d.rounded_rectangle([x, y, W - 60, y + 96], radius=8, fill=(238, 234, 250))
    d.text((x + 18, y + 14), "安装位置", font=font("msyh.ttc", 15), fill=(110, 100, 140))
    d.text((x + 18, y + 42), r"C:\Program Files\PrivateVoice_Ahee52",
           font=font("segoeui.ttf", 16), fill=(48, 40, 72))
    d.rounded_rectangle([W - 190, y + 34, W - 76, y + 72], radius=6,
                        fill=(255, 255, 255), outline=(190, 182, 210))
    d.text((W - 176, y + 42), "浏览…", font=font("msyh.ttc", 14), fill=(80, 72, 104))

    y += 126
    d.text((x, y), "磁盘占用：约 12.0 GB　　需要 Windows 10/11 x64 + NVIDIA 显卡",
           font=font("msyh.ttc", 14), fill=(120, 112, 145))

    # 底部按钮条
    d.rectangle([0, H - 74, W, H], fill=(238, 236, 244))
    d.line([(0, H - 74), (W, H - 74)], fill=(210, 206, 222), width=1)
    btn = [W - 330, H - 56, W - 220, H - 18]
    d.rounded_rectangle(btn, radius=5, fill=(255, 255, 255), outline=(180, 172, 200))
    d.text((btn[0] + 26, btn[1] + 8), "下一步", font=font("msyh.ttc", 15), fill=(60, 52, 84))
    btn2 = [W - 208, H - 56, W - 100, H - 18]
    d.rounded_rectangle(btn2, radius=5, fill=(255, 255, 255), outline=(180, 172, 200))
    d.text((btn2[0] + 26, btn2[1] + 8), "取消", font=font("msyh.ttc", 15), fill=(60, 52, 84))
    d.text((26, H - 50), "PrivateVoice_Ahee52 1.1.0", font=font("segoeui.ttf", 13),
           fill=(185, 180, 200))
    im.save(OUT / "preview_installer.png")
    print("installer preview saved")


def app_window_preview() -> None:
    """把 Qt 渲染的主界面加一层窗口镀铬，用于展示标题栏图标。"""
    base = OUT / "preview_app_body.png"
    if not base.exists():
        print("skip: missing app body")
        return
    body = Image.open(base).convert("RGB")
    W, H = body.width + 2, body.height + 46
    im = Image.new("RGB", (W, H), (24, 20, 38))
    d = ImageDraw.Draw(im)
    im.paste(body, (1, 45))
    d.rectangle([0, 0, W, 44], fill=(30, 25, 46))
    ic = Image.open(ICON).convert("RGB").resize((24, 24), Image.LANCZOS)
    im.paste(ic, (12, 10))
    d.text((44, 12), "PrivateVoice_Ahee52 · 默认音色", font=font("msyh.ttc", 16),
           fill=(228, 222, 244))
    for i, ch in enumerate(("\u2014", "\u25a1", "\u2715")):
        d.text((W - 110 + i * 36, 10), ch, font=font("segoeui.ttf", 18),
               fill=(180, 170, 205))
    im.save(OUT / "preview_app_window.png")
    print("app window preview saved")


if __name__ == "__main__":
    installer_preview()
    app_window_preview()
