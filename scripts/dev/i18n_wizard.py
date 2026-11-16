"""把 wizard.py 的中文界面串翻成英文（tz 词表 + 包装）。"""
import pathlib
import re

W = pathlib.Path(r"C:\Local Repositories\PrivateVoice\app\ui\wizard.py")
D = pathlib.Path(r"C:\Local Repositories\PrivateVoice\app\i18n_tz.py")

MAP = {
    "新建角色 — PrivateVoice_Ahee52": "New character — PrivateVoice_Ahee52",
    "训练 / 重训角色 — PrivateVoice_Ahee52": "Train character — PrivateVoice_Ahee52",
    "下一步": "Next", "上一步": "Back", "取消": "Cancel", "完成": "Finish",
    "第 1 步 · 基本信息": "Step 1 · Basics",
    "给她一个名字、一张脸和一段人设。这些以后都能改。":
        "Give her a name, a face and a persona. Everything can be changed later.",
    "例如：小满": "e.g. Emily",
    "名字 *": "Name *",
    "未选择（可跳过，默认灰色头像）": "Not selected (optional, grey avatar)",
    "选择图片…": "Choose image…",
    "头像": "Avatar",
    "人设描述": "Persona",
    "她见到你时说的第一句话": "Her first line when she meets you",
    "开场白": "Greeting",
    "选择头像": "Choose avatar",
    "图片 (*.png *.jpg *.jpeg *.webp)": "Images (*.png *.jpg *.jpeg *.webp)",
    "第 2 步 · 上传或录制声音": "Step 2 · Upload or record audio",
    "45–90 秒单人说话，安静房间，麦克风离嘴 15–20 厘米。":
        "45–90 s of one person talking, quiet room, mic 15–20 cm away.",
    "把 wav / mp3 / m4a / flac 拖到这里，或点下面的按钮":
        "Drop a wav / mp3 / m4a / flac here, or use the button below",
    "选择文件…": "Choose file…",
    "开始录制": "Start recording",
    "电平 %v dBFS": "Level %v dBFS",
    "还没有选择音频": "No audio selected",
    "选择音频": "Choose audio",
    "音频 (*.wav *.mp3 *.m4a *.flac)": "Audio (*.wav *.mp3 *.m4a *.flac)",
    "停止并保存": "Stop and save",
    "第 3 步 · 质量检查": "Step 3 · Quality check",
    "不合格的项会给出具体原因和修改建议。":
        "Failing checks explain what to fix.",
    "项目": "Item", "结果": "Result", "说明": "Details",
    "重新检测": "Re-check",
    "等待检测…": "Waiting…",
    "检测中…": "Checking…",
    "通过": "Pass", "提示": "Warning", "跳过": "Skip",
    "全部通过": "All passed",
    "有提示项，可以继续": "Warnings only — you can continue",
    "有提示项，仍可继续（不影响训练流程）": "Warnings only — continue (training is unaffected)",
    "第 4 步 · 切分与打标": "Step 4 · Segment & label",
    "改了文本、标了丢弃的，点「保存修改」再继续。":
        "After editing text or marking clips as discarded, click Save changes.",
    "时长": "Duration",
    "文本（可编辑）": "Text (editable)",
    "标记": "Tag",
    "试听": "Preview",
    "保存修改": "Save changes",
    "试听选中": "Preview selected",
    "VAD 切分 + 识别中，第一次运行要下载模型，请稍候…":
        "Running VAD + ASR. First run downloads models — please wait…",
    "好": "Good", "可疑": "Suspicious", "丢弃": "Discard",
    "分布正常": "Distribution looks fine",
    "第 5 步 · 训练": "Step 5 · Training",
    "训练在独立子进程里跑，可以随时取消；取消会清理临时文件。":
        "Training runs in a separate process and can be cancelled; temp files are cleaned up.",
    "训练档位": "Training profile",
    "尚未开始": "Not started",
    "开始训练": "Start training",
    "取消训练": "Cancel training",
    "展开日志": "Show log",
    "收起日志": "Hide log",
    "无法训练": "Cannot train",
    "还没有切分好的数据集。": "No segmented dataset yet.",
    "启动中…": "Starting…",
    "准备数据…": "Preparing data…",
    "训练完成，正在生成试听…": "Training done — generating previews…",
    "训练完成": "Training complete",
    "点「确定」后软件会自动重启，让新角色以干净状态生效。":
        "The app restarts after you click OK so the new character loads cleanly.",
    "训练失败": "Training failed",
    "正在取消…": "Cancelling…",
    "第 6 步 · 试听": "Step 6 · Preview",
    "三句固定文本，不接 LLM。不满意可以「上一步」回去重训。":
        "Three fixed sentences, no LLM. If you are not happy, go back and retrain.",
    "▶ 播放": "▶ Play",
    "A/B 对比原声": "A/B against the reference",
    "角色还没训练完，无法试听。": "Training has not finished — preview unavailable.",
    "加载 TTS 引擎并合成，第一次约需 20–40 秒…":
        "Loading TTS and synthesizing; the first run takes 20–40 s…",
    "试听失败": "Preview failed",
    "没有音频": "No audio",
    "读取失败：": "Read failed: ",
    "好": "Good",
}

# 1) 词典追加
s = D.read_text(encoding="utf-8")
add = "\n".join(f"    {k!r}: {v!r}," for k, v in MAP.items() if k not in s)
if add:
    s = s.replace("\n}\n\n\ndef tz(", "\n" + add + "\n}\n\n\ndef tz(", 1)
    D.write_text(s, encoding="utf-8")

# 2) 包装 wizard.py 的纯字面量
w = W.read_text(encoding="utf-8")
if "from app.i18n_tz import tz" not in w:
    w = w.replace("from app import i18n", "from app import i18n\nfrom app.i18n_tz import tz", 1)
    if "from app.i18n_tz import tz" not in w:
        w = w.replace("from __future__ import annotations",
                      "from __future__ import annotations\n\nfrom app.i18n_tz import tz", 1)

LIT = r'"([^"\\\n]*[\u4e00-\u9fff][^"\\\n]*)"'
count = 0
for call in (r'QLabel\(', r'QPushButton\(', r'QGroupBox\(', r'QCheckBox\(',
             r'\.setText\(', r'\.setWindowTitle\(', r'\.setPlaceholderText\(',
             r'\.setToolTip\(', r'\.addItem\(', r'\.setTabText\('):
    pat = re.compile("(" + call + r'\s*)' + LIT)

    def repl(m):
        lit = m.group(2)
        if "{" in lit or "}" in lit:
            return m.group(0)
        return f'tz("{lit}")'

    w, n = pat.subn(repl, w)
    count += n

W.write_text(w, encoding="utf-8")
print("wizard wrapped:", count, "dict entries:", len(MAP))
