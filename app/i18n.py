"""极简 i18n：中文 / English 两套文案。

设计目标：不引入 gettext/qm 文件，纯 Python 字典，切换后调用各窗口的
retranslate() 即可生效。
"""
from __future__ import annotations

LANG = "zh"
SUPPORTED = ("zh", "en")

STRINGS: dict[str, dict[str, str]] = {
    "zh": {
        "app.name": "PrivateVoice_Ahee52",
        "tray.tooltip": "PrivateVoice_Ahee52 — {state}",
        "tray.character": "当前角色",
        "tray.retry": "重试上一轮",
        "tray.new": "新建角色…",
        "tray.manage": "角色管理…",
        "tray.llm": "大模型设置…",
        "tray.settings": "设置…",
        "tray.diagnostics": "诊断…",
        "tray.quit": "退出",
        "tray.no_char": "（还没有可用角色）",
        "tray.characters_empty": "没有匹配的角色",
        "state.idle": "待机",
        "state.listening": "听着",
        "state.transcribing": "识别中",
        "state.thinking": "思考中",
        "state.speaking": "说话中",
        "state.training": "训练中",
        "main.title": "PrivateVoice_Ahee52",
        "main.talk": "按住说话",
        "main.talk_hint": "也可以按住 {hotkey} 说话",
        "main.send": "发送",
        "main.input_placeholder": "也可以打字：输入后回车（没麦克风时用这个）",
        "main.characters": "角色库",
        "main.language": "English",
        "main.language_tip": "切换到英文界面",
        "main.mood": "情绪",
        "main.warmup": "预热",
        "main.clean": "清理缓存",
        "main.settings": "设置",
        "main.llm": "大模型",
        "main.diagnostics": "诊断",
        "main.new_char": "新建角色",
        "main.starting": "正在预热模型，完成后自动恢复输入…",
        "firstrun.title": "Enjoy Your Private Time · 首次运行",
        "firstrun.subtitle": "享受你的私人独处时间。\n这台电脑是第一次运行，先做三项自检；全部通过就可以按住热键说话了。",
        "firstrun.language": "界面语言 / Language",
        "firstrun.gpu": "1 · 显卡",
        "firstrun.models": "2 · 模型文件",
        "firstrun.llm": "3 · 对话模型（OpenAI 兼容）",
        "firstrun.start": "开始使用",
        "firstrun.exit": "退出",
        "wiz.new_title": "新建角色 — PrivateVoice_Ahee52",
        "wiz.train_title": "训练 / 重训角色 — PrivateVoice_Ahee52",
        "wiz.next": "下一步",
        "wiz.back": "上一步",
        "wiz.cancel": "取消",
        "wiz.finish": "完成",
        "char.title": "角色库",
        "char.save": "保存",
        "char.revert": "还原本地修改",
        "char.refresh_hint": "F5 刷新列表",
    },
    "en": {
        "app.name": "PrivateVoice_Ahee52",
        "tray.tooltip": "PrivateVoice_Ahee52 — {state}",
        "tray.character": "Current character",
        "tray.retry": "Retry last turn",
        "tray.new": "New character…",
        "tray.manage": "Manage characters…",
        "tray.llm": "LLM settings…",
        "tray.settings": "Settings…",
        "tray.diagnostics": "Diagnostics…",
        "tray.quit": "Quit",
        "tray.no_char": "(no usable character yet)",
        "tray.characters_empty": "No matching characters",
        "state.idle": "Idle",
        "state.listening": "Listening",
        "state.transcribing": "Transcribing",
        "state.thinking": "Thinking",
        "state.speaking": "Speaking",
        "state.training": "Training",
        "main.title": "PrivateVoice_Ahee52",
        "main.talk": "Hold to talk",
        "main.talk_hint": "You can also hold {hotkey} to talk",
        "main.send": "Send",
        "main.input_placeholder": "Type here and press Enter (works without a mic)",
        "main.characters": "Characters",
        "main.language": "中文",
        "main.language_tip": "Switch to Chinese UI",
        "main.mood": "Mood",
        "main.warmup": "Warm up",
        "main.clean": "Clear cache",
        "main.settings": "Settings",
        "main.llm": "LLM",
        "main.diagnostics": "Diagnostics",
        "main.new_char": "New character",
        "main.starting": "Warming up models, input unlocks automatically…",
        "firstrun.title": "Enjoy Your Private Time · First run",
        "firstrun.subtitle": "Enjoy your private time.\nFirst launch: three quick checks. Once they pass, hold the hotkey and talk.",
        "firstrun.language": "Language / 界面语言",
        "firstrun.gpu": "1 · GPU",
        "firstrun.models": "2 · Model files",
        "firstrun.llm": "3 · Chat model (OpenAI compatible)",
        "firstrun.start": "Start",
        "firstrun.exit": "Exit",
        "wiz.new_title": "New character — PrivateVoice_Ahee52",
        "wiz.train_title": "Train character — PrivateVoice_Ahee52",
        "wiz.next": "Next",
        "wiz.back": "Back",
        "wiz.cancel": "Cancel",
        "wiz.finish": "Finish",
        "char.title": "Characters",
        "char.save": "Save",
        "char.revert": "Revert changes",
        "char.refresh_hint": "F5 to refresh",
    },
}


def set_lang(code: str) -> str:
    global LANG
    code = (code or "zh").lower()
    if code not in SUPPORTED:
        code = "zh"
    LANG = code
    return LANG


def get_lang() -> str:
    return LANG


def t(key: str, **kw) -> str:
    table = STRINGS.get(LANG) or STRINGS["zh"]
    text = table.get(key) or STRINGS["zh"].get(key) or key
    if kw:
        try:
            text = text.format(**kw)
        except Exception:
            pass
    return text


__all__ = ["set_lang", "get_lang", "t", "LANG", "SUPPORTED", "STRINGS"]
