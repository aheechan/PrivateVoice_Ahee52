"""文本归一化：数字 / 日期 / 百分比 / 英文混合。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.label import normalize_text  # noqa: E402


def test_year_digit_by_digit():
    assert normalize_text("2026年我搬到了杭州") == "二零二六年我搬到了杭州"


def test_date_spoken():
    assert normalize_text("11月1日出发") == "十一月一日出发"
    assert normalize_text("2026年11月1日") == "二零二六年十一月一日"


def test_percent():
    assert normalize_text("涨了30%") == "涨了百分之三十"
    assert normalize_text("只有3.5%") == "只有百分之三点五"


def test_plain_number():
    assert normalize_text("我买了 3 个苹果") == "我买了三个苹果"
    assert normalize_text("一共128块钱") == "一共一百二十八块钱"


def test_english_kept():
    assert normalize_text("用 GPT4 试试") == "用 GPT4 试试"
    assert normalize_text("打开 VS Code 就行") == "打开 VS Code 就行"


def test_units():
    assert normalize_text("今天 30℃") == "今天三十摄氏度"


def test_fillers_and_punct_kept():
    assert normalize_text("嗯，那个……我看看。") == "嗯，那个……我看看。"


def test_empty():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
