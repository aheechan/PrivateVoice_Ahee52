"""轻量联网搜索（无需 API Key）。

当前实现：Bing 网页版 HTML 解析（本机实测可达）。
DuckDuckGo 在这台机器上连不通，保留为可选备用，不默认使用。

设计原则：
- 搜索结果只作为"参考资料"注入系统提示，附带来源 URL；
- 失败静默返回空列表，绝不因为网络问题打断对话；
- 由 `web.enabled` 开关控制，默认关闭。
"""
from __future__ import annotations

import html
import re
import urllib.parse
from typing import Any

import httpx

# 维基媒体要求 UA 带用途与联系方式，否则 403
_WIKI_UA = ("Ahee52PrivateVoice/1.0 (local desktop app; "
            "https://example.com/ahee52) httpx")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TAG_RE = re.compile(r"<[^>]+>")
_ITEM_RE = re.compile(r'<li class="b_algo".*?</li>', re.S)
_TITLE_RE = re.compile(r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.S)


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _unwrap(url: str) -> str:
    """Bing 的跳转链接（bing.com/ck/a?...&u=a1...）还原成真实 URL。"""
    if "bing.com/ck/a" in url:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        raw = (qs.get("u") or [""])[0]
        if raw.startswith("a1"):
            import base64

            try:
                pad = "=" * (-len(raw[2:]) % 4)
                return base64.urlsafe_b64decode(raw[2:] + pad).decode("utf-8", "ignore")
            except Exception:
                return url
    return url


def search_bing(query: str, max_results: int = 5,
                timeout_ms: int = 6000) -> list[dict[str, str]]:
    try:
        resp = httpx.get("https://www.bing.com/search",
                         params={"q": query, "setlang": "zh-Hans"},
                         headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9"},
                         timeout=timeout_ms / 1000.0, follow_redirects=True)
        if resp.status_code >= 400:
            return []
        out: list[dict[str, str]] = []
        for block in _ITEM_RE.findall(resp.text)[:max_results]:
            m = _TITLE_RE.search(block)
            if not m:
                continue
            url = _unwrap(m.group(1))
            title = _clean(m.group(2))
            sm = _SNIPPET_RE.search(block)
            snippet = _clean(sm.group(1)) if sm else ""
            if title and url.startswith("http"):
                out.append({"title": title, "url": url, "snippet": snippet[:220]})
        return out
    except Exception:
        return []


def search_wikipedia(query: str, lang: str = "zh",
                     max_results: int = 3, timeout_ms: int = 6000
                     ) -> list[dict[str, str]]:
    """维基百科摘要检索：历史/文化/艺术/百科类问题最稳，不需要 API Key。"""
    try:
        base = f"https://{lang}.wikipedia.org/w/api.php"
        resp = httpx.get(base, params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": max_results, "utf8": 1,
        }, headers={"User-Agent": _WIKI_UA}, timeout=timeout_ms / 1000.0,
            follow_redirects=True)
        if resp.status_code >= 400:
            return []
        pages = (resp.json().get("query", {}) or {}).get("search", []) or []
        out = []
        for page in pages[:max_results]:
            title = page.get("title", "")
            snippet = _clean(page.get("snippet", ""))
            if not title:
                continue
            url = f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(
                title.replace(" ", "_"))
            out.append({"title": f"{title}（维基百科）", "url": url,
                        "snippet": snippet[:240]})
        return out
    except Exception:
        return []


def search_baidu(query: str, max_results: int = 4,
                 timeout_ms: int = 6000) -> list[dict[str, str]]:
    """百度网页结果（HTML 解析）。Bing 不可用时兜底。"""
    try:
        resp = httpx.get("https://www.baidu.com/s", params={"wd": query, "rn": 10},
                         headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9"},
                         timeout=timeout_ms / 1000.0, follow_redirects=True)
        if resp.status_code >= 400:
            return []
        out: list[dict[str, str]] = []
        for m in re.finditer(
                r'<h3[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<h3|\Z)',
                resp.text, re.S):
            url, title_html, rest = m.group(1), m.group(2), m.group(3)
            title = _clean(title_html)
            snippet = ""
            sm = re.search(r'class="[^"]*(?:c-abstract|content-right)[^"]*"[^>]*>(.*?)</div>',
                           rest, re.S)
            if sm:
                snippet = _clean(sm.group(1))
            if title and url.startswith("http"):
                out.append({"title": title, "url": url, "snippet": snippet[:220]})
            if len(out) >= max_results:
                break
        return out
    except Exception:
        return []


_KNOWLEDGE_HINTS = ("历史", "文化", "艺术", "百科", "是什么", "为什么", "起源",
                    "哪一年", "谁", "介绍", "定义", "原理")


def looks_like_knowledge(query: str) -> bool:
    return any(k in query for k in _KNOWLEDGE_HINTS)


def search(query: str, config: Any = None) -> list[dict[str, str]]:
    cfg = getattr(config, "data", {}).get("web", {}) if config is not None else {}
    max_results = int(cfg.get("max_results", 4))
    timeout_ms = int(cfg.get("timeout_ms", 6000))
    # 知识类问题先查维基（稳定、无 key），再补网页结果；新闻类直接走 Bing
    results: list[dict[str, str]] = []
    if looks_like_knowledge(query):
        results = search_wikipedia(query, "zh", min(3, max_results), timeout_ms)
    if len(results) < 2:
        results += search_bing(query, max_results=max_results - len(results),
                               timeout_ms=timeout_ms)
    if len(results) < 2:
        # Bing 被限流时用百度兜底（两者都在国内可直连）
        results += search_baidu(query, max_results=max_results - len(results),
                                timeout_ms=timeout_ms)
    return results[:max_results]


def format_context(results: list[dict[str, str]]) -> str:
    if not results:
        return ""
    lines = ["以下是刚从网上查到的资料（可能不完整或过时，引用时要说明来自网络）："]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}：{r['snippet']}（来源 {r['url']}）")
    return "\n".join(lines)


def needs_search(text: str, keywords: list[str] | None = None) -> bool:
    """很粗的意图判断：只有像"查最新信息"的话才联网，避免每轮都发请求。"""
    if not text:
        return False
    kws = keywords or ["新闻", "最新", "今天", "现在", "几号", "价格", "股价", "汇率",
                       "天气", "比分", "比赛", "发布会", "上映", "开售", "版本",
                       "什么时候", "多少钱", "查一下", "搜一下", "搜索"]
    return any(k in text for k in kws)


__all__ = ["search", "search_bing", "search_baidu", "search_wikipedia",
           "format_context", "needs_search", "looks_like_knowledge"]
