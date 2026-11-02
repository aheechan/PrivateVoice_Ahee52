import json, urllib.request, urllib.error

UA = {"User-Agent": "voicelab/1.0"}
BASE = "https://hf-mirror.com/api/models/%s/tree/main?recursive=true"

def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return None, str(e).encode()

for repo in ["lj1995/GPT-SoVITS",
             "nvidia/bigvgan_v2_24khz_100band_256x",
             "lj1995/chinese-hubert-base",
             "TencentGameMate/chinese-hubert-base"]:
    code, body = get(BASE % repo)
    print("=" * 60)
    print(repo, "->", code)
    if code != 200:
        continue
    items = json.loads(body)
    for it in items:
        if it.get("type") != "file":
            continue
        size = it.get("size") or (it.get("lfs") or {}).get("size") or 0
        name = it["path"]
        if repo.startswith("lj1995/GPT-SoVITS"):
            keep = any(k in name.lower() for k in ("s1v3", "s2gv3", "hubert", "roberta", "s1bert", "s2g"))
            if not keep:
                continue
        print("   %-70s %10.1f MB" % (name, size / 1e6))
