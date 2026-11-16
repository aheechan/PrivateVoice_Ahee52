"""探针 2：确认 timestamp 结构 + 独立标点模型行为。"""
import json
import time

WAV = r"C:\Users\72157\OneDrive\Desktop\Grace_voice\grace voice nanpu plus orig 0327.wav"

from funasr import AutoModel  # noqa: E402

t0 = time.time()
asr = AutoModel(model="paraformer-zh", device="cuda", disable_update=True)
print("[p2] asr model load %.1fs" % (time.time() - t0), flush=True)

t0 = time.time()
res = asr.generate(input=WAV, batch_size_s=300, sentence_timestamp=True, return_raw_text=True)
print("[p2] asr %.1fs" % (time.time() - t0), flush=True)
item = res[0]
print("[p2] keys:", list(item.keys()))
ts = item.get("timestamp")
print("[p2] timestamp type:", type(ts).__name__, "len:", len(ts) if ts else 0)
if ts:
    print("[p2] ts[:5]:", ts[:5])
    print("[p2] ts[-3:]:", ts[-3:])
text = item.get("text", "")
print("[p2] text repr[:120]:", repr(text[:120]))
tokens = text.split()
print("[p2] tokens:", len(tokens), "first:", tokens[:8])

t0 = time.time()
punc = AutoModel(model="ct-punc", device="cuda", disable_update=True)
print("[p2] punc load %.1fs" % (time.time() - t0), flush=True)
t0 = time.time()
out = punc.generate(input=text.replace(" ", ""))
print("[p2] punc %.1fs" % (time.time() - t0), flush=True)
print("[p2] punc out:", json.dumps(out, ensure_ascii=False)[:800])
