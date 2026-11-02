"""开发用探针：验证 funasr + torch 2.11 能否加载 paraformer-zh 并给出词级时间戳。"""
import json
import sys
import time

WAV = r"C:\Users\72157\OneDrive\Desktop\Grace_voice\grace voice nanpu plus orig 0327.wav"

t0 = time.time()
from funasr import AutoModel  # noqa: E402

print("[probe] import ok %.1fs" % (time.time() - t0), flush=True)

t0 = time.time()
model = AutoModel(
    model="paraformer-zh",
    punc_model="ct-punc",
    device="cuda",
    disable_update=True,
)
print("[probe] model load %.1fs" % (time.time() - t0), flush=True)

t0 = time.time()
res = model.generate(
    input=WAV,
    batch_size_s=300,
    sentence_timestamp=True,
    return_raw_text=True,
)
print("[probe] asr %.1fs" % (time.time() - t0), flush=True)

item = res[0]
print("[probe] keys:", list(item.keys()))
print("[probe] text:", item.get("text"))
sents = item.get("sentence_info") or []
print("[probe] sentences:", len(sents))
if sents:
    print("[probe] first sentence keys:", list(sents[0].keys()))
    print(json.dumps(sents[:2], ensure_ascii=False)[:1500])

import torch  # noqa: E402

print("[probe] torch", torch.__version__, "cuda", torch.version.cuda)
print("[probe] vram %.2f GB" % (torch.cuda.max_memory_allocated() / 1024 ** 3))
