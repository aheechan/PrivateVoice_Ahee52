# Stage 1 结论

**1. TTFP（5/20/50 字，3090）** p50 355 / 394 / 414 ms；p95 716 / 500 / 818 ms。全流程 RTF 0.82–0.90，快于实时。

**2. 关核代价 0 ms，但测不出真值。** nvcc 12.8 与 MSVC 14.51（VS 2026）不兼容，A 组编译崩于 `cudafe++ 0xC0000005`，`bigvgan_kernel_mode` 恒为 `pytorch`。A/B 实为同一路径，差异只在加载时间（26.6s vs 7.2s，来自那次失败的编译）。准确说法是"核从未跑起来"，不是"关掉无影响"。

**3. 无 VS Build Tools：能出声。** B 组不触发任何编译，不依赖 nvcc/cl，三档全部正常产出音频，见 `results/sample.wav`。

**4. 峰值显存** 2.17–2.19 GB（allocated），2.32 GB（reserved）。

**5. 版本** GPT-SoVITS `08d627c3338173c3229286d8787060d6559fe0f8`；Aqua `eb8e0d59a9ca317b40d8182b04c12ba01a489f8d`（v0.3.0）；torch 2.11.0+cu128；CUDA 12.8。

**6. 与 README 不符** ① README 兼写"v3 (2025-04-01)"与"validated against 08d627c"，后者实为 2026-04-30，按后者执行；② `vcvars64.bat` 路径未加引号，E: 盘无 8.3 短名时 MSVC 环境导入失败，用 junction + 官方 `MSVC_VCVARS64_PATH` 绕过，未改模型代码；③ `ninja` 未声明依赖但编译内核必需，另装；④ `infer_stream` 返回 `(sr, chunk, text)`，骨架里"兼容两种返回"的写法会把 `text` 误当音频。
