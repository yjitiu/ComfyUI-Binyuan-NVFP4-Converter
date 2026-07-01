# 💎 ComfyUI Binyuan · Universal Quant Converter (NVFP4 / FP8 / GGUF)

> A ComfyUI custom node that **quantizes diffusion models** (safetensors) to
> **NVFP4 / FP8 (e4m3fn) / GGUF** in place — with a per-architecture blacklist
> that protects structure-sensitive layers so the quantized model actually loads
> and renders correctly. `Auto (Universal)` mode works on any architecture
> (T2I/I2I/T2V/I2V, including brand-new models) without per-model tuning.

把 ComfyUI 里的扩散模型原地量化成 NVFP4 / FP8 / GGUF，自带按模型类型精确配置的"黑名单"保护敏感层，量化后能正常加载出图。

---

## 中文说明

### 这是什么
节点 `binyuan Universal Quant Converter (NVFP4/FP8/GGUF)`，分类 `binyuan/Advanced`。
读取 ComfyUI 的 `checkpoints / unet / diffusion_models` 里的源模型，按所选格式 + 模型类型量化，输出新的 safetensors(NVFP4/FP8) 或 .gguf 文件。自动承袭原模型全部元数据，并写入 `comfy_quant` 标记，让 ComfyUI 正确识别量化层。

### 怎么用
1. 依赖：
   - NVFP4/FP8 需要 `pip install comfy-kitchen`
   - GGUF 需要 `pip install gguf`（通常随 ComfyUI-GGUF 安装）
2. 放到 `ComfyUI/custom_nodes/ComfyUI_binyuan_NVFP4_Converter/`，重启 ComfyUI。
3. 右键 → `binyuan/Advanced` → `binyuan Universal Quant Converter`。
4. 选 `input_model`（下拉自动列出 checkpoints/unet/diffusion_models 里的模型）。
5. 选 `output_format`：`NVFP4` / `FP8 (e4m3fn)` / `GGUF`。
6. 选 `model_type`（决定黑名单策略）：
   - `Auto (Universal)`：通吃任意架构，无需逐模型调参（推荐首选）。
   - 或选具体模型：Flux.1/Flux.2-Klein、Krea2-Turbo、Qwen-Image、Z-Image、Wan2.2、LTX、ERNIE、Ideogram、LENS、PID-Decoder、Boogu-Image 等。
7. （GGUF）选 `gguf_qtype`：Q8_0/Q5_0/Q4_0/BF16/F16 等。
8. 输出目录：`save_to_source_dir=true` 存到源模型同目录；或在 `custom_output_dir` 手填目录。填 `output_filename`。
9. 运行，`status` 端口返回结果信息。

### 关键参数
| 参数 | 说明 |
|---|---|
| input_model | 源模型（自动扫描 checkpoints/unet/diffusion_models） |
| output_format | NVFP4 / FP8 (e4m3fn) / GGUF |
| model_type | 黑名单策略；Auto 通吃，或选具体架构 |
| gguf_qtype | 仅 GGUF 生效（Q8_0/Q5_0/Q4_0/Q6_K/Q5_K/Q4_K/Q3_K/Q2_K/IQ4_*/BF16/F16） |
| full_precision_mm | NVFP4/FP8 推理模式：开=反量化到全精度做矩阵乘（任何 GPU 能跑，最稳，不省显存）；关=原生量化矩阵乘（真省显存/加速，需 Blackwell+cu130+comfy_kitchen） |
| device | cuda / cpu |
| save_to_source_dir / custom_output_dir / output_filename | 输出位置与文件名 |

### 三种格式怎么选
- **NVFP4**（≈0.56 字节/参数）：体积最小，Blackwell 原生支持可省显存加速；其它卡用 `full_precision_mm=true` 也能跑（仅磁盘变小）。
- **FP8 (e4m3fn)**（1 字节/参数）：兼容性好，多数新卡支持。
- **GGUF**：llama.cpp 系量化，Q4_0/Q5_0 等可压到 0.56–0.69 字节/参数；注意纯 Python gguf 库通常只能量化到 Q8_0/Q5_0/Q4_0/BF16/F16，选 K 系列/IQ 系列会自动改用同尺寸可用类型并提示。

### 智能防护
- 自动检测源模型是否已量化（FP8/NVFP4）：同格式二次量化体积不会变小，直接拦截并给建议。
- FP8 源转 GGUF 时，若选的 qtype ≥1 字节/参数（会变大），拦截并建议 Q5_0/Q4_0。
- 量化前用 `weight_scale` 反量化 FP8 源到真实值，避免出噪点。
- 结构敏感层（embed/first/last/norm/modulation/pad_token 等）强制保 BF16/F32，防止"能转不能加载"。
- Windows 大文件写入用临时目录+原子改名，规避杀软/文件夹监视器导致的"磁盘空间不足"误报。

### 输出
- `status` (STRING)：转换结果/错误信息。
<img width="1052" height="884" alt="屏幕截图 2026-07-01 115521" src="https://github.com/user-attachments/assets/bbe8906f-5a71-4c33-a9b4-c46272af2784" />
<img width="1029" height="740" alt="屏幕截图 2026-07-01 115516" src="https://github.com/user-attachments/assets/4f6fd4aa-0117-4a07-97bd-9c7f627fabf5" />
<img width="1012" height="692" alt="屏幕截图 2026-07-01 115510" src="https://github.com/user-attachments/assets/03a99f75-543f-4851-ad17-01179bbbcb9e" />

---

## English

### What it is
Node `binyuan Universal Quant Converter (NVFP4/FP8/GGUF)` under category
`binyuan/Advanced`. Reads a source model from ComfyUI's checkpoints/unet/
diffusion_models, quantizes it to NVFP4 / FP8 / GGUF with a per-architecture
blacklist, and writes a new file (safetensors or .gguf). Preserves original
metadata and writes `comfy_quant` markers so ComfyUI loads quantized layers
correctly.

### How to use
1. Deps: `pip install comfy-kitchen` (NVFP4/FP8) and/or `pip install gguf` (GGUF).
2. Drop into `ComfyUI/custom_nodes/ComfyUI_binyuan_NVFP4_Converter/`, restart.
3. Right-click → `binyuan/Advanced` → add the node.
4. Pick `input_model`.
5. Pick `output_format`: NVFP4 / FP8 (e4m3fn) / GGUF.
6. Pick `model_type`: `Auto (Universal)` (works on any architecture, recommended)
   or a specific model (Flux/Krea2/Qwen-Image/Z-Image/Wan2.2/LTX/ERNIE/Ideogram/
   LENS/PID/Boogu…).
7. (GGUF) pick `gguf_qtype`.
8. Output: `save_to_source_dir=true` for same dir, or fill `custom_output_dir`
   and `output_filename`.
9. Run; read `status`.

### Format chooser
- **NVFP4** (~0.56 B/param): smallest; native speedup on Blackwell, runs
  anywhere with `full_precision_mm=true` (disk-only savings).
- **FP8 e4m3fn** (1 B/param): widely compatible.
- **GGUF**: Q4_0/Q5_0 ≈ 0.56–0.69 B/param. Pure-Python gguf usually quantizes
  only to Q8_0/Q5_0/Q4_0/BF16/F16; K/IQ types auto-fall back to nearest size.

### Smart guards
- Detects already-quantized sources (FP8/NVFP4) and blocks same-format re-quant
  (no size gain).
- Blocks FP8→GGUF when chosen qtype ≥1 B/param (would grow); suggests Q5_0/Q4_0.
- De-quantizes FP8 source via `weight_scale` before re-quantizing (avoids noise).
- Forces BF16/F32 on structure-sensitive layers (embed/first/last/norm/modulation/
  pad_token…) so the result loads.
- Large-file write uses a temp dir + atomic rename to dodge antivirus false
  "disk full" errors on Windows.

### Output
`status` (STRING): result / error message.

## Files
- `__init__.py` — registration
- `binyuan_nvfp4_converter.py` — node `BinyuanNVFP4Converter`
- `README.md` / `LICENSE` (MIT)

## Install
- ComfyUI Manager: search `Binyuan NVFP4 Converter`.
- Manual: `git clone https://github.com/yjitiu/ComfyUI-Binyuan-NVFP4-Converter.git ComfyUI_binyuan_NVFP4_Converter`
