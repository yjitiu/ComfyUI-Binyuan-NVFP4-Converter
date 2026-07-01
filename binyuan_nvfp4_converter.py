import os
import json
import torch
import gc  # 垃圾回收
import importlib.util
import folder_paths
import safetensors.torch
import comfy.utils
import safetensors
from collections import OrderedDict

# 检查 comfy-kitchen 是否可用（NVFP4 / FP8 量化需要）
try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
    HAS_KITCHEN = True
except ImportError:
    HAS_KITCHEN = False

# --------------------------------------------------------------------------- #
# 通用黑名单（Auto 模式用，架构无关）
# --------------------------------------------------------------------------- #
# 设计原则：把「量化必崩 / 量化会让 ComfyUI 识别错模型」的结构敏感层全部保护到
# BF16，其余 2D 权重（attn 的 to_q/k/v、FFN 的 proj 等，占体积 95%+）走量化。
# 这样任意架构（文生图 / 图生图 / 图生视频 / 文生视频，含未来新模型）都能直接量化，
# 不必为每个新模型手工调黑名单。已避开会误伤的子串（不用裸 "mod"→会命中 "model"，
# 不用 "gate"→会命中 "aggregate"，不用 "rms"→会命中 "transforms"，不用 "table"→会
# 命中 "stable"）。
UNIVERSAL_BLACKLIST = [
    # 输入嵌入 / 条件投影：量化会破坏 ComfyUI 对条件/无条件模型的识别，且精度敏感。
    # ⚠️ first / last（Krea2 等的输入/输出投影层）绝对不能量化——ComfyUI 靠它们的
    # weight.shape 反推模型 channels 配置；NVFP4 会把最后一维压一半（64→32），导致
    # channels 被误判、模型按错误维度实例化、加载时报 last.linear.bias 形状不匹配。
    "embed", "patch_emb", "pos_emb", "first", "last",
    "img_in", "txt_in", "time_in", "vector_in", "guidance_in",
    "time_text", "time_projection", "time_embedding", "text_embedding",
    "class_embedding", "img_in_proj",
    # 裸 pad token Parameter（Z-Image 的 x_pad_token / cap_pad_token 等=[1,dim]）：
    # 同 first/last，不走 GGMLLayer，量化后以打包形状 [1, dim*9/16] 直接 load_state_dict
    # → 形状不匹配 + 非 float dtype 仍带 requires_grad → 加载报错。必须保 F32。
    "pad_token",
    # 归一化 / 调制：极敏感，量化必崩
    "norm", "bias", "scale", "modulation", "adaLN", "adaln",
    "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt",
    "img_mod", "txt_mod", "scale_shift",
    # 输出层 / 最终投影
    "final_layer", "proj_out", "norm_out", "head", "out_proj",
    # 统一模型里的非-DiT 子模块（VAE / 跨模态连接器 / 分词 / MoE 门控 / 规划器等）
    "vae.", "vocoder.", "connector", "patchify", "tokenizer", "moe_gate",
    "noise_refiner", "context_refiner", "projection", "adaln_single", "mllm_planner",
    "pixel_dit_adapter",
]

# 构造 ComfyUI 原生量化层配置键 comfy_quant（uint8 编码的 JSON）。
# ops.py 的 _load_quantized_module 靠每个权重旁的 {prefix}.comfy_quant 判断该层是否量化、
# 用哪种格式。没有这个键，ComfyUI 会把量化后的 uint8/fp8 权重当普通权重加载 → 形状不匹配
# →「出不了图」。这是本节点以前「能量化但加载报错」的根因。
def _comfy_quant_tensor(fmt, full_precision_mm):
    conf = {"format": fmt}
    if full_precision_mm:
        # True：推理时反量化到全精度做矩阵乘（任何 GPU 都能跑，但无省显存收益，仅磁盘变小）。
        # False：用原生量化矩阵乘（真省显存/加速，但需 Blackwell + cu130 + comfy_kitchen CUDA 后端）。
        conf["full_precision_matrix_mult"] = True
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)

# GGUF 量化类型 → gguf 库枚举（懒加载，避免未装 gguf 时导入失败）。
# 注意：gguf 库的 GGMLQuantizationType 没有 Q4_K_M / Q5_K_M 这类「子型」枚举——
# K 系列只有 Q4_K / Q5_K / Q3_K / Q2_K，M/S 是 llama.cpp 转换期的分组策略，不是
# 逐张量 qtype。所以这里统一映射到 K 型，并用 getattr 兜底，跨 gguf 版本不崩。
def _gguf_qtype_map():
    import gguf
    Q = gguf.GGMLQuantizationType
    g = lambda name, fb="Q8_0": getattr(Q, name, getattr(Q, fb))
    return {
        "Q8_0": g("Q8_0"),
        "Q6_K": g("Q6_K"),
        "Q5_K": g("Q5_K"),
        "Q5_0": g("Q5_0"),
        "Q4_K": g("Q4_K"),
        "Q4_0": g("Q4_0"),
        "Q3_K": g("Q3_K"),
        "Q2_K": g("Q2_K"),
        "IQ4_NL": g("IQ4_NL"),
        "IQ4_XS": g("IQ4_XS"),
        "IQ3_S": g("IQ3_S"),
        "BF16": g("BF16"),
        "F16": g("F16"),
    }

# gguf 纯 Python 库只实现了 Q8_0/Q5_0/Q4_0/BF16/F16 的「量化」；K 系列 / IQ 系列
# 仅有反量化、没量化。用户选了不支持的 qtype 时，按「目标字节成本最接近」自动改用
# 支持的类型，让节点直接产出可用文件而不是硬报错。映射按 bpp 就近：
#   Q4_K(≈0.56)→Q4_0(≈0.56) 同尺寸；Q5_K(≈0.69)→Q5_0(≈0.69) 同尺寸；
#   Q6_K 是高精度型 → 取更高精度的 Q8_0(≈0.85)；其余更小的 K/IQ → Q4_0（最小可用）。
_GGUF_QTYPE_FALLBACK = {
    "Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q4_0", "Q5_K": "Q5_0", "Q6_K": "Q8_0",
    "IQ4_NL": "Q4_0", "IQ4_XS": "Q4_0", "IQ3_S": "Q4_0",
}

def _gguf_supported_qtypes():
    """实测本机 gguf 库真能量化的 qtype 名集合（跨版本自适应，升级 gguf 后自动认新类型）。"""
    import gguf
    import numpy as _np
    Q = gguf.GGMLQuantizationType
    probe = _np.zeros((2, 256), dtype=_np.float32)
    ok = []
    for name in ["Q8_0", "Q5_0", "Q4_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K",
                 "IQ4_NL", "IQ4_XS", "IQ3_S", "BF16", "F16"]:
        q = getattr(Q, name, None)
        if q is None:
            continue
        try:
            gguf.quants.quantize(probe, q); ok.append(name)
        except Exception:
            pass
    return set(ok)

# 自动扫描并合并所有的模型路径到下拉列表
def get_all_models():
    choices = []
    for folder_type in ("checkpoints", "unet", "diffusion_models"):
        try:
            for f in folder_paths.get_filename_list(folder_type):
                choices.append(f"{folder_type}/{f}")
        except Exception:
            pass

    if not choices:
        choices = ["没有检测到模型（请确认模型已放入 checkpoints/unet/diffusion_models 文件夹内）"]
    return choices

# --------------------------------------------------------------------------- #
# GGUF：复用 ComfyUI-GGUF 的架构检测 / 前缀剥离（保持对新模型的支持）
# --------------------------------------------------------------------------- #
class _DefaultArch:
    """ComfyUI-GGUF 不可用 / detect_arch 失败时的兜底架构：不做任何特殊处理，通吃但无 hiprec 保护。

    注意：arch 字符串会在 _convert_to_gguf 里被 _safe_gguf_arch_str() 覆盖成两加载器都
    接受的通用值（'flux' 等）。这里的 'diffuser' 只是占位——pig.py 和 ComfyUI-GGUF 都不认
    'diffuser'，留着它会导致加载报 'Unknown architecture: diffuser'。
    """
    arch = "diffuser"
    shape_fix = False
    keys_hiprec = []
    keys_ignore = []
    def handle_nd_tensor(self, key, data):
        print(f"[binyuan NVFP4 Converter] 跳过 >4D 张量（兜底架构）: {key} {getattr(data,'shape',None)}")

_gguf_helpers_cache = None
def _load_gguf_helpers():
    """加载 ComfyUI-GGUF 的 tools/convert.py，复用其 arch 检测；不可用则返回 None。"""
    global _gguf_helpers_cache
    if _gguf_helpers_cache is not None:
        return _gguf_helpers_cache if _gguf_helpers_cache is not False else None
    try:
        for cn_dir in folder_paths.get_folder_paths("custom_nodes"):
            p = os.path.join(cn_dir, "ComfyUI-GGUF", "tools", "convert.py")
            if os.path.isfile(p):
                spec = importlib.util.spec_from_file_location("binyuan_gguf_convert_helper", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _gguf_helpers_cache = mod
                return mod
    except Exception as e:
        print(f"[binyuan NVFP4 Converter] 未加载 ComfyUI-GGUF 助手（将用兜底架构）: {e}")
    _gguf_helpers_cache = False
    return None

# detect_arch 失败时（Krea2 等两工具都不认识的模型）用的兜底 arch 候选。
# 这些 arch 同时在 pig.py 的 PIG_ARCH_LIST 和 ComfyUI-GGUF 的 IMG_ARCH_LIST 里，
# 任一加载器都能通过校验。加载时 arch 仅作校验令牌，真实模型类型由 ComfyUI 按
# state_dict 键名检测，所以非 Flux 模型写成 'flux' 也能正确加载。
_GGUF_SAFE_FALLBACK_ARCHES = ["flux", "sd3", "sdxl", "sd1", "ltxv", "wan", "lumina2", "qwen_image"]

def _safe_gguf_arch_str():
    """返回一个本机已安装 GGUF 加载器都接受的通用 arch。

    旧兜底值 'diffuser' 在 pig.py 和 ComfyUI-GGUF 两个加载器的接受列表里都不存在，
    加载时报 'Unknown architecture: diffuser'。这里读取 pig.py 的 version.json 取其
    PIG_ARCH_LIST，从候选里挑一个它接受的（'flux' 等）；读不到就直接用 'flux'。
    """
    try:
        for cn_dir in folder_paths.get_folder_paths("custom_nodes"):
            vj = os.path.join(cn_dir, "gguf", "version.json")
            if os.path.isfile(vj):
                with open(vj, encoding="utf-8") as f:
                    data = json.load(f)
                pig = set(data[0].get("PIG_ARCH_LIST", [])) if data else set()
                for a in _GGUF_SAFE_FALLBACK_ARCHES:
                    if a in pig:
                        return a
                # 候选都不在，退而取 pig 列表里第一个非文本 arch
                return next(iter(pig), "flux")
    except Exception as e:
        print(f"[binyuan NVFP4 Converter] 读取 pig.py 架构列表失败，兜底用 'flux': {e}")
    return "flux"


def _convert_to_gguf(input_path, output_path, qtype_name):
    """把 safetensors 模型转成指定量化类型的 GGUF 文件。"""
    import gguf

    sd = safetensors.torch.load_file(input_path)

    helpers = _load_gguf_helpers()
    _used_fallback_arch = False
    if helpers is not None:
        sd = helpers.strip_prefix(sd)
        try:
            arch = helpers.detect_arch(sd)
        except Exception as e:
            # Krea2 等两工具都不认识的模型：detect_arch 抛错。旧代码用 _DefaultArch.arch
            # = "diffuser"，但 pig.py / ComfyUI-GGUF 两个加载器都不认 "diffuser"，加载时
            # 报 'Unknown architecture: diffuser'。这里改用两加载器都接受的通用 arch。
            print(f"[binyuan NVFP4 Converter] 架构检测失败（{e}），改用通用 arch。")
            arch = _DefaultArch()
            arch.arch = _safe_gguf_arch_str()
            _used_fallback_arch = True
    else:
        arch = _DefaultArch()
        arch.arch = _safe_gguf_arch_str()
        _used_fallback_arch = True
    if _used_fallback_arch:
        print(f"[binyuan NVFP4 Converter] ⚠️ 未识别模型，GGUF arch 写为 '{arch.arch}'"
              f"（仅作加载校验令牌，真实模型类型由 ComfyUI 按键名检测）。"
              f"若加载器仍报 Unknown architecture，请反馈此 arch 名。")

    QT = gguf.GGMLQuantizationType

    # 本机 gguf 库很多 qtype（所有 K 系列、IQ 系列）只实现了「反量化」、没实现「量化」。
    # 旧逻辑是探针失败就硬报错——但用户多半就是想要那个尺寸（如 Q4_K≈0.56）。这里改为：
    # 先实测支持集，选了不支持的就按 _GGUF_QTYPE_FALLBACK 自动改用同尺寸可用类型，打印
    # 提示并继续，让节点直接产出可用文件。探针仍保留作兜底（替换后仍不可量化的极端情况）。
    _supported = _gguf_supported_qtypes()
    _subst_notice = ""
    if qtype_name not in _supported:
        _alt = _GGUF_QTYPE_FALLBACK.get(qtype_name, "Q8_0")
        if _alt not in _supported:
            _alt = next((x for x in ("Q8_0", "Q5_0", "Q4_0", "BF16", "F16") if x in _supported), "Q8_0")
        _subst_notice = (f"⚠️ 本机 gguf 库不支持量化到 {qtype_name}（K/IQ 系列仅有反量化），"
                         f"已自动改用 {_alt}——目标体积相近、精度可能略低。"
                         f"本机可用: {', '.join(sorted(_supported))}")
        print(f"[binyuan NVFP4 Converter] {_subst_notice}")
        qtype_name = _alt

    target_q = _gguf_qtype_map().get(qtype_name, QT.Q8_0)

    # 兜底探针：替换后的 qtype 仍不可量化（理论不该发生）时，列可用项报错，不再静默写 F16。
    import numpy as _np
    _probe = _np.zeros((2, 256), dtype=_np.float32)
    try:
        gguf.quants.quantize(_probe, target_q)
    except Exception as _e:
        return (f"错误: 当前 gguf 库版本不支持把权重量化成 {qtype_name}"
                f"（{type(_e).__name__}），且无可用替代。\n"
                f"  本机 gguf 实际可量化的类型: {', '.join(sorted(_supported)) or '（无）'}\n"
                "  请改选 Q8_0 / Q5_0 / Q4_0 / BF16 / F16。",)

    QUANTIZATION_THRESHOLD = 1024
    REARRANGE_THRESHOLD = 512
    MAX_TENSOR_DIMS = 4

    writer = gguf.GGUFWriter(path=None, arch=getattr(arch, "arch", "diffuser"))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    # 文件类型：按主精度给 BF16/F16（loader 按每个张量的 raw_dtype 读取，file_type 仅信息性）
    try:
        dtypes = [x.dtype for x in sd.values()]
        main_dtype = max(set(dtypes), key=dtypes.count) if dtypes else torch.bfloat16
        if main_dtype == torch.bfloat16:
            writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
        else:
            writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    except Exception:
        pass

    pbar = comfy.utils.ProgressBar(len(sd))
    print(f"[binyuan NVFP4 Converter] 📦 GGUF 转换（量化类型 {qtype_name}，架构 {getattr(arch,'arch','diffuser')}）")

    keys_hiprec = getattr(arch, "keys_hiprec", []) or []
    keys_ignore = getattr(arch, "keys_ignore", []) or []
    shape_fix = getattr(arch, "shape_fix", False)

    fallback_count = 0
    fallback_bytes = 0
    for i, (key, data) in enumerate(sd.items()):
        pbar.update_absolute(i + 1)

        if any(x in key for x in keys_ignore):
            continue

        # 跳过 comfy 量化产物（.comfy_quant / .weight_scale / .weight_scale_2 / .input_scale）——
        # 它们是上次量化的元数据/标量，不是真实权重。写进 GGUF 既无意义，又会被当 F16/F32
        # 膨胀体积，还会让架构检测误判。源模型的量化层靠 .weight 自身的 raw_dtype 重新量化即可。
        if (key.endswith(".comfy_quant") or key.endswith(".weight_scale")
                or key.endswith(".weight_scale_2") or key.endswith(".input_scale")):
            continue

        old_dtype = data.dtype
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        elif data.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
            # FP8 源：必须反量化到真实 float32 再走目标 qtype 量化。社区/comfy FP8 把权重存为
            # float8 = real/scale，配套 <base>.weight_scale 标量。直接 cast 不乘 scale 得到的是
            # fp8 自然范围值（abs.max≈448），与真实权重（abs.max≈0.3）差上千倍，量化后出图全噪点。
            _sk = key + "_scale"  # base.weight -> base.weight_scale（与 comfy 写法一致）
            if _sk in sd:
                data = (data.to(torch.float32) * sd[_sk].to(torch.float32)).numpy()
            else:
                data = data.to(torch.float32).numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        n_params = 1
        for d in data_shape:
            n_params *= d

        if n_dims > MAX_TENSOR_DIMS:
            try:
                arch.handle_nd_tensor(key, data)
            except Exception as e:
                print(f"[binyuan NVFP4 Converter] 跳过 >4D 张量: {key} ({e})")
            continue

        # 量化类型选择
        # FP8 源（float8_e4m3fn/e5m2）也走目标 qtype：已先反量化到 float32，可直接量化成
        # Q8_0/Q4_K 等，体积才能真正变小。否则会被 else 强制存成 F16（翻倍且不量化）。
        _quantizable_src = old_dtype in (torch.float32, torch.bfloat16,
                                         getattr(torch, "float8_e4m3fn", None),
                                         getattr(torch, "float8_e5m2", None))
        if _quantizable_src:
            if n_dims == 1:
                data_qtype = QT.F32
            elif n_params <= QUANTIZATION_THRESHOLD:
                data_qtype = QT.F32
            elif any(x in key for x in keys_hiprec):
                data_qtype = QT.F32
            elif any(x in key for x in UNIVERSAL_BLACKLIST):
                # 结构敏感层 / 裸 Parameter（modulation 调制表、embed、norm、first/last 投影等）
                # 必须保持高精度：GGUF 加载器只在 operations.Linear/Conv 等 GGMLLayer 模块里
                # 懒反量化；裸 nn.Parameter（如 Krea2 的 last.modulation.lin=[2,dim]）不走
                # GGMLLayer，量化后会以打包形状 [2, dim*9/16] 直接 load_state_dict → 形状不匹配
                # →「SingleStreamDiT: last.modulation.lin [2,6144] vs [2,3456]」报错。
                # 这正是 GGUF 产出「能转不能加载」的根因。和 NVFP4 分支共用同一份黑名单。
                data_qtype = QT.F32
            else:
                data_qtype = target_q
        else:
            data_qtype = QT.F16

        # SD1/SDXL 等需要 reshape 对齐 256
        if (shape_fix and n_dims > 1 and n_params >= REARRANGE_THRESHOLD
                and (n_params / 256).is_integer() and not (data.shape[-1] / 256).is_integer()):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            try:
                writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(x) for x in orig_shape))
            except Exception:
                pass

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, Exception) as e:
            # 单张量回退：通常是该层形状不满足目标 qtype 的块对齐（如 Q8_0 需末维 %32==0）。
            # 记账，结尾统一警告——避免「大量层静默回退 F16 → 文件反而变大」却无人察觉。
            fallback_count += 1
            fallback_bytes += data.nbytes
            print(f"[binyuan NVFP4 Converter] 量化回退 F16: {key} ({e})")
            data_qtype = QT.F16
            data = gguf.quants.quantize(data, data_qtype)

        try:
            writer.add_tensor(key, data, raw_dtype=data_qtype)
        except Exception as e:
            print(f"[binyuan NVFP4 Converter] 写入张量失败，跳过: {key} ({e})")

    if fallback_count:
        print(f"[binyuan NVFP4 Converter] ⚠️ 共 {fallback_count} 个张量回退为 F16"
              f"（约 {fallback_bytes / 1e9:.2f} GB 按 F16 计）。若回退量很大，说明目标 qtype "
              f"对源模型形状不友好，可改选 Q8_0（块对齐最宽松）。")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer.write_header_to_file(path=output_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    _msg = f"成功！GGUF 模型已保存至: {output_path}"
    if _subst_notice:
        _msg += f"\n{_subst_notice}"
    return _msg


def _detect_source_quant(input_path):
    """检测源模型是否已被量化过。

    返回 (fmt, note)：
      fmt  ∈ {None, "FP8", "NVFP4", "已量化"}
      note 为人类可读的判定依据。

    为什么要检测：本节点把 FP8 源当普通模型处理时会出两个问题——
      1) FP8→FP8：FP8→BF16→FP8，1 字节→1 字节，体积不可能再缩小；
      2) FP8→GGUF：旧逻辑把 FP8 权重 cast 成 F16 存盘（翻倍）且完全不量化。
    检测到后要么拦截提示、要么走修正后的反量化路径。
    """
    try:
        with safetensors.safe_open(input_path, framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata() or {}
            # 路径 A：本节点 / comfy 原生量化的产物，带 comfy_quant 键或 _quantization_metadata
            if "_quantization_metadata" in meta or any(k.endswith(".comfy_quant") for k in keys):
                # 优先解码一个 comfy_quant 张量（uint8 编码的 JSON）拿精确格式，比遍历
                # _quantization_metadata 更可靠（社区 FP8 文件常只有 comfy_quant 键、无该元数据）。
                cq_keys = [k for k in keys if k.endswith(".comfy_quant")]
                if cq_keys:
                    try:
                        raw = bytes(f.get_tensor(cq_keys[0]).tolist()).decode("utf-8", "replace")
                        fmt_val = json.loads(raw).get("format", "")
                        if "nvfp4" in fmt_val:
                            return ("NVFP4", f"comfy 量化标记（comfy_quant → {fmt_val}）")
                        if "float8" in fmt_val:
                            return ("FP8", f"comfy 量化标记（comfy_quant → {fmt_val}）")
                    except Exception:
                        pass
                try:
                    qm = json.loads(meta.get("_quantization_metadata", "{}"))
                    fmts = {layer.get("format") for layer in qm.get("layers", {}).values()}
                    if "nvfp4" in fmts:
                        return ("NVFP4", "comfy 量化标记（_quantization_metadata）")
                    if any("float8" in str(x) for x in fmts):
                        return ("FP8", "comfy 量化标记（_quantization_metadata）")
                except Exception:
                    pass
                return ("已量化", "comfy 量化标记（comfy_quant 键）")
            # 路径 B：社区分发的 FP8 模型（无 comfy 标记）——采样大权重看 dtype
            fp8_seen = 0
            for k in keys:
                if not k.endswith(".weight"):
                    continue
                try:
                    t = f.get_tensor(k)
                except Exception:
                    continue
                if t.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
                    fp8_seen += 1
                    if fp8_seen >= 3:
                        break
            if fp8_seen >= 1:
                return ("FP8", f"权重 dtype 为 float8（社区 FP8 发布版，采样命中 {fp8_seen} 个权重）")
    except Exception as e:
        print(f"[binyuan NVFP4 Converter] 源量化检测跳过: {e}")
    return (None, "")


class BinyuanNVFP4Converter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = get_all_models()
        return {
            "required": {
                "input_model": (model_list,),
                "save_to_source_dir": ("BOOLEAN", {"default": True}),
                "custom_output_dir": ("STRING", {"default": "F:/custom_models_dir/", "tooltip": "手动输入保存目录。留空则看 save_to_source_dir：开=存到源模型同目录，关=报错。填了目录就以此为准（save_to_source_dir 被忽略）"}),
                "output_filename": ("STRING", {"default": "model_nvfp4"}),
                "model_type": ([
                    "Auto (Universal)",                # 任意模型通吃：文生图/图生图/文生视频/图生视频/最新模型
                    "Boogu-Image",                     # Boogu image base/edit/turbo（Flux/Z-Image 混合）
                    "Krea2-Turbo",                     # Krea2 turbo DiT
                    "Flux.2-Klein-9b",
                    "Flux.2-Klein-4b",                 # KLEN4 (4B 蒸馏版)
                    "Flux.2-Klein-base-9b",            # KLEN9B (9B 基础版)
                    "Flux.2-Klein-base-4b",            # KLEN4 (4B 基础版)
                    "Flux.1-dev",
                    "Flux.1-Fill",
                    "Flux.2-dev",
                    "Baidu-ERNIE-Image",               # 百度文心 8B
                    "Ideogram-4.0",                    # Ideogram 4.0
                    "Bernini-Wan2.2",                  # ByteDance Bernini
                    "LTX-2.3-Unified",                 # LTX-2.3 (22B)
                    "Microsoft-LENS",                  # 微软 LENS
                    "NVIDIA-PID-Decoder",              # 英伟达 PID Decoder
                    "Wan2.2-i2v-high-low",
                    "LTX-2-19b-dev-or-distilled",
                    "Z-Image-Turbo",
                    "Z-Image-Base",
                    "Qwen-Image-Edit-2511",
                    "Qwen-Image-2512"
                ], {"default": "Auto (Universal)", "tooltip": "NVFP4/FP8 时选择黑名单策略；GGUF 时忽略（用架构自动检测）"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                # 新增控件放在末尾：旧工作流存的 widgets_values 按位置对齐时，
                # 前面几个（input_model...device）顺序不变，这两个用默认值，不会错位报错。
                "output_format": ([
                    "NVFP4",
                    "FP8 (e4m3fn)",
                    "GGUF",
                ], {"default": "NVFP4"}),
                "gguf_qtype": ([
                    "Q8_0", "Q5_0", "Q4_0",
                    "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K",
                    "IQ4_NL", "IQ4_XS", "IQ3_S",
                    "BF16", "F16",
                ], {"default": "Q8_0", "tooltip": "仅 output_format=GGUF 时生效。⚠️ 本机 gguf 库通常只支持量化到 Q8_0/Q5_0/Q4_0/BF16/F16；K 系列(Q4_K 等)与 IQ 系列在纯 Python gguf 里没实现量化。选了不支持的类型会自动改用同尺寸可用类型（Q4_K→Q4_0、Q5_K→Q5_0、Q6_K→Q8_0…）并打印提示，不会报错。要压 FP8 源推荐直接选 Q4_0(≈0.56) 或 Q5_0(≈0.69)。"}),
                "full_precision_mm": ("BOOLEAN", {"default": True, "tooltip": "NVFP4/FP8 推理模式。开=反量化到全精度做矩阵乘（任何GPU都能跑、最稳，但无省显存收益）；关=原生量化矩阵乘（真省显存/加速，需Blackwell+cu130+comfy_kitchen CUDA后端，否则可能报错）"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "binyuan/Advanced"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _save_safetensors_robust(sd, final_path, metadata):
        """稳健保存 safetensors。

        Windows 上把 7GB+ 的 safetensors 直接写进 ComfyUI 的 models 目录时，ComfyUI 的
        文件夹监视器 / 杀毒实时扫描会中途打开正在写入的文件，导致 safetensors 抛出误导性的
        「磁盘空间不足 (os error 112)」——磁盘其实没满。对策：先写到「不被监视的临时目录」
        （ComfyUI output 目录优先，且与目标同卷以便原子改名），写完后用 os.replace 原子替换
        到最终名，监视器只在文件完整后才能看到它。失败再换卷用系统临时目录 + shutil.move。
        """
        import shutil, tempfile, time

        final_dir = os.path.dirname(os.path.abspath(final_path)) or "."

        def dev_of(d):
            try:
                return os.stat(d).st_dev
            except OSError:
                return None

        final_dev = dev_of(final_dir)

        # 候选临时目录：output（不被 models 监视）> 系统临时 > 目标目录兜底
        candidates = []
        try:
            od = folder_paths.get_output_directory()
            if od:
                candidates.append(od)
        except Exception:
            pass
        candidates.append(tempfile.gettempdir())
        candidates.append(final_dir)
        # 同卷优先（可原子 os.replace），去重
        seen = set()
        ordered = []
        for d in candidates:
            d = os.path.abspath(d)
            if d in seen:
                continue
            seen.add(d)
            ordered.append(d)
        ordered.sort(key=lambda d: 0 if dev_of(d) == final_dev else 1)

        def put_final(tmp):
            """把临时文件落到最终名：同卷原子 os.replace，跨卷 shutil.move。"""
            if dev_of(os.path.dirname(tmp)) == final_dev:
                try:
                    os.replace(tmp, final_path)
                    return
                except OSError:
                    try:
                        os.remove(final_path)
                    except OSError:
                        pass
                    os.replace(tmp, final_path)
            else:
                try:
                    os.remove(final_path)
                except OSError:
                    pass
                shutil.move(tmp, final_path)

        last_err = None
        for cdir in ordered:
            try:
                os.makedirs(cdir, exist_ok=True)
            except OSError:
                continue
            for attempt in range(2):
                tmp = None
                try:
                    fd, tmp = tempfile.mkstemp(suffix=".safetensors", prefix="binyuan_", dir=cdir)
                    os.close(fd)
                    safetensors.torch.save_file(sd, tmp, metadata=metadata)
                    put_final(tmp)
                    return f"成功！模型已保存至: {final_path}"
                except Exception as e:
                    last_err = e
                    print(f"[binyuan NVFP4 Converter] 保存到 {cdir} 第 {attempt + 1} 次失败: {e}")
                    if tmp and os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                    time.sleep(1)

        return (f"保存量化模型失败: {last_err}\n"
                "  建议：1) 把 ComfyUI 的 models 目录加入杀毒软件/Windows Defender 白名单；"
                "2) 关闭可能占用目标文件的程序（模型加载器/资源管理器预览）后重试；"
                "3) 用 save_to_source_dir=False 输出到 models 之外的目录。")

    def convert(self, input_model, save_to_source_dir, custom_output_dir, output_filename,
                output_format, gguf_qtype, model_type, device, full_precision_mm=True):

        # 依赖检查
        if output_format != "GGUF" and not HAS_KITCHEN:
            return ("错误: 未检测到 comfy-kitchen 库。NVFP4/FP8 需要在终端运行 'pip install comfy-kitchen'。",)
        if output_format == "GGUF":
            try:
                import gguf  # noqa
            except ImportError:
                return ("错误: 未检测到 gguf 库。GGUF 需要在终端运行 'pip install gguf'（通常随 ComfyUI-GGUF 安装）。",)

        if "没有检测到模型" in input_model:
            return ("错误: 未选择任何有效输入模型，请检查您的模型文件夹。",)

        # 1. 解析选择的文件路径
        try:
            folder, filename = input_model.split("/", 1)
            input_path = folder_paths.get_full_path(folder, filename)
        except Exception as e:
            return (f"解析输入模型路径失败: {str(e)}",)

        if not input_path or not os.path.exists(input_path):
            return (f"错误: 找不到输入模型的实际物理路径: {input_path}",)

        # 检测源模型是否已量化过。FP8/NVFP4 源再走相同或更弱的量化不会缩小，直接拦截并给出
        # 正确建议，避免用户等半天得到一个「没变小 / 反而变大」的文件。
        src_fmt, src_note = _detect_source_quant(input_path)
        if src_fmt:
            print(f"[binyuan NVFP4 Converter] ⚠️ 检测到源模型已量化: {src_fmt}（{src_note}）")
            same_format = (
                (src_fmt == "FP8" and output_format == "FP8 (e4m3fn)")
                or (src_fmt == "NVFP4" and output_format == "NVFP4")
            )
            if same_format:
                return (f"⚠️ 源模型已是 {src_fmt} 量化版（{src_note}），再转 {output_format} 体积不会缩小"
                        f"（已经是该精度，{src_fmt}→{src_fmt} 是 1 字节→1 字节）。\n"
                        "  想要更小请换格式：NVFP4≈0.56 字节/参数；GGUF Q5_0≈0.69、Q4_0≈0.56"
                        "（K 系列/IQ 系列需升级 gguf 库才能用）。\n"
                        "  若确实要二次量化（会再掉一点精度），请先用其它工具反量化到 BF16 再来转。",)
            # NVFP4 源 → GGUF：当前 GGUF 分支无法解读 NVFP4 的 uint4 打包，会把它当 uint8
            # 存成 F16，体积反而膨胀。先拦截。
            if src_fmt == "NVFP4" and output_format == "GGUF":
                return ("⚠️ 源模型已是 NVFP4 量化版，本节点 GGUF 分支无法直接解读 NVFP4 的 uint4 打包格式"
                        "（会按 uint8 当 F16 存盘导致体积膨胀）。\n"
                        "  请先用其它工具把 NVFP4 反量化到 BF16，再转 GGUF；或直接用 NVFP4 推理。",)
            # FP8 源 → GGUF：Q8_0(≈1.06)/BF16/F16 每个 ≥1 字节，比 FP8(1.0) 还大，量化反而
            # 膨胀——这正是「FP8 模型转 Q8_0 GGUF 后比原版还大」的根因。只有 <1 字节的
            # Q5_0(≈0.69)/Q4_0(≈0.56) 才能真正缩小 FP8 源。
            if output_format == "GGUF" and src_fmt == "FP8":
                import gguf as _gguf
                _qt = _gguf_qtype_map().get(gguf_qtype, _gguf.GGMLQuantizationType.Q8_0)
                _bs = _gguf.GGML_QUANT_SIZES.get(_qt)
                _bpp = (_bs[1] / _bs[0]) if _bs else 2.0
                if _bpp >= 1.0:
                    _src_gb = os.path.getsize(input_path) / 1e9
                    return (f"⚠️ 源模型已是 FP8（1.0 字节/参数），而你选的 {gguf_qtype} ≈{_bpp:.2f} 字节/参数，"
                            f"量化后约 {_src_gb * _bpp:.1f}G，会比源（{_src_gb:.1f}G）还大，没意义。\n"
                            "  FP8 源要缩小只能选 Q5_0（≈0.69 字节/参数）或 Q4_0（≈0.56）。\n"
                            "  Q8_0（≈1.06）只适合从 BF16/F16 压缩，压不了 FP8；BF16/F16 同理只会变大。",)
                print(f"[binyuan NVFP4 Converter] 提示: FP8 源二次量化到 {gguf_qtype}（≈{_bpp:.2f} 字节/参数），"
                      "体积会缩小，但会有轻微精度损失。")
            elif output_format == "NVFP4" and src_fmt == "FP8":
                print("[binyuan NVFP4 Converter] 提示: FP8 源二次量化到 NVFP4（≈0.56 字节/参数），"
                      "体积会缩小，但会有轻微精度损失。")

        # 2. 输出目录：手填的 custom_output_dir 优先；否则按 save_to_source_dir
        manual_dir = (custom_output_dir or "").strip().replace("\\", "/")
        if manual_dir:
            output_dir = manual_dir
        elif save_to_source_dir:
            output_dir = os.path.dirname(input_path)
        else:
            return ("错误: 未指定输出目录：请在 custom_output_dir 手填目录，或开启 save_to_source_dir。",)

        # 3. 按格式决定扩展名
        if output_format == "GGUF":
            if not output_filename.lower().endswith(".gguf"):
                output_filename = os.path.splitext(output_filename)[0] + ".gguf"
        else:
            if not output_filename.endswith(".safetensors"):
                output_filename += ".safetensors"
        output_path = os.path.join(output_dir, output_filename)

        # ============ GGUF 独立分支（不同文件格式，不走 safetensors 流程）============
        if output_format == "GGUF":
            print(f"[binyuan NVFP4 Converter] 🚀 GGUF 转换: {input_path} -> {output_path}")
            try:
                status = _convert_to_gguf(input_path, output_path, gguf_qtype)
            except Exception as e:
                import traceback
                traceback.print_exc()
                status = f"GGUF 转换失败: {e}"
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[binyuan NVFP4 Converter] {status}")
            return (status,)

        # ============ NVFP4 / FP8 分支（safetensors）============
        # --- 精确配置黑名单机制 ---
        if model_type == "Auto (Universal)":
            BLACKLIST = list(UNIVERSAL_BLACKLIST)
            FP8_LAYERS = []
        elif model_type == "Boogu-Image":
            # Boogu：Flux/Z-Image 混合架构（x_embedder / context_refiner / noise_refiner
            # / double_stream_layers / single_stream_layers / img_instruct_attn）。
            # ⚠️ 不能用裸 "img_in"——会误伤 img_instruct_attn 的大权重（要量化的）。
            # "norm" 同时覆盖 norm_out（最终输出投影）与各 norm/modulation-linear。
            BLACKLIST = ["embed", "refiner", "norm", "bias", "modulation", "adaLN", "adaln", "final_layer"]
            FP8_LAYERS = []
        elif model_type == "Krea2-Turbo":
            # Krea2 DiT（first 输入嵌入 / last 输出层 / blocks / txtfusion 文本融合）。
            # 保护 first、last、mod（调制）、norm、文本投影 tmlp/tproj/projector；
            # 量化 attn.wq/wk/wv/wo 与 mlp.down/gate/up。
            # ⚠️ 不能用裸 "gate"——会误伤 mlp.gate（SwiGLU 大权重）；用 "mod." 而非
            # "mod"——"mod." 不会命中 "model." 前缀（checkpoint 版有 model.diffusion_model.）。
            BLACKLIST = ["first", "last", "mod.", "norm", "projector", "tmlp", "tproj", "bias", "vae.", "text_encoders"]
            FP8_LAYERS = []
        elif model_type == "Baidu-ERNIE-Image":
            BLACKLIST = ["img_in", "txt_in", "time_in", "final_layer", "proj_out"]
            FP8_LAYERS = []
        elif model_type == "Ideogram-4.0":
            BLACKLIST = [
                "cap_embedder", "x_embedder", "noise_refiner", "final_layer", "t_embedder", "norm", "bias", "pos_embed",
                "embed_image_indicator", "txt_in", "img_in", "time_in"
            ]
            FP8_LAYERS = []
        elif model_type == "Bernini-Wan2.2":
            BLACKLIST = ["text_embedding", "time_embedding", "time_projection", "head", "mllm_planner", "connector", "norm", "bias"]
            FP8_LAYERS = []
        elif model_type == "LTX-2.3-Unified":
            BLACKLIST = [
                "vae.", "vocoder.", "connector", "proj_out",
                "norm", "bias", "scale", "embedder", "patchify", "table",
                "transformer_blocks.0.",
                "transformer_blocks.43.", "transformer_blocks.44.",
                "transformer_blocks.45.", "transformer_blocks.46.",
                "transformer_blocks.47.", "projection", "adaln_single"
            ]
            FP8_LAYERS = []
        elif model_type == "Microsoft-LENS":
            BLACKLIST = ["img_in", "txt_in", "time_in", "final_layer", "proj_out", "tokenizer", "moe_gate", "norm", "bias"]
            FP8_LAYERS = []
        elif model_type == "NVIDIA-PID-Decoder":
            BLACKLIST = ["img_in", "txt_in", "norm", "bias", "final_layer", "proj_out", "pixel_dit_adapter"]
            FP8_LAYERS = []
        elif model_type == "Qwen-Image-Edit-2511":
            BLACKLIST = ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"]
            FP8_LAYERS = []
        elif model_type == "Qwen-Image-2512":
            BLACKLIST = ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out", "img_mod.1"]
            FP8_LAYERS = ["txt_mlp", "txt_mod"]
        elif model_type == "Wan2.2-i2v-high-low":
            BLACKLIST = ["text_embedding", "time_embedding", "time_projection", "head"]
            FP8_LAYERS = []
        elif model_type in ["Flux.1-dev", "Flux.1-Fill", "Flux.2-dev", "Flux.2-Klein-9b", "Flux.2-Klein-4b", "Flux.2-Klein-base-9b", "Flux.2-Klein-base-4b"]:
            BLACKLIST = [
                "bias", "txt_attn", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer", "class_embedding",
                "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt",
                "img_mod", "txt_mod", "modulation", "mod"
            ]
            FP8_LAYERS = []
        elif model_type == "Z-Image-Base":
            BLACKLIST = ["attention", "adaLN_modulation", "norm", "final_layer", "cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "pad_token"]
            FP8_LAYERS = []
        elif model_type == "Z-Image-Turbo":
            BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer", "pad_token"]
            FP8_LAYERS = []
        else:
            BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer", "pad_token"]
            FP8_LAYERS = []

        print(f"[binyuan NVFP4 Converter] 🚀 模型类型 {model_type}，输出格式 {output_format}")

        # 自动且无条件读取源模型的所有原有元数据（Metadata）
        temp_orig_meta = {}
        try:
            with safetensors.safe_open(input_path, framework="pt") as f:
                orig_meta = f.metadata()
                if orig_meta:
                    for key, val in orig_meta.items():
                        temp_orig_meta[key] = val
        except Exception as e:
            print(f"[binyuan NVFP4 Converter] 读取原文件元数据时发生异常（如无元数据将跳过）: {e}")

        sd = safetensors.torch.load_file(input_path)
        quant_map = {"format_version": "1.0", "layers": {}}
        new_sd = {}

        pbar = comfy.utils.ProgressBar(len(sd))
        print(f"[binyuan NVFP4 Converter] ⚙️ 开始转换，计算设备: {device}")

        for i, (k, v) in enumerate(sd.items()):
            pbar.update_absolute(i + 1)

            # 丢弃 FP8 格式标量（weight_scale/weight_scale_2/input_scale）——weight_scale 已
            # 用于下面的反量化，input_scale 是激活用（NVFP4/FP8 full_precision_mm 不需要）。
            # 留着会作为 BF16 标量写进输出，污染加载器、让 comfy_quant 层被误判。
            if (k.endswith(".weight_scale") or k.endswith(".weight_scale_2")
                    or k.endswith(".input_scale") or k.endswith(".comfy_quant")):
                continue

            # 源是带 scale 的 FP8 时，乘 weight_scale 还原真实值。直接 v.to(bf16) 不乘 scale
            # 得到 fp8 自然范围值（abs.max≈448），与真实权重（abs.max≈0.3）差上千倍，量化后
            # 模型出图全是噪点——这是 Klein 等 FP8 源转 NVFP4「能加载但出噪点」的根因。
            # 无 scale 的纯 FP8（如 Krea2）和 BF16 源不受影响（走 else 直接 cast）。
            if v.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
                _sk = k + "_scale"  # base.weight -> base.weight_scale
                if _sk in sd:
                    v = v.to(dtype=torch.bfloat16) * sd[_sk].to(dtype=torch.bfloat16)
                else:
                    v = v.to(dtype=torch.bfloat16)

            if any(name in k for name in BLACKLIST):
                new_sd[k] = v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
                continue

            if v.ndim == 2 and ".weight" in k:
                base_k_file = k.replace(".weight", "")

                if model_type in ["LTX-2-19b-dev-or-distilled", "LTX-2.3-Unified"]:
                    base_k_meta = k.replace(".weight", "")
                else:
                    if "model.diffusion_model." in base_k_file:
                        base_k_meta = base_k_file.split("model.diffusion_model.")[-1]
                    else:
                        base_k_meta = base_k_file

                v_tensor = v.to(device=device, dtype=torch.bfloat16)

                # ---- FP8 全量量化（不要求 16 整除，per-tensor scale）----
                if output_format == "FP8 (e4m3fn)":
                    try:
                        weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                        weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                        new_sd[k] = weight_quantized.cpu()
                        new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                        new_sd[f"{base_k_file}.comfy_quant"] = _comfy_quant_tensor("float8_e4m3fn", full_precision_mm).cpu()
                        qconf = {"format": "float8_e4m3fn"}
                        if full_precision_mm:
                            qconf["full_precision_matrix_mult"] = True
                        quant_map["layers"][base_k_meta] = qconf
                    except Exception as e:
                        print(f"⚠️ FP8 量化出错 {k}，保留 BF16。原因: {e}")
                        new_sd[k] = v_tensor.cpu()
                    if device == "cuda":
                        del v_tensor
                    continue

                # ---- NVFP4 路径：FP8 混合层 ----
                if FP8_LAYERS and any(name in k for name in FP8_LAYERS):
                    print(f"🌸 FP8 混合层 : {k}")
                    weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                    weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                    new_sd[k] = weight_quantized.cpu()
                    new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                    new_sd[f"{base_k_file}.comfy_quant"] = _comfy_quant_tensor("float8_e4m3fn", full_precision_mm).cpu()
                    qconf = {"format": "float8_e4m3fn"}
                    if full_precision_mm:
                        qconf["full_precision_matrix_mult"] = True
                    quant_map["layers"][base_k_meta] = qconf
                    if device == "cuda":
                        del v_tensor
                    continue

                # ---- NVFP4 核心转换：两维都能被 16 整除才量化 ----
                if v_tensor.shape[0] % 16 == 0 and v_tensor.shape[1] % 16 == 0:
                    print(f"💎 NVFP4 量化 : {k} 尺寸 {list(v_tensor.shape)}")
                    try:
                        qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                        tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                        for suffix, tensor in tensors.items():
                            new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                        new_sd[f"{base_k_file}.comfy_quant"] = _comfy_quant_tensor("nvfp4", full_precision_mm).cpu()
                        qconf = {"format": "nvfp4"}
                        if full_precision_mm:
                            qconf["full_precision_matrix_mult"] = True
                        quant_map["layers"][base_k_meta] = qconf
                    except Exception as e:
                        print(f"⚠️ 量化出错 {k}，保留原精度。 错误原因: {e}")
                        new_sd[k] = v_tensor.cpu()
                else:
                    # 维度不整除时，自动保护降级到 BF16
                    print(f"🛡️ 降级防护（维度未整除）: {k} 尺寸 {list(v_tensor.shape)}")
                    new_sd[k] = v_tensor.cpu()

                if device == "cuda":
                    del v_tensor
            else:
                new_sd[k] = v.to(dtype=torch.bfloat16) if v.is_floating_point() else v

        # 写入全局元数据信息（完全承袭原文件原有的全部元数据）
        final_metadata = OrderedDict()
        if temp_orig_meta:
            for k_meta, v_meta in temp_orig_meta.items():
                final_metadata[k_meta] = v_meta

        final_metadata["_quantization_metadata"] = json.dumps(quant_map)
        final_metadata["converted_by"] = "Binyuan NVFP4 Converter"
        final_metadata["converter_url"] = "Customised with ComfyUI Integration"
        final_metadata["output_format"] = output_format

        print(f"[binyuan NVFP4 Converter] 💾 正在保存模型到目标路径: {output_path}")
        os.makedirs(output_dir, exist_ok=True)
        status = self._save_safetensors_robust(new_sd, output_path, final_metadata)

        del sd, new_sd
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"[binyuan NVFP4 Converter] {status}")
        return (status,)

NODE_CLASS_MAPPINGS = {
    "BinyuanNVFP4Converter": BinyuanNVFP4Converter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BinyuanNVFP4Converter": "binyuan Universal Quant Converter (NVFP4/FP8/GGUF)"
}
