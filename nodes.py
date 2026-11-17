"""王知风 · 视频改拍（基于 Viggle Remake）

①视频分析：截取（最长 30 秒）、边解码边转 24fps 并缩放到 Viggle 画布、自动找切镜、按每段帧数切成几段（段尽量断在切镜处，
  一个镜头比一段还长就让下一段接力）、每段给它的每个镜头选一帧关键帧、出缩略图。
②画关键帧：本地 Qwen-Image 2.1（换整个人 / 只换头用 BFS 换头 LoRA / 改物），画好的帧决定成片里人长什么样（实测：
  只给头像不画帧，脸不会换）。8B 先描述参考照片里的人写进提示词，画完自检脸换掉没有，没换掉换种子重画。
③自动写提示词：六段式结构由代码按 Viggle 官方句式写死，本地 Qwen3-VL 8B 只看图写短描述（实测：整段交给 8B 写，
  会把原来的人和新人写成两个角色）。句式改编自 Viggle Remake 的 mcp/（Apache-2.0，见 data/ 和 NOTICE.md）。
④改拍模型：显存档位默认「自动」（16G int8 / 12G GGUF Q8_0 + 运行时多留 8GB 显存 / 80G+ bf16+LoRA）× 模式（换人 / 改物），
  同时给出官方采样曲线。
⑤生成：逐段生成。先把每段的条件都编码好（32B 文本编码器只上一次显卡），再逐段采样；接力段开头 17 帧用上一段成片的帧，
  原样保留不重画；最后拼起来、修切镜接缝的闪帧、配原声。分段和接力的做法照 Viggle Remake 的 mcp/kit.py。
⑥左右对比：原片 | 成片，拼成一个画面。

MiniMax H3 · Powered by Viggle Remake (https://huggingface.co/Viggle/Viggle-Remake)
"""
import gc
import io as _io
import json
import logging
import os
import re

import numpy as np
import torch

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import nodes
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_custom_sampler import Guider_Basic
from comfy_extras.nodes_qwen import QwenImage21Cache, TextEncodeQwenImage21
from comfy_extras.nodes_textgen import TextGenerate

from . import remake_core as R

log = logging.getLogger("WZF-VideoRemake")
HERE = os.path.dirname(os.path.abspath(__file__))

MODES = ["换整个人", "只换头", "改物/改场景"]
MODE_KEY = {"换整个人": "whole", "只换头": "head", "改物/改场景": "edit"}
TIERS = ["16G 本地（int8）", "12G 本地（Q8_0，需12G补充包）", "云端 80G+（bf16 + LoRA）"]
AUTO = "自动（按显卡显存选）"
RESERVE_12G = 8 * 1024 ** 3  # 官方 12G 配方 --reserve-vram 8；运行时改同一个变量，不用单独的启动器
SIGMAS = {"swap": [1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579, 0.0],
          "edit": [1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882, 0.0]}

QWEN = {"unet": "qwen_image_2.1_int8_convrot.safetensors",
        "detail": "elusarcas-qwen2-1-detailer-v1.safetensors",
        "turbo": "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
        "bfs": "bfs_head_v1.1_qwen_2.1.safetensors",
        "te": "qwen3vl_8b_w4a8.safetensors",
        "vae": "qwen_image_2.1_vae_bf16.safetensors"}
VLM_TYPE = os.environ.get("WZF_REMAKE_VLM_TYPE", "ltxv")  # 同 H3 生视频：Qwen3-VL 8B 按 ltxv 类型加载当看图写字模型

HEADSWAP = ("head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. "
            "remove the head from <image1> completely and replace it with the head from <image2>, strictly preserving "
            "the hair, eye color, nose structure from <image2>. copy the direction of the eye, head rotation, micro "
            "expressions from <image1>, high quality, sharp details, 4k")


def _base(n):
    return n.replace("\\", "/").split("/")[-1]


def find(folder, name, *alts, keys=None, optional=False):
    """先按准确文件名找（name，再依次 alts），找不到再按关键词找（文件名同时包含 keys 里每个词，不分大小写）。
    云平台（如 RunningHub）的模型库是用户上传的，文件名常被改，例如 Viggle_Remake_swap_pruned_int8_convrot_1108_S。"""
    try:
        names = folder_paths.get_filename_list(folder)
    except Exception:  # noqa: BLE001  文件夹类型没登记（如没装 ComfyUI-GGUF 时的 unet_gguf）
        names = []
    for want in (name,) + alts:
        for n in names:
            if _base(n) == want:
                return n
    if keys:
        hits = sorted(n for n in names if all(k.lower() in _base(n).lower() for k in keys))
        if hits:
            log.info(f"[找模型] 没有 {name}，按关键词 {list(keys)} 用 {hits[0]}")
            return hits[0]
    if optional:
        return None
    raise RuntimeError(f"缺少模型：ComfyUI/models/{folder}/{name}\n请按使用说明下载后放进对应文件夹。")


TE_ALTS = ("qwen3vl_8b_bf16.safetensors", "qwen3vl_8b_int8_convrot.safetensors", "qwen3vl_8b_fp8_scaled.safetensors")


def qwen_te():
    return find("text_encoders", QWEN["te"], *TE_ALTS)


def free():
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def resize(img, w, h):
    """[B,H,W,C] → [B,h,w,C]，lanczos。"""
    return comfy.utils.common_upscale(img.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1)


def fit_canvas(frames, short_edge):
    """等比缩放到短边 = short_edge，再居中裁到 32 的倍数（Viggle 节点见到这种尺寸就原样使用，不再裁成 16:9）。"""
    h, w = frames.shape[1], frames.shape[2]
    s = short_edge / min(h, w)
    nh, nw = max(short_edge, round(h * s)), max(short_edge, round(w * s))
    ch, cw = nh // 32 * 32, nw // 32 * 32
    out = []
    for i in range(0, frames.shape[0], 32):
        x = resize(frames[i:i + 32, ..., :3].float(), nw, nh)
        t, l = (nh - ch) // 2, (nw - cw) // 2
        out.append(x[:, t:t + ch, l:l + cw])
    return torch.cat(out)


def contact_sheet(frames, shots, n=12, cols=4):
    """缩略图：均匀抽 n 帧，标上帧号、时间、镜头号，给 8B 看。"""
    from PIL import Image, ImageDraw
    import numpy as np
    nf = frames.shape[0]
    idx = sorted(set([round(i * (nf - 1) / (n - 1)) for i in range(n)]))
    tw = 320
    th = round(frames.shape[1] * tw / frames.shape[2])
    rows = (len(idx) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for k, i in enumerate(idx):
        im = (frames[i, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        tile = Image.fromarray(im).resize((tw, th), Image.LANCZOS)
        shot = next(j for j, (s, e) in enumerate(shots) if s <= i < e) + 1
        d = ImageDraw.Draw(tile)
        label = f"f{i} {R.mmss(i)} Shot {shot}"
        d.rectangle([0, 0, 8 * len(label) + 8, 16], fill=(0, 0, 0))
        d.text((4, 2), label, fill=(255, 255, 0))
        sheet.paste(tile, ((k % cols) * tw, (k // cols) * th))
    return torch.from_numpy(np.asarray(sheet).astype("float32") / 255)[None]


def _decode_stream(clip, short_edge, limit):
    """用 ComfyUI 自己的逐帧解码（它处理旋转、裁剪、奇数宽度、HDR），每攒 32 帧就缩到画布，内存里只留缩小后的帧。"""
    import av
    from comfy_api.latest._input_impl import video_types as VT
    src = clip.get_stream_source()
    if isinstance(src, _io.BytesIO):
        src.seek(0)
    with av.open(src, mode="r") as container:
        vs = clip._get_first_video_stream(container)
        vs.thread_type = "AUTO"
        fps = float(vs.average_rate) if vs.average_rate else float(R.FPS)
        same = abs(fps - R.FPS) < 0.01
        ast = VT.last_decodable_audio_stream(container)
        afr, out, buf = [], [], []
        j, want, stop = 0, 0, None  # j：下一个要的 24fps 帧；want：它对应的源帧号（同 R.resample_indices）

        def flush():
            if buf:
                x = torch.from_numpy(np.stack(buf))
                x = x.float() / 255 if x.dtype == torch.uint8 else x.float()
                out.append(fit_canvas(x, short_edge))
                buf.clear()

        for i, img in enumerate(clip._decode_frames(container, vs, ast, afr, VT.FrameChunks())):
            while j < limit and want == i:
                buf.append(img)
                j += 1
                want = j if same else int(round(j / R.FPS * fps))
            if len(buf) >= 32:
                flush()
            if j >= limit:
                stop = i + int(fps) if stop is None else stop  # 再读 1 秒，让交错存放的音频读够
                if i >= stop:
                    break
        flush()
        audio = None
        if afr:
            a = np.concatenate(afr, axis=1)
            audio = {"waveform": torch.from_numpy(a).unsqueeze(0), "sample_rate": int(ast.sample_rate or 1)}
    if not out:
        raise RuntimeError("视频里没解出画面。")
    return torch.cat(out), audio, fps


def decode_canvas(video, start, seconds, short_edge):
    """截取 → 转 24fps → 缩到画布。30 秒 1080p 整段解码成 float 要 20GB 以上内存，所以边解码边缩小；
    ComfyUI 内部接口对不上时（别的版本）退回整段解码。"""
    limit = int(round(seconds * R.FPS))
    clip = video.as_trimmed(start, seconds + 0.5, strict_duration=False) or video
    try:
        src, audio, fps = _decode_stream(clip, short_edge, limit)
    except (AttributeError, TypeError, ImportError) as e:
        log.warning(f"[视频分析] 逐帧解码用不了（{e}），改用整段解码")
        comp = clip.get_components()
        fps, audio = float(comp.frame_rate), comp.audio
        idx = R.resample_indices(comp.images.shape[0], fps)[:limit]
        src = fit_canvas(comp.images[torch.tensor(idx)], short_edge)
    return src, audio, fps


def cut_audio(audio, f0, f1):
    """原声的 [f0, f1) 帧那一段（24fps）。"""
    if audio is None:
        return None
    sr = audio["sample_rate"]
    return {"waveform": audio["waveform"][..., round(f0 / R.FPS * sr):round(f1 / R.FPS * sr)], "sample_rate": sr}


class WZFRemakePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakePlan", display_name="① 视频分析（切镜 · 分段 · 关键帧）", category="王知风/视频改拍",
            description="上传的视频 → 24fps、缩放到画布、自动找切镜；超过一段的长度就自动分段（段尽量断在切镜处，"
                        "一个镜头比一段还长就让下一段接力），每段给它的每个镜头选一帧关键帧。",
            inputs=[
                io.Video.Input("video", display_name="原视频"),
                io.Combo.Input("mode", options=MODES, default=MODES[0], display_name="改什么"),
                io.String.Input("who", default="", display_name="换谁（多人时写清楚）",
                                tooltip="例：穿灰色外套的女人 / 左边的男人。画面里只有一个人可以不填。改物时不用填。"),
                io.String.Input("target", multiline=True, default="", display_name="改成什么样",
                                tooltip="换人：可以写衣服，如“穿红色西装”；不写就自动按原片的衣服画（每张关键帧穿得一样）。"
                                        "改物：写清楚怎么改，如“把红伞换成透明伞”。中文就行。"),
                io.Float.Input("start_time", default=0.0, min=0.0, max=36000.0, step=0.1, display_name="从第几秒开始"),
                io.Float.Input("seconds", default=30.0, min=3.8, max=30.0, step=0.1, display_name="处理多长（秒，最长 30）",
                               tooltip="视频比这短就处理整段。超过一段（下面的「每段最多帧数」，默认 9.4 秒）会自动分成几段接力生成，"
                                       "30 秒大约 4 段，16G 本地每段约 6～7 分钟。"),
                io.Int.Input("max_frames", default=226, min=90, max=345, step=17, display_name="每段最多帧数（24帧=1秒）",
                             tooltip="16G/12G 本地默认 226 帧（9.4 秒）；云端可到 345 帧（14.4 秒）。必须是 17n+5。"),
                io.Combo.Input("short_edge", options=["480", "736"], default="480", display_name="清晰度（短边）",
                               tooltip="480 = 832×480，本地 16G/12G 用；736 = 1280×736，云端 48G 以上用。"),
            ],
            outputs=[
                io.Image.Output(display_name="source"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="plan"),
                io.Image.Output(display_name="contact_sheet"),
            ],
        )

    @classmethod
    def execute(cls, video, mode, who, target, start_time, seconds, max_frames, short_edge) -> io.NodeOutput:
        se = int(short_edge)
        maxlen = R.valid_frames(max_frames, max_frames)
        src, audio, fps = decode_canvas(video, start_time, seconds, se)
        n = src.shape[0]
        if n < R.MIN_FRAMES:
            raise RuntimeError(f"视频太短：从第 {start_time} 秒起只有 {n / R.FPS:.1f} 秒，至少要 3.8 秒（90 帧）。")
        if n <= maxlen:  # 一段就够：截到 17n+5（最多丢掉尾巴 0.7 秒）；多段时最后一段对齐片尾，不用截
            n = R.valid_frames(n, maxlen)
            src = src[:n]
        raw_cuts, _ = R.detect_cuts(R.small(src))
        shots = R.shots_from_cuts(raw_cuts, n)
        cuts = [s for s, _ in shots[1:]]
        ws = R.windows(cuts, n, maxlen, 3)
        wk = R.window_keyframes(ws, shots, 4)
        kfs, win = [], []
        for w, fs in zip(ws, wk):
            r0, r1 = w["render"]
            segs = [(max(s, r0), min(e, r1)) for s, e in shots if s < r1 and e > r0]
            w = {**w, "kf": [], "shots": [{"start": s - r0, "end": e - r0} for s, e in segs]}
            for f, cands, rng in fs:
                w["kf"].append(len(kfs))
                kfs.append({"frame": f, "range": list(rng), **({"cands": cands} if cands else {})})
            win.append(w)
        plan = {"mode": MODE_KEY[mode], "mode_cn": mode, "who": who.strip(), "target": target.strip(),
                "frames": n, "max_window": max(w["render"][1] - w["render"][0] for w in ws),
                "width": src.shape[2], "height": src.shape[1], "fps_src": fps, "start_time": start_time,
                "shots": [{"start": s, "end": e, "at": R.mmss(s)} for s, e in shots],
                "keyframes": kfs, "windows": win}
        log.info(f"[视频分析] {n} 帧（{n / R.FPS:.2f} 秒）{src.shape[2]}x{src.shape[1]}，源 {fps:.2f}fps；"
                 f"{len(shots)} 个镜头，切镜在 {cuts}；分 {len(ws)} 段：")
        for i, w in enumerate(win):
            log.info(f"  第 {i + 1} 段 渲染 {w['render']} 输出 {w['own']}"
                     + (f" 接力第 {w['prefix_of'] + 1} 段" if "prefix_of" in w else "")
                     + " 关键帧 " + ", ".join(f"{kfs[k]['cands']}（挑一帧）" if "cands" in kfs[k] else str(kfs[k]["frame"])
                                            for k in w["kf"]))
        sheet = contact_sheet(src, shots)
        return io.NodeOutput(src, cut_audio(audio, 0, n), json.dumps(plan, ensure_ascii=False), sheet)


def finalize_plan(P, frames):
    """②画完以后：frames[i] = 第 i 个关键帧最终用的帧号（None = 不画）。给每段算出 keys（painted 里的序号）、
    pins（段内帧号）、keyframes（段内帧号 + 段内第几个镜头），③和⑤按这个用。"""
    kept = sorted({f for f in frames if f is not None})
    idx = {f: k for k, f in enumerate(kept)}
    P = dict(P, keyframes=[{"frame": f, "at": R.mmss(f)} for f in kept])
    wins = []
    for w in P["windows"]:
        r0 = w["render"][0]
        fs = sorted({frames[k] for k in w["kf"] if frames[k] is not None})
        shot = [next(j for j, s in enumerate(w["shots"]) if s["start"] <= f - r0 < s["end"]) + 1 for f in fs]
        wins.append(dict(w, keys=[idx[f] for f in fs], pins=[f - r0 for f in fs],
                         keyframes=[{"frame": f - r0, "shot": j} for f, j in zip(fs, shot)]))
    P["windows"] = wins
    return P


class WZFRemakePaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakePaint", display_name="② 画关键帧（Qwen 改图）", category="王知风/视频改拍",
            description="把每个镜头选出的那一帧改成目标样子。成片里的人长什么样，完全由这几张图决定。"
                        "换人 / 换头时，8B 在每个镜头里挑人脸最大的一帧来画（接力段都看不见脸就不画）。",
            inputs=[
                io.Image.Input("source"),
                io.String.Input("plan", force_input=True),
                io.Image.Input("reference", optional=True, display_name="人物参考图",
                               tooltip="换人 / 换头必须接：要换上去的那个人的照片（最好露脸、背景干净）。改物可以不接。"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             display_name="种子（关键帧不满意就换）"),
                io.Int.Input("steps", default=6, min=4, max=30, display_name="步数", advanced=True),
            ],
            outputs=[io.Image.Output(display_name="painted"), io.Image.Output(display_name="compare"),
                     io.String.Output(display_name="plan")],
        )

    @classmethod
    def execute(cls, source, plan, seed, steps, reference=None) -> io.NodeOutput:
        P = json.loads(plan)
        mode, who, target = P["mode"], P["who"], P["target"]
        if mode in ("whole", "head") and reference is None:
            raise RuntimeError("「换整个人 / 只换头」要接人物参考图：在「人物参考图」那里上传要换上去的人的照片。")
        if mode == "edit" and not target:
            raise RuntimeError("「改物/改场景」要在①里写清楚「改成什么样」，例如：把红伞换成透明伞。")
        swap = mode in ("whole", "head")
        kfs = P["keyframes"]
        relay = any("cands" in k for k in kfs)
        # 8B 先看参考图写一句长相，写进画图提示词（11-14 实测：不写具体长相时，特写镜头里 Qwen 只换了衣服、脸没换）
        vlm = nodes.CLIPLoader().load_clip(qwen_te(), VLM_TYPE, "default")[0] if (swap or relay) else None
        # 接力段的关键帧：8B 从候选帧里挑一帧看得见要改的人 / 东西的（离中间近的优先），都看不见就不画。
        # 先看第一张关键帧写一句英文描述，再对每个候选帧单独问"有没有它"（11-15 实测：两张图对比着问"是不是同一个人"，
        # 8B 全答 NO；单张图按描述问，9 个候选帧全对）
        first = next(k["frame"] for k in kfs if "cands" not in k)
        q, frames, boxes = None, [], {}

        def box(f):  # 8B 框出要换的人的脸（全片帧号 → 相对坐标框或 None），同一帧只问一次
            if f not in boxes:
                boxes[f] = _face_box(vlm, source[f:f + 1], who)
            return boxes[f]

        for k in kfs:
            if swap:
                # 换人：在这个关键帧能挑的范围里均匀看 8 帧，选要换的人脸最大的一帧（官方指南：选人最大、最清楚的帧）。
                # 11-17 实测：一个连续推拉镜头取中间帧落在远景、脸很小，开头结尾的特写没有新脸可参照，成片特写里还是原来的人。
                # 接力段也这样挑；范围里都看不见脸（背影）就不画
                lo, hi = k.get("range", [k["frame"], k["frame"] + 1])
                hi = max(hi, lo + 1)
                pool = {k["frame"], *(k.get("cands") or []), *(lo + (hi - 1 - lo) * j // 7 for j in range(8))}
                pool = sorted(f for f in pool if 0 <= f < source.shape[0])
                best = max(pool, key=lambda f: (_area(box(f)), -abs(f - k["frame"])))
                pick = best if _area(box(best)) > 0 else (None if "cands" in k else k["frame"])
                log.info(f"[画关键帧] 挑脸最大的一帧：第 {pick} 帧（原定第 {k['frame']} 帧，看了 {len(pool)} 帧，"
                         f"脸占画面 {_area(box(best)) * 100:.1f}%）")
                frames.append(pick)
                continue
            if "cands" not in k:
                frames.append(k["frame"])
                continue
            if q is None:
                d = _phrase(_ask(vlm, VIS_DESC["edit"].format(who=who or "the main person", target=target),
                                 [_small(source[first:first + 1])], 60), 30)
                q = VIS_ASK["edit"].format(desc=d or (who or "the main person"))
                log.info(f"[画关键帧] 接力段找：{d}")
            pick = None
            for f in sorted(k["cands"], key=lambda f: abs(f - k["frame"])):
                ans = _ask(vlm, q, [_small(source[f:f + 1])], 8).upper()
                log.info(f"[画关键帧] 接力段候选第 {f} 帧：{ans}")
                if "YES" in ans:
                    pick = f
                    break
            frames.append(pick)
        first = next((f for f in frames if f is not None), first)  # 描述衣服用脸最大、最清楚的那帧
        todo = sorted({f for f in frames if f is not None})
        keyframes = source[torch.tensor(todo)][..., :3]
        look = ""
        if swap:
            look = _phrase(_ask(vlm, PERSON_ASK, [_small(reference)], 80), 30)
            log.info(f"[画关键帧] 参考图里的人：{look}")
        outfit = target
        if mode == "whole" and not target:
            # 用户没写衣服：8B 看第一张关键帧写出原来的衣服，每张关键帧都用这同一句，免得各画各的
            # （11-15 实测：分开画时有一张把黑衬衫画成米色，一镜到底里衬衫中途变色）
            outfit = _phrase(_ask(vlm, CLOTHES_ASK.format(who=who or "the main person"), [_small(source[first:first + 1])], 60), 25)
            log.info(f"[画关键帧] 用户没写衣服，按原片的衣服画：{outfit}")
        if mode == "head":
            prompt, sampler = HEADSWAP, "lcm"
        elif mode == "whole":
            prompt = (f"Replace {who or 'the main person'} in <image1> with the person from <image2>"
                      + (f" ({look})" if look else "") + ": the face, the hair style and the hair colour of <image2>"
                      + (f"; outfit: {outfit}" if outfit else "")
                      + ". Keep exactly the same pose, hand positions, head angle, expression, objects, room, lighting, "
                        "camera angle and composition as <image1>. Photorealistic, sharp details.")
            sampler = "euler"
        else:
            prompt = (f"In <image1>: {target}" + (" (see <image2>)" if reference is not None else "")
                      + ". Keep everything else exactly the same: the people, their faces, hair, clothes and poses, the "
                        "background, lighting, camera angle, framing and composition. Photorealistic, sharp details.")
            sampler = "euler"
        model = nodes.UNETLoader().load_unet(find("diffusion_models", QWEN["unet"], "qwen_image_2.1_bf16.safetensors"), "default")[0]
        loras = [find("loras", QWEN["detail"], optional=True),  # 细节 LoRA 可选：云平台没有就跳过
                 find("loras", QWEN["turbo"], keys=("viggle-turbo", "6step"))]
        for name in loras:
            if name:
                model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, name, 1.0)[0]
        # 换头模型 = 同一个底模 + BFS 换头 LoRA。「只换头」直接用它画；「换整个人」先整个人换掉（衣服、身形、发型），
        # 再用它把头照参考图重画一遍（11-17 实测：只画一遍时脸型沿用原帧那个人，女换男成了"瘦脸版的他"，不够像）
        bfs = find("loras", QWEN["bfs"], keys=("bfs_head_v1.1", "qwen"), optional=mode == "whole") if swap else None
        head = nodes.LoraLoaderModelOnly().load_lora_model_only(model, bfs, 1.0)[0] if bfs else None
        if mode == "head":
            model = head
        model = QwenImage21Cache.execute(model, "auto", "default")[0]
        if mode == "whole" and head is not None:
            head = QwenImage21Cache.execute(head, "auto", "default")[0]
        else:
            head = None
        clip = nodes.CLIPLoader().load_clip(qwen_te(), "qwen_image", "default")[0]
        vae = nodes.VAELoader().load_vae(find("vae", QWEN["vae"]))[0]
        H, W = keyframes.shape[1], keyframes.shape[2]

        def draw(base, m, p, smp, s):
            imgs = {"image_1": base}
            if reference is not None:
                imgs["image_2"] = reference[:1, ..., :3]
            enc = TextEncodeQwenImage21.execute(clip, p, "", vae, 1024, images=imgs)
            samples = nodes.common_ksampler(m, s, steps, 1.0, smp, "simple", enc[0], enc[1], enc[2])[0]
            return nodes.VAEDecode().decode(vae, samples)[0][:1, ..., :3]

        def paint(i, s):
            img = draw(keyframes[i:i + 1], model, prompt, sampler, s)
            if head is not None:  # 第二遍：只重画头（接第一遍的原尺寸结果，不先缩小）
                img = draw(img, head, HEADSWAP, "lcm", s)
            return resize(img, W, H)

        outs = []
        for i in range(keyframes.shape[0]):
            if swap:
                # 自检：脸换掉没有（官方智能体也查 kept_source_face：新头发下面还是原来那张脸）。
                # 只给 8B 一张图（画好的那张裁到只剩脸），让它在"原来的人 / 参考图的人"两句描述里二选一。
                # 11-17 实测女换男：三张图对比着问"是参考图的人还是原来的人"，整图、裁脸都 6 次全答 OLD（画对了也说没换）；
                # 改成单图二选一后，画好的 / 原帧各 2 张、两种选项顺序，8 次全对
                ob = box(todo[i])
                old = _phrase(_ask(vlm, PERSON_ASK, [_small(keyframes[i:i + 1])], 80), 30)
            for t in range(3):
                img = paint(i, seed + i + 1000 * t)
                if not swap or not (old and look):
                    break
                ans = _ask(vlm, PICK_ASK.format(a=old, b=look), [_small(_crop(img, ob) if ob else img)], 4).upper()
                ok = (re.findall(r"\b([AB])\b", ans) or ["B"])[-1] == "B"  # 答不出 A/B 时不重画
                log.info(f"[画关键帧] 第 {i + 1} 张第 {t + 1} 次：自检 {'换掉了' if ok else '还是原来的人，换种子重画'}（{ans}）")
                if ok:
                    break
            outs.append(img)
            log.info(f"[画关键帧] 第 {i + 1}/{keyframes.shape[0]} 张（第 {todo[i]} 帧）完成")
        del model, head, clip, vae, vlm
        free()
        painted = torch.cat(outs)
        compare = torch.cat([keyframes[..., :3], painted], dim=2)
        P = finalize_plan(P, frames)
        for i, w in enumerate(P["windows"]):
            log.info(f"[画关键帧] 第 {i + 1} 段用关键帧 {[todo[k] for k in w['keys']]}")
        return io.NodeOutput(painted, compare, json.dumps(P, ensure_ascii=False))


TO_ASCII = str.maketrans({"—": "-", "–": "-", "‑": "-", "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
                          "，": ", ", "。": ". ", "：": ": ", "；": "; ", "（": "(", "）": ")"})


def _ask(vlm, text, images, n=300):
    imgs = {f"image{k + 1}": im for k, im in enumerate(images)}
    out = TextGenerate.execute(vlm, text, n, {"sampling_mode": "off"}, thinking=False, images=imgs)[0]
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S)
    return re.sub(r"```\w*", "", out).translate(TO_ASCII).strip()


def _phrase(s, limit=60):
    """一句话：去引号、句号、换行，截到 limit 词。"""
    s = re.sub(r"[^\x00-\x7f]", "", s.split("\n")[0]).strip().strip('"').strip("'").rstrip(".").strip()
    return " ".join(s.split()[:limit])


def _small(img, w=640):
    return resize(img[:1, ..., :3], w, max(32, round(w * img.shape[1] / img.shape[2])))


PERSON_ASK = ("Describe the person in this photo in one short English phrase: apparent age and gender, face, hair style and "
              "hair colour. No clothes, no background.")
PICK_ASK = "Which description fits the person in this photo better? A: {a}. B: {b}. Answer A or B."
FACE_BOX = ("Find the face of {who} in this image. Reply only with its bounding box as [x1, y1, x2, y2] in 0-1000 "
            "coordinates. If no face is visible (for example the person is seen from behind), reply NONE.")


def _face_box(vlm, img, who=""):
    """8B 框人脸（Qwen3-VL 给 0-1000 的相对坐标）→ (x1, y1, x2, y2) 0-1，看不见脸返回 None。"""
    a = _ask(vlm, FACE_BOX.format(who=who or "the main person"), [_small(img)], 30)
    n = [int(x) for x in re.findall(r"\d+", a)[:4]]
    if "NONE" in a.upper() or len(n) < 4:
        return None
    x1, y1, x2, y2 = (min(max(v, 0), 1000) / 1000 for v in n)
    return (x1, y1, x2, y2) if x2 - x1 > 0.01 and y2 - y1 > 0.01 else None


def _area(b):
    return 0.0 if b is None else (b[2] - b[0]) * (b[3] - b[1])


def _crop(img, b, pad=0.4):
    """按脸框裁出来（四周各多留 40%）。"""
    H, W = img.shape[1], img.shape[2]
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int((x1 - w * pad) * W)), max(0, int((y1 - h * pad) * H))
    X2, Y2 = min(W, int((x2 + w * pad) * W) + 1), min(H, int((y2 + h * pad) * H) + 1)
    return img[:1, Y1:Y2, X1:X2, :3]


CLOTHES_ASK = ("Describe the clothes of {who} in this video frame in one short English phrase: each garment and its colour, "
               "for example: grey blazer, black shirt, black trousers. Clothes only, no person, no background.")
VIS_DESC = {
    "swap": ("Describe {who} in this video frame in one short English phrase: apparent age and gender, hair, clothes. "
             "No background."),
    "edit": ("In this video frame, the edit to make is: {target}. Describe the thing that this edit changes, as it looks here, "
             "in one short English phrase. No background."),
}
VIS_ASK = {
    "swap": "Is there {desc} in this image, with the face visible (not only seen from behind)? Answer YES or NO.",
    "edit": "Is there {desc} clearly visible in this image? Answer YES or NO.",
}
PAIR_ASK = {
    "swap": ("Image 1 is a frame of a video. Image 2 is the same frame after the person{who} was replaced by a new person. "
             "Answer with exactly two lines and nothing else:\nBEFORE: <a short English phrase for the replaced person as seen in "
             "image 1: apparent age and gender, hair, clothes, position in frame>\nAFTER: <a short English phrase for the new person "
             "in image 2: apparent age and gender, face, hair style and colour, clothes>"),
    "edit": ("Image 1 is a frame of a video. Image 2 is the same frame after this edit: {target}. Answer with exactly two lines "
             "and nothing else:\nBEFORE: <a short English phrase for the thing that changes, as it looks in image 1>\n"
             "AFTER: <a short English phrase for the same thing as it looks in image 2>"),
}
SHOT_ASK = ("The {n} images are the start, middle and end of one shot of a video. In one or two short English sentences in present "
            "tense, describe the framing and camera move, then what happens. Call {who_tag} 'SUBJ'. If someone talks, write "
            "'speaks <d>[unclear]</d>' and never guess words. No time stamps, no lists, no quotes.")
LOOK_ASK = ("This is a contact sheet of a video. In one short English sentence, describe its overall look: realism or style, "
            "lighting, colour and camera style. Never use the word 'dim'.")


def compose(P, before, after, look, shot_lines):
    """一段的六段式提示词，结构全由代码写（照 Viggle 官方示例的句式），8B 只填描述。
    P：mode + 这段的 shots（段内帧号）+ keyframes（段内帧号、第几个镜头）。时间都是段内时间。
    接力段开头那个延续上一段的镜头没有关键帧（官方：开头 17 帧已经定了长相），整段可能一张图都没有，这时只用文字描述新的人。"""
    shots, kfs = P["shots"], P["keyframes"]
    n = len(kfs)
    pic_list = ", ".join(f"<Picture {k + 1}>" for k in range(n))
    as_pics = f", exactly as in {pic_list}" if n else ""
    cuts = [R.mmss(s["start"]) for s in shots[1:]]
    cut_txt = (f"the {len(shots)} shots and the cuts at {', '.join(cuts)}" if cuts else "the single continuous shot with no cut")
    shot_of = {k: kf["shot"] for k, kf in enumerate(kfs)}
    match = {kf["shot"]: k + 1 for k, kf in enumerate(kfs)}
    desc = []
    for k, s in enumerate(shots):
        head = f"[Shot {k + 1}] " + ("" if k == 0 else f"At {R.mmss(s['start'])}, the shot cuts. ")
        line = shot_lines[k] if k < len(shot_lines) else ""
        tail = f" Matching <Picture {match[k + 1]}>." if (k + 1) in match else ""
        desc.append(head + (line or "Everything moves exactly as in <Video 1>.") + tail)
    if P["mode"] == "edit":
        pics = "".join(f"<Picture {k + 1}> is a frame of the target video, exactly as the target video shows it at that moment.\n"
                       for k in range(n))
        return (f"subject_definitions:\n<Video 1> is the source video for the target video edit.\n{pics}"
                f"<Subject 1> is {before} in <Video 1>.\n\n"
                f"summary:\n[video editing] The target video is an edited version of <Video 1> in which <Subject 1> is {after}"
                f"{as_pics}. Everything else, the people, the motion and the timing are exactly as in <Video 1>.\n\n"
                f"retention_analysis:\n<Video 1> (source video editing): fully_preserved - {cut_txt}, the framing and camera "
                f"movement, the people, their faces, clothing and motion, and everything not named are copied frame for frame.\n"
                + "".join(f"<Picture {k + 1}> (target keyframe): fully_preserved - the target video passes through this exact frame.\n"
                          for k in range(n))
                + f"<Subject 1> (appears in every shot where it is visible): changed - in every frame it is {after}; "
                  f"never {before} at any moment.\n\n"
                f"detailed_description:\n{look}.\n" + "\n".join(desc).replace("SUBJ", "the person")
                + f"\nIn every frame <Subject 1> is {after}{as_pics}.\n\n"
                "overall_soundscape:\nThe soundtrack of <Video 1>, unchanged.\n\nnon_diegetic_music:\nN/A\n")
    head_only = P["mode"] == "head"
    if n:
        where = ", ".join(f"<Picture {k + 1}> from [Shot {shot_of[k]}]" for k in range(n))
        keep = " Her or his clothes, body and pose are exactly as painted." if head_only else ""
        defs = (f"{pic_list} {'is a frame' if n == 1 else 'are frames'} of the target video, {where}.\n"
                f"<Subject 1> is {after}, in {pic_list}.{keep}")
    else:
        keep = " Her or his clothes, body and pose are exactly as in <Video 1>." if head_only else ""
        defs = f"<Subject 1> is {after}.{keep}"
    return (f"subject_definitions:\n<Video 1> is the source video for the editing task.\n"
            f"{defs} This person replaces {before} in <Video 1>.\n\n"
            f"summary:\n[video editing + character replacement] The target video is an edited version of <Video 1> in which "
            f"{before} becomes <Subject 1>" + (f", looking exactly as in {pic_list}" if n else f", {after}") + ". All other "
            "people, the place and the soundtrack are unchanged; the shots, cuts, camera, motion and timing of <Video 1> are kept.\n\n"
            f"retention_analysis:\n<Video 1> (source video editing): fully_preserved - {cut_txt}, the camera framing and movement, "
            "the background, the unchanged people, and the motion and timing of every performance.\n"
            + "".join(f"<Picture {k + 1}> (appears in [Shot {shot_of[k]}]): reference - a frame of the target video. The people "
                      f"in it, their faces and clothing, and the background, framing and lighting are taken from <Picture {k + 1}>.\n"
                      for k in range(n))
            + f"<Subject 1> (appears in every shot where the person is visible): fully_preserved - {after} stays identical in "
              "every frame.\n\n"
            f"detailed_description:\n{look}.\n" + "\n".join(desc).replace("SUBJ", "<Subject 1>")
            + "\nIn every frame, from the first to the last, the person is this same <Subject 1>"
            + (f" exactly as in {pic_list}" if n else f", {after}") + f": never {before} at any moment. Only this one person "
              "changes, and everything else stays as in <Video 1>.\n\n"
            "overall_soundscape:\nThe soundtrack of <Video 1>, unchanged.\n\nnon_diegetic_music:\nN/A\n")


class WZFRemakePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakePrompt", display_name="③ 自动写提示词（本地 8B）", category="王知风/视频改拍",
            description="六段式英文提示词的结构（谁换成谁、哪张图对哪个镜头、切镜时间、保留什么）由代码按 Viggle 官方句式写死；"
                        "Qwen3-VL 8B 只看图写描述：换掉的人和新的人长什么样、每个镜头在做什么、整体画面风格。"
                        "「自己写」框里有内容就原样用（专业模式）。分了几段就写几段提示词，段与段之间用单独一行 ===== 隔开。",
            inputs=[
                io.String.Input("plan", force_input=True),
                io.Image.Input("source"),
                io.Image.Input("painted"),
                io.Image.Input("contact_sheet"),
                io.String.Input("manual", multiline=True, default="", display_name="自己写（留空 = 自动）",
                                tooltip="分成几段就写几段，段与段之间用单独一行 ===== 隔开；只写一段就每段都用它。"),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, plan, source, painted, contact_sheet, manual) -> io.NodeOutput:
        P = json.loads(plan)
        ws = P["windows"]
        if manual.strip():
            parts = split_prompts(manual)
            if len(parts) == 1:
                parts = parts * len(ws)
            if len(parts) != len(ws):
                raise RuntimeError(f"「自己写」里有 {len(parts)} 段提示词，但视频分成了 {len(ws)} 段。")
            return io.NodeOutput(SEP.join(parts))
        swap = P["mode"] != "edit"
        vlm = nodes.CLIPLoader().load_clip(qwen_te(), VLM_TYPE, "default")[0]
        try:
            ask = PAIR_ASK["swap" if swap else "edit"].format(
                who=f" ({P['who']})" if P["who"] else "", target=P["target"])
            f0 = P["keyframes"][0]["frame"]
            pair = _ask(vlm, ask, [_small(source[f0:f0 + 1]), _small(painted[0:1])], 160)
            m1, m2 = re.search(r"BEFORE:\s*(.+)", pair), re.search(r"AFTER:\s*(.+)", pair)
            before = _phrase(m1.group(1), 30) if m1 else ""
            after = _phrase(m2.group(1), 40) if m2 else ""
            look = _phrase(_ask(vlm, LOOK_ASK, [contact_sheet[:1]], 80), 40)
            seen, per_win = {}, []
            for W in ws:  # 每段里的每个镜头（长镜头在不同段里是不同的部分，各写各的）
                r0 = W["render"][0]
                lines = []
                for s in W["shots"]:
                    a, e = r0 + s["start"], r0 + s["end"] - 1
                    if (a, e) not in seen:
                        idx = sorted({a, (a + e) // 2, e})
                        txt = _ask(vlm, SHOT_ASK.format(n=len(idx), who_tag=("the person who is replaced" if swap
                                                                              else "the main person")),
                                   [_small(source[i:i + 1], 448) for i in idx], 120)
                        seen[(a, e)] = " ".join(re.sub(r"[^\x00-\x7f]", "", txt).split()[:60])
                    lines.append(seen[(a, e)])
                per_win.append(lines)
        finally:
            del vlm
            free()
        if not before:
            before = "the main person" if swap else "the thing that changes"
        if not after:
            after = "the new person" if swap else P["target"]
        if not look:
            look = "The target video keeps the look, light and camera style of <Video 1>"
        log.info(f"[写提示词] 8B 看图：原来 = {before}；换成 = {after}；风格 = {look}")
        texts = []
        for k, (W, lines) in enumerate(zip(ws, per_win)):
            text = compose({"mode": P["mode"], "shots": W["shots"], "keyframes": W["keyframes"]}, before, after, look, lines)
            text = re.sub(r"\bdimly lit\b|\bdimly\b|\bdim\b", "softly lit", text)
            log.info(f"[写提示词] 第 {k + 1}/{len(ws)} 段 {len(text.split())} 词：\n{text}")
            texts.append(text)
        return io.NodeOutput(SEP.join(texts))


SEP = "\n=====\n"


def split_prompts(text):
    return [p.strip() for p in re.split(r"\n\s*={5,}\s*\n", "\n" + text.strip() + "\n") if p.strip()]


class WZFRemakeModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakeModel", display_name="④ 改拍模型（显存档位）", category="王知风/视频改拍",
            description="按①里选的模式（换人 / 改物）和这里的显存档位，加载对应的 Viggle Remake 模型和官方采样曲线。",
            inputs=[io.String.Input("plan", force_input=True),
                    io.Combo.Input("tier", options=[AUTO] + TIERS, default=AUTO, display_name="显存档位",
                                   tooltip="自动：16G 以上用 int8；12G 左右用 Q8_0（要先装 12G 补充包）并自动多留 8GB 显存；"
                                           "80G 以上且有 bf16 文件就用 bf16 + LoRA。")],
            outputs=[io.Model.Output(display_name="model"), io.Sigmas.Output(display_name="sigmas")],
        )

    @classmethod
    def execute(cls, plan, tier) -> io.NodeOutput:
        P = json.loads(plan)
        kind = "edit" if P["mode"] == "edit" else "swap"
        # .gguf 不在 diffusion_models 列表里（要查 ComfyUI-GGUF 登记的 unet_gguf），以前这里永远找不到 Q8_0
        has_bf16 = (find("diffusion_models", "minimax_h3_fl2va_bf16.safetensors", optional=True)
                    and find("loras", f"viggle_remake_{kind}_comfyui.safetensors", keys=("viggle_remake", kind, "comfyui"),
                             optional=True))
        has_q8 = find("unet_gguf", f"viggle_remake_{kind}-Q8_0.gguf", keys=("viggle_remake", kind, "q8_0"), optional=True)
        win = P.get("max_window", P["frames"])  # 最长的一段（显存只看一段多长，分几段不影响）
        big = min(P["width"], P["height"]) >= 736 and win > 124  # 官方：736 档 int8 跑 226 帧在 48G、80G 上都会爆显存
        if tier == AUTO:
            total = mm.get_total_memory(mm.get_torch_device()) / 1024 ** 3
            if total >= 70 and has_bf16:
                tier = TIERS[2]
            elif big and has_q8:
                tier = TIERS[1]
            elif total >= 15:
                tier = TIERS[0]
            elif has_q8:
                tier = TIERS[1]
            elif win <= 124:
                tier = TIERS[0]  # 官方：12G 用 int8 只能跑 124 帧
            else:
                raise RuntimeError(f"你的显卡约 {total:.0f}GB 显存。请下载「12G 补充包」（viggle_remake_{kind}-Q8_0.gguf）放进 "
                                   "ComfyUI\\models\\diffusion_models\\，或者把①里「每段最多帧数」改成 124（约 5 秒）。")
            log.info(f"[改拍模型] 显卡 {total:.1f}GB，自动选：{tier}")
        # 预留显存：12G 卡用官方 8GB；大卡跑 736 档 GGUF 时官方用 --reserve-vram 20。
        # 这里只记下来，⑤生成时才设、跑完（出错也算）立刻还原，不留给同一个 ComfyUI 里的别的工作流（如 H3 生视频）
        reserve = None
        if tier == TIERS[1]:
            total = mm.get_total_memory(mm.get_torch_device()) / 1024 ** 3
            reserve = (20 * 1024 ** 3) if (big and total >= 24) else RESERVE_12G
        if tier == TIERS[0]:
            model = nodes.UNETLoader().load_unet(find("diffusion_models", f"viggle_remake_{kind}_pruned_int8_convrot.safetensors", keys=("viggle_remake", kind, "int8")),
                                                 "default")[0]
        elif tier == TIERS[1]:
            if "UnetLoaderGGUF" not in nodes.NODE_CLASS_MAPPINGS:
                raise RuntimeError("12G 档要 ComfyUI-GGUF 节点（基础包自带），当前没加载。")
            model = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet(find("unet_gguf", f"viggle_remake_{kind}-Q8_0.gguf", keys=("viggle_remake", kind, "q8_0")))[0]
        else:
            model = nodes.UNETLoader().load_unet(find("diffusion_models", "minimax_h3_fl2va_bf16.safetensors"), "default")[0]
            lora = nodes.NODE_CLASS_MAPPINGS["LoraLoaderBypassModelOnly"]()
            fn = getattr(lora, lora.FUNCTION)
            model = fn(model, find("loras", f"viggle_remake_{kind}_comfyui.safetensors", keys=("viggle_remake", kind, "comfyui")), 1.0)[0]
        model.wzf_reserve_vram = reserve
        return io.NodeOutput(model, torch.FloatTensor(SIGMAS[kind]))


def prefix_latent(vae, latent, prev, at, k=R.P):
    """接力：上一段成片的 [at, at+k) 帧当这一段的开头 k 帧，干净地放进 latent、噪声遮罩设 0，采样时原样保留
    （和 Viggle 的 RemakePrefix 一样；云平台的 Viggle 节点不一定有这个，所以自己写一份）。"""
    px = prev[at:at + k, ..., :3]
    video, audio = latent["samples"].unbind()
    vmask, amask = latent["noise_mask"].unbind()
    n = k // 17 * 5
    assert px.shape[0] == k and tuple(px.shape[1:3]) == (video.shape[3] * 16, video.shape[4] * 16), \
        f"接力帧尺寸不对：{tuple(px.shape)}"
    # VAE 按 17 帧一组编码、丢掉最后 3 个 latent：补到 17m+5，留前 5m 个
    clean = vae.encode(torch.cat([px, px[-1:].repeat(5, 1, 1, 1)]))[:, :, :n]
    video, vmask = video.clone(), vmask.clone()
    video[:, :, :n] = clean.to(video)
    vmask[:, :, :n] = 0
    return {**latent, "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((vmask, amask))}


class WZFRemakeRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakeRender", display_name="⑤ 生成（自动分段接力）", category="王知风/视频改拍",
            description="按①分好的段逐段生成：每段用它自己的关键帧和提示词；接力段开头 17 帧用上一段成片的帧；"
                        "最后拼起来、修切镜接缝的闪帧、配上原声。只有一段时就是普通的一次生成。",
            inputs=[
                io.Model.Input("model"),
                io.Sigmas.Input("sigmas"),
                io.Clip.Input("clip", display_name="H3 文本编码器"),
                io.Vae.Input("vae", display_name="H3 视频 VAE"),
                io.Vae.Input("audio_vae", display_name="H3 音频 VAE"),
                io.Image.Input("source"),
                io.Audio.Input("audio", optional=True),
                io.Image.Input("painted"),
                io.String.Input("plan", force_input=True),
                io.String.Input("prompt", force_input=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             display_name="种子（成片不满意就换）"),
            ],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, model, sigmas, clip, vae, audio_vae, source, painted, plan, prompt, seed, audio=None) -> io.NodeOutput:
        reserve = getattr(model, "wzf_reserve_vram", None)
        old = mm.EXTRA_RESERVED_VRAM
        if reserve is not None:
            mm.EXTRA_RESERVED_VRAM = reserve
            log.info(f"[生成] 本次多留 {reserve / 1024 ** 3:.0f}GB 显存（官方配方），跑完还原")
        try:
            return cls._run(model, sigmas, clip, vae, audio_vae, source, painted, plan, prompt, seed, audio)
        finally:
            mm.EXTRA_RESERVED_VRAM = old

    @classmethod
    def _run(cls, model, sigmas, clip, vae, audio_vae, source, painted, plan, prompt, seed, audio):
        P = json.loads(plan)
        ws, n = P["windows"], P["frames"]
        prompts = split_prompts(prompt)
        if len(prompts) == 1:
            prompts = prompts * len(ws)
        if len(prompts) != len(ws):
            raise RuntimeError(f"提示词有 {len(prompts)} 段，但视频分成了 {len(ws)} 段。")
        V = nodes.NODE_CLASS_MAPPINGS
        if "RemakeConditioning" not in V:
            raise RuntimeError("缺少 Viggle Remake 节点（ComfyUI-Viggle-Remake），基础包里应该自带。")
        cond, noise_node, until = V["RemakeConditioning"](), V["RemakeNoise"](), V["RemakeKeyframeUntil"]()
        se = min(source.shape[1], source.shape[2])
        # 1) 先把每段的条件编码好：32B 文本编码器只在这里用，之后整段时间显卡留给主模型
        conds = []
        for i, w in enumerate(ws):
            r0, r1 = w["render"]
            refs = [painted[k:k + 1] for k in w["keys"]] + [None] * (4 - len(w["keys"]))
            conds.append(cond.run(clip, vae, audio_vae, source[r0:r1], prompts[i], refs[0], " ".join(map(str, w["pins"])),
                                  r1 - r0, se, cut_audio(audio, r0, r1), *refs[1:]))
            log.info(f"[生成] 第 {i + 1}/{len(ws)} 段条件编码完成（{r1 - r0} 帧，{len(w['keys'])} 张关键帧）")
        # 2) 逐段采样、解码；接力段等上一段出来再做
        m = until.run(model, 0.9)[0]
        sampler = comfy.samplers.sampler_object("euler")
        out, prev = None, None
        for i, (w, (positive, latent)) in enumerate(zip(ws, conds)):
            mm.throw_exception_if_processing_interrupted()
            r0, r1 = w["render"]
            a0, z0 = w["own"]
            if "prefix_of" in w:
                latent = prefix_latent(vae, latent, prev, w["prefix_at"])
            guider = Guider_Basic(m)
            guider.set_conds(positive)
            noise = noise_node.run(positive, seed)[0]
            callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
            samples = guider.sample(noise.generate_noise(latent), latent["samples"], sampler, sigmas,
                                    denoise_mask=latent["noise_mask"], callback=callback,
                                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
            take = nodes.VAEDecode().decode(vae, {"samples": samples.to(mm.intermediate_device())})[0][:r1 - r0, ..., :3]
            if out is None:
                out = torch.empty((n, *take.shape[1:]), dtype=take.dtype)
            out[a0:z0] = take[a0 - r0:z0 - r0]
            prev = take
            log.info(f"[生成] 第 {i + 1}/{len(ws)} 段完成：渲染 {w['render']}，输出 {w['own']}"
                     + (f"（接力第 {w['prefix_of'] + 1} 段）" if "prefix_of" in w else ""))
        del prev, conds
        cuts = [s["start"] for s in P["shots"][1:]]
        src = source[:n, ..., :3]
        if src.shape[1:3] != out.shape[1:3]:
            src = resize(src, out.shape[2], out.shape[1])
        fixed = R.deflash(out, src, ws, cuts)
        if fixed:
            log.info(f"[生成] 切镜接缝修了闪帧：{fixed}")
        return io.NodeOutput(out, cut_audio(audio, 0, n))


class WZFRemakeCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakeCompare", display_name="⑥ 左右对比（原片 | 成片）", category="王知风/视频改拍",
            inputs=[io.Image.Input("source"), io.Image.Input("result")],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, source, result) -> io.NodeOutput:
        n = min(source.shape[0], result.shape[0])
        src = source[:n, ..., :3]
        if src.shape[1:3] != result.shape[1:3]:
            src = resize(src, result.shape[2], result.shape[1])
        return io.NodeOutput(torch.cat([src, result[:n, ..., :3]], dim=2))


class WZFVideoRemakeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [WZFRemakePlan, WZFRemakePaint, WZFRemakePrompt, WZFRemakeModel, WZFRemakeRender, WZFRemakeCompare]


async def comfy_entrypoint() -> WZFVideoRemakeExtension:
    return WZFVideoRemakeExtension()
