"""王知风 · 视频改拍（基于 Viggle Remake）

①视频分析：截取、转 24fps、按显存档位定帧数（17n+5）、缩放到 Viggle 画布、自动找切镜、每个镜头选一帧关键帧、出缩略图。
②画关键帧：本地 Qwen-Image 2.1（换整个人 / 只换头用 BFS 换头 LoRA / 改物），画好的帧决定成片里人长什么样（实测：
  只给头像不画帧，脸不会换）。8B 先描述参考照片里的人写进提示词，画完自检脸换掉没有，没换掉换种子重画。
③自动写提示词：六段式结构由代码按 Viggle 官方句式写死，本地 Qwen3-VL 8B 只看图写短描述（实测：整段交给 8B 写，
  会把原来的人和新人写成两个角色）。句式改编自 Viggle Remake 的 mcp/（Apache-2.0，见 data/ 和 NOTICE.md）。
④改拍模型：显存档位默认「自动」（16G int8 / 12G GGUF Q8_0 + 运行时多留 8GB 显存 / 80G+ bf16+LoRA）× 模式（换人 / 改物），
  同时给出官方采样曲线。
⑤左右对比：原片 | 成片，拼成一个画面。

MiniMax H3 · Powered by Viggle Remake (https://huggingface.co/Viggle/Viggle-Remake)
"""
import gc
import json
import logging
import os
import re

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import ComfyExtension, io
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
_DEFAULT_RESERVE = mm.EXTRA_RESERVED_VRAM
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


def find(folder, name):
    for n in folder_paths.get_filename_list(folder):
        if n.replace("\\", "/").split("/")[-1] == name:
            return n
    raise RuntimeError(f"缺少模型：ComfyUI/models/{folder}/{name}\n请按使用说明下载后放进对应文件夹。")


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


class WZFRemakePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakePlan", display_name="① 视频分析（切镜 · 帧数 · 关键帧）", category="王知风/视频改拍",
            description="上传的视频 → 24fps、按显存档位截到能跑的长度、缩放到画布、自动找切镜，每个镜头选一帧关键帧。",
            inputs=[
                io.Video.Input("video", display_name="原视频"),
                io.Combo.Input("mode", options=MODES, default=MODES[0], display_name="改什么"),
                io.String.Input("who", default="", display_name="换谁（多人时写清楚）",
                                tooltip="例：穿灰色外套的女人 / 左边的男人。画面里只有一个人可以不填。改物时不用填。"),
                io.String.Input("target", multiline=True, default="", display_name="改成什么样",
                                tooltip="换人：可以补充衣服，如“穿红色西装”。改物：写清楚怎么改，如“把红伞换成透明伞”。中文就行。"),
                io.Float.Input("start_time", default=0.0, min=0.0, max=36000.0, step=0.1, display_name="从第几秒开始"),
                io.Int.Input("max_frames", default=226, min=90, max=345, step=17, display_name="最多帧数（24帧=1秒）",
                             tooltip="16G/12G 本地默认 226 帧（9.4 秒）；云端可到 345 帧（14.4 秒）。必须是 17n+5。"),
                io.Combo.Input("short_edge", options=["480", "736"], default="480", display_name="清晰度（短边）",
                               tooltip="480 = 832×480，本地 16G/12G 用；736 = 1280×736，云端 48G 以上用。"),
            ],
            outputs=[
                io.Image.Output(display_name="source"),
                io.Audio.Output(display_name="audio"),
                io.Image.Output(display_name="keyframes"),
                io.String.Output(display_name="pins"),
                io.Int.Output(display_name="frames"),
                io.Int.Output(display_name="short_edge"),
                io.String.Output(display_name="plan"),
                io.Image.Output(display_name="contact_sheet"),
            ],
        )

    @classmethod
    def execute(cls, video, mode, who, target, start_time, max_frames, short_edge) -> io.NodeOutput:
        se = int(short_edge)
        dur = max_frames / R.FPS + 0.5
        # 先按时间截取再解码：长视频整段解码会占十几 GB 内存
        clip = video.as_trimmed(start_time, dur, strict_duration=False) or video
        comp = clip.get_components()
        imgs, fps, audio = comp.images, float(comp.frame_rate), comp.audio
        idx = R.resample_indices(imgs.shape[0], fps)
        nf = R.valid_frames(len(idx), max_frames)
        if nf == 0:
            raise RuntimeError(f"视频太短：从第 {start_time} 秒起只有 {len(idx) / R.FPS:.1f} 秒，至少要 3.8 秒（90 帧）。")
        src = fit_canvas(imgs[torch.tensor(idx[:nf])], se)
        del imgs, comp
        cuts, _ = R.detect_cuts(R.small(src))
        shots = R.shots_from_cuts(cuts, nf)
        kfs = R.pick_keyframes(shots, 4)
        keyframes = src[torch.tensor([f for f, _ in kfs])]
        pins = ", ".join(str(f) for f, _ in kfs)
        plan = {"mode": MODE_KEY[mode], "mode_cn": mode, "who": who.strip(), "target": target.strip(),
                "frames": nf, "width": src.shape[2], "height": src.shape[1], "fps_src": fps, "start_time": start_time,
                "shots": [{"start": s, "end": e, "at": R.mmss(s)} for s, e in shots],
                "keyframes": [{"frame": f, "shot": k + 1, "at": R.mmss(f)} for f, k in kfs]}
        log.info(f"[视频分析] {nf} 帧（{nf / R.FPS:.2f} 秒）{src.shape[2]}x{src.shape[1]}，源 {fps:.2f}fps；"
                 f"{len(shots)} 个镜头，切镜在 {cuts}；关键帧 {pins}")
        sheet = contact_sheet(src, shots)
        return io.NodeOutput(src, audio, keyframes, pins, nf, se, json.dumps(plan, ensure_ascii=False), sheet)


class WZFRemakePaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakePaint", display_name="② 画关键帧（Qwen 改图）", category="王知风/视频改拍",
            description="把每个镜头选出的那一帧改成目标样子。成片里的人长什么样，完全由这几张图决定。",
            inputs=[
                io.Image.Input("keyframes"),
                io.String.Input("plan", force_input=True),
                io.Image.Input("reference", optional=True, display_name="人物参考图",
                               tooltip="换人 / 换头必须接：要换上去的那个人的照片（最好露脸、背景干净）。改物可以不接。"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             display_name="种子（关键帧不满意就换）"),
                io.Int.Input("steps", default=6, min=4, max=30, display_name="步数", advanced=True),
            ],
            outputs=[io.Image.Output(display_name="ref_1"), io.Image.Output(display_name="ref_2"),
                     io.Image.Output(display_name="ref_3"), io.Image.Output(display_name="ref_4"),
                     io.Image.Output(display_name="painted"), io.Image.Output(display_name="compare")],
        )

    @classmethod
    def execute(cls, keyframes, plan, seed, steps, reference=None) -> io.NodeOutput:
        P = json.loads(plan)
        mode, who, target = P["mode"], P["who"], P["target"]
        if mode in ("whole", "head") and reference is None:
            raise RuntimeError("「换整个人 / 只换头」要接人物参考图：在「人物参考图」那里上传要换上去的人的照片。")
        if mode == "edit" and not target:
            raise RuntimeError("「改物/改场景」要在①里写清楚「改成什么样」，例如：把红伞换成透明伞。")
        swap = mode in ("whole", "head")
        # 8B 先看参考图写一句长相，写进画图提示词（11-14 实测：不写具体长相时，特写镜头里 Qwen 只换了衣服、脸没换）
        vlm = nodes.CLIPLoader().load_clip(find("text_encoders", QWEN["te"]), VLM_TYPE, "default")[0] if swap else None
        look = ""
        if swap:
            look = _phrase(_ask(vlm, "Describe the person in this photo in one short English phrase: apparent age and gender, "
                                     "face, hair style and hair colour. No clothes, no background.", [_small(reference)], 80), 30)
            log.info(f"[画关键帧] 参考图里的人：{look}")
        if mode == "head":
            prompt, sampler = HEADSWAP, "lcm"
        elif mode == "whole":
            prompt = (f"Replace {who or 'the main person'} in <image1> with the person from <image2>"
                      + (f" ({look})" if look else "") + ": the face, the hair style and the hair colour of <image2>"
                      + (f"; outfit: {target}" if target else "")
                      + ". Keep exactly the same pose, hand positions, head angle, expression, objects, room, lighting, "
                        "camera angle and composition as <image1>. Photorealistic, sharp details.")
            sampler = "euler"
        else:
            prompt = (f"In <image1>: {target}" + (" (see <image2>)" if reference is not None else "")
                      + ". Keep everything else exactly the same: the people, their faces, hair, clothes and poses, the "
                        "background, lighting, camera angle, framing and composition. Photorealistic, sharp details.")
            sampler = "euler"
        model = nodes.UNETLoader().load_unet(find("diffusion_models", QWEN["unet"]), "default")[0]
        for k in ("detail", "turbo") + (("bfs",) if mode == "head" else ()):
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, find("loras", QWEN[k]), 1.0)[0]
        model = QwenImage21Cache.execute(model, "auto", "default")[0]
        clip = nodes.CLIPLoader().load_clip(find("text_encoders", QWEN["te"]), "qwen_image", "default")[0]
        vae = nodes.VAELoader().load_vae(find("vae", QWEN["vae"]))[0]
        H, W = keyframes.shape[1], keyframes.shape[2]

        def paint(i, s):
            imgs = {"image_1": keyframes[i:i + 1]}
            if reference is not None:
                imgs["image_2"] = reference[:1, ..., :3]
            enc = TextEncodeQwenImage21.execute(clip, prompt, "", vae, 1024, images=imgs)
            samples = nodes.common_ksampler(model, s, steps, 1.0, sampler, "simple", enc[0], enc[1], enc[2])[0]
            return resize(nodes.VAEDecode().decode(vae, samples)[0][:1, ..., :3], W, H)

        outs = []
        for i in range(keyframes.shape[0]):
            for t in range(3):
                img = paint(i, seed + i + 1000 * t)
                if not swap:
                    break
                # 自检：脸换掉没有（官方智能体也查 kept_source_face：新头发下面还是原来那张脸）
                ans = _ask(vlm, "Image 1 is a video frame. Image 2 is the same frame after editing. Image 3 is a reference "
                                "photo. Look only at the main person's face and hair in image 2. Is it the person from image 3 "
                                "(answer NEW) or still the person from image 1 (answer OLD)? Answer one word.",
                           [_small(keyframes[i:i + 1]), _small(img), _small(reference)], 8).upper()
                ok = "OLD" not in ans
                log.info(f"[画关键帧] 第 {i + 1} 张第 {t + 1} 次：自检 {'换掉了' if ok else '还是原来的人，换种子重画'}（{ans}）")
                if ok:
                    break
            outs.append(img)
            log.info(f"[画关键帧] 第 {i + 1}/{keyframes.shape[0]} 张（第 {P['keyframes'][i]['frame']} 帧）完成")
        del model, clip, vae, vlm
        free()
        painted = torch.cat(outs)
        refs = [painted[i:i + 1] if i < painted.shape[0] else None for i in range(4)]
        compare = torch.cat([keyframes[..., :3], painted], dim=2)
        return io.NodeOutput(*refs, painted, compare)


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
    """六段式提示词，结构全由代码写（照 Viggle 官方示例的句式），8B 只填描述。"""
    shots, kfs = P["shots"], P["keyframes"]
    n = len(kfs)
    pic_list = ", ".join(f"<Picture {k + 1}>" for k in range(n))
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
        pics = "\n".join(f"<Picture {k + 1}> is a frame of the target video, exactly as the target video shows it at that moment."
                         for k in range(n))
        return (f"subject_definitions:\n<Video 1> is the source video for the target video edit.\n{pics}\n"
                f"<Subject 1> is {before} in <Video 1>.\n\n"
                f"summary:\n[video editing] The target video is an edited version of <Video 1> in which <Subject 1> is {after}, "
                f"exactly as in {pic_list}. Everything else, the people, the motion and the timing are exactly as in <Video 1>.\n\n"
                f"retention_analysis:\n<Video 1> (source video editing): fully_preserved - {cut_txt}, the framing and camera "
                f"movement, the people, their faces, clothing and motion, and everything not named are copied frame for frame.\n"
                + "\n".join(f"<Picture {k + 1}> (target keyframe): fully_preserved - the target video passes through this exact frame."
                            for k in range(n))
                + f"\n<Subject 1> (appears in every shot where it is visible): changed - in every frame it is {after}; "
                  f"never {before} at any moment.\n\n"
                f"detailed_description:\n{look}.\n" + "\n".join(desc).replace("SUBJ", "the person")
                + f"\nIn every frame <Subject 1> is {after}, exactly as in {pic_list}.\n\n"
                "overall_soundscape:\nThe soundtrack of <Video 1>, unchanged.\n\nnon_diegetic_music:\nN/A\n")
    where = ", ".join(f"<Picture {k + 1}> from [Shot {shot_of[k]}]" for k in range(n))
    head_only = P["mode"] == "head"
    keep = " Her or his clothes, body and pose are exactly as painted." if head_only else ""
    return (f"subject_definitions:\n<Video 1> is the source video for the editing task.\n"
            f"{pic_list} {'is a frame' if n == 1 else 'are frames'} of the target video, {where}.\n"
            f"<Subject 1> is {after}, in {pic_list}.{keep} This person replaces {before} in <Video 1>.\n\n"
            f"summary:\n[video editing + character replacement] The target video is an edited version of <Video 1> in which "
            f"{before} becomes <Subject 1>, looking exactly as in {pic_list}. All other people, the place and the soundtrack are "
            f"unchanged; the shots, cuts, camera, motion and timing of <Video 1> are kept.\n\n"
            f"retention_analysis:\n<Video 1> (source video editing): fully_preserved - {cut_txt}, the camera framing and movement, "
            "the background, the unchanged people, and the motion and timing of every performance.\n"
            + "\n".join(f"<Picture {k + 1}> (appears in [Shot {shot_of[k]}]): reference - a frame of the target video. The people "
                        f"in it, their faces and clothing, and the background, framing and lighting are taken from <Picture {k + 1}>."
                        for k in range(n))
            + f"\n<Subject 1> (appears in every shot where the person is visible): fully_preserved - {after} stays identical in "
              "every frame.\n\n"
            f"detailed_description:\n{look}.\n" + "\n".join(desc).replace("SUBJ", "<Subject 1>")
            + f"\nIn every frame, from the first to the last, the person is this same <Subject 1> exactly as in {pic_list}: "
              f"never {before} at any moment. Only this one person changes, and everything else stays as in <Video 1>.\n\n"
            "overall_soundscape:\nThe soundtrack of <Video 1>, unchanged.\n\nnon_diegetic_music:\nN/A\n")


class WZFRemakePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakePrompt", display_name="③ 自动写提示词（本地 8B）", category="王知风/视频改拍",
            description="六段式英文提示词的结构（谁换成谁、哪张图对哪个镜头、切镜时间、保留什么）由代码按 Viggle 官方句式写死；"
                        "Qwen3-VL 8B 只看图写描述：换掉的人和新的人长什么样、每个镜头在做什么、整体画面风格。"
                        "「自己写」框里有内容就原样用（专业模式）。",
            inputs=[
                io.String.Input("plan", force_input=True),
                io.Image.Input("source"),
                io.Image.Input("keyframes"),
                io.Image.Input("painted"),
                io.Image.Input("contact_sheet"),
                io.String.Input("manual", multiline=True, default="", display_name="自己写（留空 = 自动）"),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, plan, source, keyframes, painted, contact_sheet, manual) -> io.NodeOutput:
        if manual.strip():
            return io.NodeOutput(manual.strip())
        P = json.loads(plan)
        swap = P["mode"] != "edit"
        vlm = nodes.CLIPLoader().load_clip(find("text_encoders", QWEN["te"]), VLM_TYPE, "default")[0]
        try:
            ask = PAIR_ASK["swap" if swap else "edit"].format(
                who=f" ({P['who']})" if P["who"] else "", target=P["target"])
            pair = _ask(vlm, ask, [_small(keyframes[0:1]), _small(painted[0:1])], 160)
            m1, m2 = re.search(r"BEFORE:\s*(.+)", pair), re.search(r"AFTER:\s*(.+)", pair)
            before = _phrase(m1.group(1), 30) if m1 else ""
            after = _phrase(m2.group(1), 40) if m2 else ""
            look = _phrase(_ask(vlm, LOOK_ASK, [contact_sheet[:1]], 80), 40)
            lines = []
            for s in P["shots"]:
                a, e = s["start"], s["end"] - 1
                idx = sorted({a, (a + e) // 2, e})
                txt = _ask(vlm, SHOT_ASK.format(n=len(idx), who_tag=("the person who is replaced" if swap else "the main person")),
                           [_small(source[i:i + 1], 448) for i in idx], 120)
                lines.append(" ".join(re.sub(r"[^\x00-\x7f]", "", txt).split()[:60]))
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
        text = compose(P, before, after, look, lines)
        text = re.sub(r"\bdimly lit\b|\bdimly\b|\bdim\b", "softly lit", text)
        log.info(f"[写提示词] {len(text.split())} 词：\n{text}")
        return io.NodeOutput(text)


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
        have = lambda n: any(x.replace("\\", "/").split("/")[-1] == n for x in folder_paths.get_filename_list("diffusion_models"))  # noqa: E731
        if tier == AUTO:
            total = mm.get_total_memory(mm.get_torch_device()) / 1024 ** 3
            if total >= 70 and have("minimax_h3_fl2va_bf16.safetensors"):
                tier = TIERS[2]
            elif total >= 15:
                tier = TIERS[0]
            elif have(f"viggle_remake_{kind}-Q8_0.gguf"):
                tier = TIERS[1]
            elif P["frames"] <= 124:
                tier = TIERS[0]  # 官方：12G 用 int8 只能跑 124 帧
            else:
                raise RuntimeError(f"你的显卡约 {total:.0f}GB 显存。请下载「12G 补充包」（viggle_remake_{kind}-Q8_0.gguf）放进 "
                                   "ComfyUI\\models\\diffusion_models\\，或者把①里「最多帧数」改成 124（约 5 秒）。")
            log.info(f"[改拍模型] 显卡 {total:.1f}GB，自动选：{tier}")
        mm.EXTRA_RESERVED_VRAM = RESERVE_12G if tier == TIERS[1] else _DEFAULT_RESERVE
        if tier == TIERS[0]:
            model = nodes.UNETLoader().load_unet(find("diffusion_models", f"viggle_remake_{kind}_pruned_int8_convrot.safetensors"),
                                                 "default")[0]
        elif tier == TIERS[1]:
            if "UnetLoaderGGUF" not in nodes.NODE_CLASS_MAPPINGS:
                raise RuntimeError("12G 档要 ComfyUI-GGUF 节点（基础包自带），当前没加载。")
            model = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet(find("diffusion_models", f"viggle_remake_{kind}-Q8_0.gguf"))[0]
        else:
            model = nodes.UNETLoader().load_unet(find("diffusion_models", "minimax_h3_fl2va_bf16.safetensors"), "default")[0]
            lora = nodes.NODE_CLASS_MAPPINGS["LoraLoaderBypassModelOnly"]()
            fn = getattr(lora, lora.FUNCTION)
            model = fn(model, find("loras", f"viggle_remake_{kind}_comfyui.safetensors"), 1.0)[0]
        return io.NodeOutput(model, torch.FloatTensor(SIGMAS[kind]))


class WZFRemakeCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WZFRemakeCompare", display_name="⑤ 左右对比（原片 | 成片）", category="王知风/视频改拍",
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
        return [WZFRemakePlan, WZFRemakePaint, WZFRemakePrompt, WZFRemakeModel, WZFRemakeCompare]


async def comfy_entrypoint() -> WZFVideoRemakeExtension:
    return WZFVideoRemakeExtension()
