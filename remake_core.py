"""王知风 · 视频改拍（Viggle Remake）核心逻辑，不依赖 ComfyUI，方便离线核对。

- 帧数：Viggle Remake 每段 17n+5 帧，90～345，24fps。
- 切镜：缩到 64x36 后看相邻帧的像素差 + 颜色直方图差，取局部最大且明显高于周围的点。
- 关键帧：每个镜头一张，取镜头中间；镜头超过 4 个时取最长的 4 个（官方智能体规则：先保证镜头最长的几个）。
- 时间戳：MM:SS.mmm（官方六段式提示词的写法）。
- 长视频：照 Viggle 官方的分段办法切成几段，段尽量断在切镜处；一个镜头比一段还长时，下一段接着上一段成片的最后
  17 帧继续（接力）；最后把每段自己负责的帧拼起来，切镜接缝处修 1～2 帧闪帧。
"""
import math

import torch
import torch.nn.functional as F

FPS = 24
MIN_FRAMES, MAX_FRAMES = 90, 345


def resample_indices(n_src, fps_src, fps_dst=FPS):
    """按真实时间把任意帧率映射到 24fps：第 i 帧取源视频 round(i/24*fps_src)。"""
    if not fps_src or abs(fps_src - fps_dst) < 0.01:
        return list(range(n_src))
    n_dst = int(math.floor((n_src - 1) / fps_src * fps_dst)) + 1
    return [min(n_src - 1, int(round(i / fps_dst * fps_src))) for i in range(n_dst)]


def valid_frames(n, cap):
    """不超过 n 和 cap 的最大 17k+5；不足 90 帧返回 0。"""
    m = min(n, cap, MAX_FRAMES)
    if m < MIN_FRAMES:
        return 0
    return (m - 5) // 17 * 17 + 5


def small(images, w=64, h=36):
    """[N,H,W,C] 0~1 → [N,3,h,w]，分批缩小，省内存。"""
    out = []
    for i in range(0, images.shape[0], 64):
        x = images[i:i + 64, ..., :3].movedim(-1, 1).float()
        out.append(F.interpolate(x, size=(h, w), mode="area"))
    return torch.cat(out)


def cut_scores(sm):
    """相邻帧差异：像素平均绝对差（0~1）和颜色直方图差（0~1），相加。sm: [N,3,h,w]。"""
    n = sm.shape[0]
    if n < 2:
        return torch.zeros(n)
    pix = (sm[1:] - sm[:-1]).abs().mean(dim=(1, 2, 3))
    bins = 16
    q = (sm.clamp(0, 1) * (bins - 1)).round().long()  # [N,3,h,w]
    hist = torch.zeros(n, 3, bins)
    for c in range(3):
        hist[:, c] = F.one_hot(q[:, c].flatten(1), bins).sum(1).float()
    hist = hist / hist.sum(-1, keepdim=True)
    hd = 0.5 * (hist[1:] - hist[:-1]).abs().sum(-1).mean(-1)
    return torch.cat([torch.zeros(1), pix + hd])  # score[i] = 第 i 帧和第 i-1 帧的差


def detect_cuts(sm, min_shot=8, ratio=4.0, floor=0.18):
    """返回新镜头起始帧列表。条件：分数是前后 ±3 帧内的最大值，比周围中位数高 ratio 倍，且超过绝对下限 floor。"""
    s = cut_scores(sm)
    n = s.shape[0]
    cuts = []
    for i in range(1, n):
        lo, hi = max(1, i - 3), min(n, i + 4)
        if s[i] < s[lo:hi].max():
            continue
        ctx = torch.cat([s[max(1, i - 12):i], s[i + 1:i + 13]])
        base = float(ctx.median()) if ctx.numel() else 0.0
        if s[i] >= floor and s[i] >= ratio * max(base, 0.01):
            if not cuts or i - cuts[-1] >= min_shot:
                cuts.append(i)
    return cuts, s


def shots_from_cuts(cuts, nf, stub=6):
    """[(start, end)]，end 不含。短于 stub 帧的头尾碎镜头并到相邻镜头。"""
    edges = [0] + [c for c in cuts if 0 < c < nf] + [nf]
    shots = [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]
    merged = []
    for s in shots:
        if merged and s[1] - s[0] < stub:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < stub:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged


def pick_keyframes(shots, max_refs=4):
    """每个镜头取中间一帧（离镜头边缘至少 6 帧）；超过 max_refs 个镜头时取最长的几个，按时间排序。"""
    order = sorted(range(len(shots)), key=lambda k: shots[k][1] - shots[k][0], reverse=True)[:max_refs]
    out = []
    for k in sorted(order):
        s, e = shots[k]
        mid = (s + e) // 2
        mid = min(max(mid, s + min(6, (e - s) // 2)), e - 1 - min(6, (e - s) // 2))
        out.append((mid, k))
    return out


def mmss(frame):
    t = frame / FPS
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


# ---------------- 长视频分段（移植自 Viggle Remake mcp/viggle_remake_mcp/kit.py，Apache-2.0，见 NOTICE.md） ----------------
LENS = [17 * k + 5 for k in range(5, 21)]  # 90..345
P = 17  # 接力段的前缀：上一段成片的最后 17 帧（VAE 的一个片段）


def windows(cuts, n, maxlen, maxcuts=3):
    """切窗口：尽量在切镜点结束，每段 17k+5 帧、不超过 maxlen，一段里最多 maxcuts 个切镜（切镜太多模型会丢镜头、乱序）。
    一个镜头比一段还长时不硬切（接缝处脸会跳），下一段做"接力"：开头 P 帧是上一段成片的帧（prefix_of/prefix_at）。
    最后一段对齐到片尾。render = 这段实际渲染的帧范围，own = 这段负责输出的帧范围。"""
    ls = [x for x in LENS if x <= maxlen]
    rnd = lambda m: min(x for x in ls if x >= m)  # noqa: E731
    out, s, p = [], 0, 0
    while True:
        r = s - p
        cand = [e for e in cuts if r + ls[0] <= e <= r + ls[-1] and s < e < n and r + rnd(e - r) <= n]
        if n - r <= ls[-1] and (not cand or sum(s < c < n for c in cuts) <= maxcuts):
            break
        ok = [e for e in cand if sum(r < c < r + rnd(e - r) for c in cuts) <= maxcuts]
        e = max(ok) if ok else min(cand) if cand else r + ls[-1]
        out.append({"render": [r, r + rnd(e - r)], "own": [s, e],
                    **({"prefix_of": len(out) - 1, "prefix_at": r - out[-1]["render"][0]} if p else {})})
        s, p = e, 0 if e in cuts else P
    ln = rnd(max(n - s + p, ls[0]))
    if p:
        s = out[-1]["own"][1] = n - ln + P
        out.append({"render": [n - ln, n], "own": [s, n], "prefix_of": len(out) - 1, "prefix_at": n - ln - out[-1]["render"][0]})
    else:
        out.append({"render": [n - ln, n], "own": [s, n]})
    return out


def window_keyframes(ws, shots, max_refs=4):
    """每段给它负责的每个镜头选一帧关键帧（镜头中间，离边缘至少 6 帧），最多 max_refs 个（取段内最长的）。
    接力段开头那个延续上一段的镜头：官方默认不给关键帧（上一段的结尾已经定了长相），但脸在接力之后才转过来
    （背影 / 侧脸 → 正脸）时会变回原来的人（11-15 实测：30 秒一镜到底第 3 段从背影接力，转正脸后成了原演员；
    官方指南也写了这种情况要补一帧）。所以给它列 3 个候选帧，由②让 8B 挑一帧看得见脸的来画，都看不见就不画。
    返回每段 [(帧号, 候选帧列表或 None, (lo, hi)), ...]（全片帧号，按帧号排）。(lo, hi) 是这个关键帧可以挑的范围
    （镜头在这段里的部分，去掉两头各 6 帧）：换人时②在这个范围里挑人脸最大的一帧（官方指南：选人最大、最清楚的帧）。"""
    res = []
    for w in ws:
        a0, z0 = w["own"]
        r0, r1 = w["render"]
        cand = []
        for s, e in shots:
            lo, hi = max(s, a0, r0), min(e, z0, r1)
            if hi - lo < 1:
                continue
            m = min(6, (hi - lo) // 2)
            mid = min(max((lo + hi) // 2, lo + m), hi - 1 - m)
            relay = "prefix_of" in w and s < a0
            q = (hi - lo) // 4
            cand.append((hi - lo, mid, sorted({lo + q, mid, hi - 1 - q}) if relay else None, (lo + m, hi - m)))
        keep = sorted(cand, key=lambda t: -t[0])[:max_refs]
        res.append(sorted(((f, c, r) for _, f, c, r in keep), key=lambda t: t[0]))
    return res


def deflash(v, src, ws, cuts):
    """段从切镜点开始时，模型可能把切镜放偏 1~2 帧，成片在接缝处闪一下上个镜头。切镜点前后 2 帧里，
    哪一帧站错了边就用相邻帧覆盖。v、src: [N,H,W,C]，原地修改，返回改过的帧号。"""
    cuts = set(cuts)

    def sides(x):
        x = x[:, ::4, ::4, :3].float()
        return [bool((x[k] - x[0]).abs().mean() < (x[k] - x[7]).abs().mean()) for k in range(2, 6)]

    todo = []
    for w in ws[1:]:
        b = w["own"][0]
        if b not in cuts or b < 4 or b + 4 > v.shape[0] or any(0 < abs(b - c) < 8 for c in cuts):
            continue
        want, got = sides(src[b - 4:b + 4]), sides(v[b - 4:b + 4])
        fix = {}
        for k in range(4):
            if want[k] and not got[k]:
                fix[b - 2 + k] = -1
        for k in range(3, -1, -1):
            if not want[k] and got[k]:
                fix[b - 2 + k] = 1
        todo += sorted(fix.items(), key=lambda t: -t[0] * t[1])
    for f, s in todo:
        v[f] = v[f + s]
    return [f for f, _ in todo]
