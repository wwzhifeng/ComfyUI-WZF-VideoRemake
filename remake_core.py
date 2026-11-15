"""王知风 · 视频改拍（Viggle Remake）核心逻辑，不依赖 ComfyUI，方便离线核对。

- 帧数：Viggle Remake 每段 17n+5 帧，90～345，24fps。
- 切镜：缩到 64x36 后看相邻帧的像素差 + 颜色直方图差，取局部最大且明显高于周围的点。
- 关键帧：每个镜头一张，取镜头中间；镜头超过 4 个时取最长的 4 个（官方智能体规则：先保证镜头最长的几个）。
- 时间戳：MM:SS.mmm（官方六段式提示词的写法）。
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
