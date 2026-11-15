# ComfyUI-WZF-VideoRemake · 视频改拍（换人 / 换头 / 改物）

> 传一段视频 + 一张人物照片，自动把视频里的人换掉（或者把物体、场景改掉），动作、镜头、切镜和原声都保留。
> 底层模型是 [Viggle Remake](https://huggingface.co/Viggle/Viggle-Remake)（MiniMax H3 视频改拍 LoRA），本节点包负责把它变成"一键"：自动找切镜、自动画关键帧、自动写提示词、按显卡自动选模型。

**MiniMax H3 · Powered by Viggle Remake**。本项目与 Viggle、MiniMax 无关联，也未获其背书。

## 为什么需要它

Viggle Remake 很强，但原生用法门槛很高：

1. 每个镜头都要先用改图工具把一帧"画"成目标样子，并且知道它在第几帧；
2. 要写一段六段式英文提示词，时间戳要和帧数对上；
3. 帧数必须是 17n+5，还要按显卡选 int8 / GGUF / bf16 模型。

实测（RTX 5070 Ti 16GB）发现这几步一步都省不掉：

- 只给一张人物照片、不画关键帧：**脸换不掉**，只会把头发颜色改一改；
- 提示词把"原来的人"和"新的人"写混：**有的镜头换不掉**；
- 画关键帧时不写清新人长相：特写镜头里**只换了衣服，脸没换**。

本节点包把这些全部自动化，并且每一步都有兜底。

## 节点

| 节点 | 作用 |
|---|---|
| ① 视频分析（切镜 · 帧数 · 关键帧） | 按时间截取（长视频不会整段读进内存）→ 转 24fps → 按显存档位定帧数（17n+5，本地 226 帧 ≈ 9.4 秒）→ 缩放到 Viggle 画布（短边 480 / 736，任意比例）→ 自动找切镜 → 每个镜头选一帧关键帧（最多 4 个，镜头多时取最长的 4 个）→ 出带帧号的缩略图 |
| ② 画关键帧（Qwen 改图） | 用 Qwen-Image 2.1 把关键帧改成目标样子：换整个人 / 只换头（BFS 换头 LoRA）/ 改物。先让 Qwen3-VL 8B 描述参考照片里的人长什么样写进提示词；画完让 8B 自检"脸换掉没有"，没换掉自动换种子重画（最多 2 次） |
| ③ 自动写提示词（本地 8B） | 六段式提示词的结构（谁换成谁、哪张图对哪个镜头、切镜时间、保留什么）由代码按 Viggle 官方句式写死；8B 只看图写短描述（原来的人 / 新的人、每个镜头在做什么、画面风格）。「自己写」框里有内容就原样用 |
| ④ 改拍模型（显存档位） | 默认「自动」：16G 以上用 int8；12G 左右用 Q8_0 GGUF 并在运行时多留 8GB 显存（等同 `--reserve-vram 8`，不用改启动参数）；80G 以上且有 bf16 文件用 bf16 + LoRA。同时给出换人 / 改物各自的官方采样曲线 |
| ⑤ 左右对比 | 原片 \| 成片 拼成一个画面 |

切镜检测在 6 段已知切镜位置的素材上全部正确（包括只换衣服颜色的细微切镜、面团糊住镜头和平滑航拍这类容易误报的情况）。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wwzhifeng/ComfyUI-WZF-VideoRemake
```

还需要：

- ComfyUI **0.42 或更新**（实测 0.45.0）。
- Viggle 官方节点包 `ComfyUI-Viggle-Remake`（Apache-2.0）：从 [Viggle/Viggle-Remake 的 comfyui/ 目录](https://huggingface.co/Viggle/Viggle-Remake/tree/main/comfyui) 复制到 `custom_nodes/`。
- 12G 档要 [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)。

没有额外的 pip 依赖。

## 模型

| 文件 | 放在 | 来源 | 用途 |
|---|---|---|---|
| `viggle_remake_swap_pruned_int8_convrot.safetensors`（20.97 GB） | `models/diffusion_models/` | Viggle/Viggle-Remake `int8/` | 16G：换人 / 换头 |
| `viggle_remake_edit_pruned_int8_convrot.safetensors`（20.97 GB） | 同上 | 同上 | 16G：改物 / 改场景 |
| `viggle_remake_swap-Q8_0.gguf`、`viggle_remake_edit-Q8_0.gguf`（各 22.25 GB） | 同上 | Viggle/Viggle-Remake `gguf/` | 12G |
| `minimax_h3_fl2va_bf16.safetensors` + `viggle_remake_{swap,edit}_comfyui.safetensors` | `diffusion_models/`、`loras/` | Comfy-Org/MiniMax-H3、Viggle | 云端 80G+ |
| H3 文本编码器（`qwen3vl_32b_minimax_h3_*`，int4 / nvfp4 / int8 / bf16 任选） | `models/text_encoders/` | Comfy-Org/MiniMax-H3 等 | 必需 |
| `minimax_h3_video_vae_fp16.safetensors`、`minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | Comfy-Org/MiniMax-H3 | 必需 |
| `qwen_image_2.1_int8_convrot.safetensors` | `models/diffusion_models/` | Comfy-Org/Qwen-Image-2.1 | 画关键帧 |
| `qwen3vl_8b_w4a8.safetensors` | `models/text_encoders/` | Comfy-Org/Qwen-Image-2.1 | 画关键帧 + 8B 看图写字 |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` | Comfy-Org/Qwen-Image-2.1 | 画关键帧 |
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors`、`elusarcas-qwen2-1-detailer-v1.safetensors`、`bfs_head_v1.1_qwen_2.1.safetensors` | `models/loras/` | Viggle、reverentelusarcas、Alissonerdx/BFS-Best-Face-Swap | 画关键帧（6 步加速 / 细节 / 换头） |

下载时请核对 SHA256。

## 使用

打开 `example_workflows/视频改拍_换人换头改物_王知风.json`：

1. 上传原视频（多镜头短剧也行）；
2. 「换整个人」「只换头」要上传人物参考照片（露脸、背景干净最好）；
3. 在 ① 里选「改什么」，写「换谁」（多人时写清楚，如"穿灰色外套的女人"）和「改成什么样」（中文就行），点运行。

实测 RTX 5070 Ti 16GB、64GB 内存：9.4 秒 832×480（或同比例）成片约 15 分钟（含首次从机械硬盘加载模型）。**内存建议 64GB**（进程峰值约 62GB）。

## 许可与使用限制

- 本仓库代码：MIT（见 `LICENSE.txt`）。
- 提示词句式改编自 Viggle Remake 的 `mcp/` 示例（Apache-2.0，见 `data/LICENSE-Viggle-Apache-2.0.txt` 和 `NOTICE.md`）。
- **模型权重不属于本仓库**，各自遵循原许可。Viggle Remake 和 MiniMax H3 权重适用 MiniMax H3 Community License：美国、欧盟、英国、韩国不在许可地域内；商业产品要在界面醒目显示 "MiniMax H3"；要随分发附带许可原文和 NOTICE。
- 仅限用于你有权修改的素材。**禁止未经本人同意替换他人肖像**；发布生成内容时请标注"AI 生成"。

作者：王知风（wwzhifeng） · 官网 https://huigui.ai · AI 工具群 957732664
