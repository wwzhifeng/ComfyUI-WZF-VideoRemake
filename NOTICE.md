# NOTICE

ComfyUI-WZF-VideoRemake — Copyright (c) 2026 wwzhifeng, MIT License.

## Third-party material

- **Viggle Remake** (https://huggingface.co/Viggle/Viggle-Remake), Copyright (c) 2026 Viggle AI.
  The six-section prompt structure and sentence patterns in `nodes.py` (`compose`, `PAIR_ASK`, `SHOT_ASK`) are adapted from
  the example prompts and the prompt brief in Viggle Remake's `mcp/` directory, which is licensed under the Apache License 2.0
  (copy in `data/LICENSE-Viggle-Apache-2.0.txt`). Changes: rewritten as fixed templates filled by a local vision-language model;
  added head-only and edit variants.
- The head-swap instruction text used for Qwen-Image 2.1 follows the prompt published with the BFS head-swap LoRA
  (Alissonerdx/BFS-Best-Face-Swap).

## Model weights

No model weights are included in this repository. Viggle Remake and MiniMax H3 weights are licensed under the
MiniMax H3 Community License Agreement: "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement,
Copyright (c) 2026 MiniMax. All Rights Reserved." Other weights follow their own licenses.

"Viggle" is a name of Viggle AI and "MiniMax H3" is a mark of MiniMax; this project is not affiliated with or endorsed by either.
