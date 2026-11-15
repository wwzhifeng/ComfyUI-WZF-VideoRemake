// WZF 大字链接：画超大字的横幅节点，点一下打开链接。
// 纯前端虚拟节点（和 Note 一样），不参与运行。
// 和 ComfyUI-WZF-BigLink 是同一份代码；随本节点包带一份，只装了本节点包的人打开工作流也不缺节点。
// 两边都装时谁先注册用谁，不重复注册。
import { app } from "../../scripts/app.js";

const TYPE = "WZFBigLink";

function wrapLines(ctx, text, maxW) {
  const out = [];
  for (const para of String(text || "").split("\n")) {
    let line = "";
    for (const ch of para) {
      if (ctx.measureText(line + ch).width > maxW && line) {
        out.push(line);
        line = ch;
      } else {
        line += ch;
      }
    }
    out.push(line);
  }
  return out;
}

app.registerExtension({
  name: "wzf.videoremake.biglink",
  registerCustomNodes() {
    if (LiteGraph.registered_node_types?.[TYPE]) return;
    class WZFBigLink extends LGraphNode {
      constructor(title) {
        super(title);
        this.isVirtualNode = true;
        this.serialize_widgets = false;
        this.properties = Object.assign(
          {
            text: "算力不够？请来云端",
            sub: "点这里打开在线版",
            url: "https://www.runninghub.cn",
            fontSize: 110,
            subSize: 52,
            textColor: "#ffd166",
            subColor: "#ffffff",
            panelColor: "#3a2600",
          },
          this.properties || {}
        );
        this.title = "☁️ 点击打开";
        this.color = "#7a5a00";
        this.bgcolor = "#3a2600";
        this.size = [1400, 360];
      }

      _layout(ctx) {
        const p = this.properties;
        const pad = 40;
        const maxW = this.size[0] - pad * 2;
        ctx.font = `900 ${p.fontSize}px "Microsoft YaHei","PingFang SC",sans-serif`;
        const big = wrapLines(ctx, p.text, maxW);
        ctx.font = `700 ${p.subSize}px "Microsoft YaHei","PingFang SC",sans-serif`;
        const sub = wrapLines(ctx, p.sub, maxW);
        const h = pad + big.length * p.fontSize * 1.2 + (p.sub ? 16 + sub.length * p.subSize * 1.35 : 0) + pad;
        return { pad, big, sub, h };
      }

      onDrawForeground(ctx) {
        if (this.flags.collapsed) return;
        const p = this.properties;
        const L = this._layout(ctx);
        if (Math.abs(L.h - this.size[1]) > 2) this.size[1] = L.h;
        ctx.save();
        ctx.fillStyle = p.panelColor;
        ctx.beginPath();
        ctx.roundRect(0, 0, this.size[0], this.size[1], 12);
        ctx.fill();
        ctx.textBaseline = "top";
        let y = L.pad;
        ctx.font = `900 ${p.fontSize}px "Microsoft YaHei","PingFang SC",sans-serif`;
        ctx.fillStyle = p.textColor;
        for (const line of L.big) {
          ctx.fillText(line, L.pad, y);
          y += p.fontSize * 1.2;
        }
        if (p.sub) {
          y += 16;
          ctx.font = `700 ${p.subSize}px "Microsoft YaHei","PingFang SC",sans-serif`;
          ctx.fillStyle = p.subColor;
          for (const line of L.sub) {
            ctx.fillText(line, L.pad, y);
            const w = ctx.measureText(line).width;
            ctx.fillRect(L.pad, y + p.subSize * 1.12, w, 3); // underline = link
            y += p.subSize * 1.35;
          }
        }
        ctx.restore();
      }

      onMouseDown(e, pos) {
        // pos is relative to the node body (title bar has negative y)
        if (pos[1] < 0 || !this.properties.url) return false;
        window.open(this.properties.url, "_blank", "noopener");
        return true;
      }
    }

    WZFBigLink.title = "WZF 大字链接";
    WZFBigLink.title_mode = LiteGraph.NORMAL_TITLE;
    WZFBigLink.collapsable = false;
    LiteGraph.registerNodeType(TYPE, Object.assign(WZFBigLink, { category: "WZF" }));
  },
});
