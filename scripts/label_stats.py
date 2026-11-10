"""扫描 L0 骨架表标签列，输出曝光/点击/转化与三级转化率到 L4 报告层。

列结构（实测，见 doc/202610-选题/08-数据探测记录.md）：
    0=行号  1=click  2=conversion  3=user_id  4=?  5=特征串
只切前 5 个逗号，不解析特征串，10 GB 级文件数分钟内完成。
"""

import json
from pathlib import Path

RAW = Path("data/aliccp/00_raw")
OUT = Path("data/aliccp/90_reports/01_probe/label_stats.json")


def scan(path: Path) -> dict:
    n = click = conv = bad = 0
    with path.open("rb") as f:
        carry = b""
        while True:
            block = f.read(1 << 22)
            if not block:
                break
            lines = (carry + block).split(b"\n")
            carry = lines.pop()          # 末段可能是半行，留到下一块
            for ln in lines:
                if not ln:
                    continue
                n += 1
                parts = ln.split(b",", 5)
                c = parts[1] if len(parts) > 1 else b""
                v = parts[2] if len(parts) > 2 else b""
                click += c == b"1"
                if v == b"1":
                    conv += 1
                    if c != b"1":
                        bad += 1
    return {
        "exposure": n,
        "click": click,
        "conversion": conv,
        "click0_but_conv1": bad,
        "ctr": click / n if n else 0.0,
        "cvr_click_space": conv / click if click else 0.0,
        "ctcvr": conv / n if n else 0.0,
    }


out = {}
for split in ("train", "test"):
    s = scan(RAW / split / f"sample_skeleton_{split}.csv")
    out[split] = s
    print(f"{split}: 曝光 {s['exposure']:,}  点击 {s['click']:,}  转化 {s['conversion']:,}  "
          f"CTR {s['ctr']:.4%}  CVR {s['cvr_click_space']:.4%}  CTCVR {s['ctcvr']:.4%}  "
          f"click=0&conv=1 异常 {s['click0_but_conv1']:,}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已写 {OUT}")
