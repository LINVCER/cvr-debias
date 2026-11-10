"""数据集下载器：支持断点续传、md5 校验、解包，只为把「拿到原始日志」这一步标准化。

天池（tianchi.aliyun.com/dataset/408）的 Ali-CCP 需要登录并同意数据协议，
匿名请求只会拿到 4 KB 的页面壳（302/403），所以本脚本不猜下载地址：

1. 浏览器登录天池、进入数据集页、点下载，复制真实文件链接（或整段 curl 命令里的 URL）；
2. 把链接交给本脚本：
       python scripts/download_data.py --out data/aliccp/raw \
           --url "<你复制的链接>" --cookie "<可选：浏览器里的 Cookie>"
3. 脚本会断点续传、校验大小、算 md5 并写进 manifest.json，最后自动解包。

也可以一次给多个链接（train / test 分开时）：
    --url a.tar.gz --url b.tar.gz

若压缩包是 7z，需要本机已装 7z（Windows 下一般自带 tar 也能解 .tar.gz）。
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CHUNK = 1 << 20                      # 1 MB
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
TIANCHI_PAGE = "https://tianchi.aliyun.com/dataset/408"


def log(msg: str) -> None:
    print(msg, flush=True)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, cookie: str = "", referer: str = "", retries: int = 3) -> None:
    """带断点续传的下载。服务端不支持 Range 时退化为整包重下。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0

    for attempt in range(1, retries + 1):
        headers = {"User-Agent": UA, "Accept": "*/*"}
        if cookie:
            headers["Cookie"] = cookie
        if referer:
            headers["Referer"] = referer
        if offset:
            headers["Range"] = f"bytes={offset}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = resp.status
                total = int(resp.headers.get("Content-Length") or 0)
                if offset and status != 206:
                    log("[warn] 服务端不支持断点续传，从头开始")
                    offset, total = 0, total
                elif status == 206:
                    total += offset
                mode = "ab" if offset else "wb"
                done = offset
                start = time.time()
                with part.open(mode) as f:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if total:
                            pct = done * 100.0 / total
                            speed = done / max(time.time() - start, 1e-6) / CHUNK
                            log(f"\r  {dest.name}: {done/CHUNK:.1f}/{total/CHUNK:.1f} MB "
                                f"({pct:.1f}%, {speed:.1f} MB/s)")
                log("")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            log(f"[warn] 第 {attempt} 次失败：{exc}")
            if attempt == retries:
                raise SystemExit(
                    f"下载失败。若返回 302/403，说明链接需要登录态：\n"
                    f"  1) 浏览器打开 {TIANCHI_PAGE} 并登录；\n"
                    f"  2) 复制真实下载链接（页面里「下载」按钮的地址）；\n"
                    f"  3) 重跑并加上 --cookie \"<浏览器 Cookie>\" --referer {TIANCHI_PAGE}"
                )
            time.sleep(3 * attempt)
            continue

        size = part.stat().st_size
        if size < 1024 * 1024 and part.suffix == ".part":
            head = part.read_bytes()[:200]
            if b"<html" in head.lower() or b"<!doctype" in head.lower():
                part.unlink(missing_ok=True)
                raise SystemExit(
                    "下到的是网页而不是数据文件：链接需要登录态。"
                    f"请在浏览器打开 {TIANCHI_PAGE} 登录后复制真实下载链接。"
                )
        if offset and total and size != total:
            log(f"[warn] 大小不符（{size} != {total}），保留 .part 供续传")
            offset = size
            continue
        part.replace(dest)
        return


def unpack(path: Path, out_dir: Path) -> None:
    name = path.name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as tf:
            try:
                tf.extractall(out_dir, filter="data")   # 3.12+：拒绝绝对路径与越界链接
            except TypeError:
                tf.extractall(out_dir)
    elif name.endswith(".gz"):
        import gzip
        target = out_dir / path.name[:-3]
        with gzip.open(path, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    elif name.endswith(".7z"):
        exe = shutil.which("7z") or shutil.which("7za")
        if not exe:
            raise SystemExit("需要 7z 解包：请先安装 7-Zip 并把 7z 加入 PATH")
        subprocess.run([exe, "x", "-y", f"-o{out_dir}", str(path)], check=True)
    else:
        log(f"[info] {path.name} 不是压缩包，跳过解包")
        return
    log(f"[ok] 已解包到 {out_dir}")


def write_manifest(out_dir: Path, records: list) -> Path:
    path = out_dir / "manifest.json"
    old = []
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8")).get("files", [])
    merged = {r["file"]: r for r in old}
    for r in records:
        merged[r["file"]] = r
    path.write_text(
        json.dumps({"dataset": "Ali-CCP", "page": TIANCHI_PAGE, "files": list(merged.values())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Ali-CCP 下载器（断点续传 + md5 + 解包）")
    ap.add_argument("--url", action="append", default=[], help="真实下载链接，可重复")
    ap.add_argument("--out", default="data/aliccp/raw", help="输出目录")
    ap.add_argument("--cookie", default="", help="天池登录态 Cookie（可选）")
    ap.add_argument("--referer", default="", help="Referer，天池一般填数据集页")
    ap.add_argument("--no-unpack", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not args.url:
        log("没有给链接，先说明怎么拿：")
        log(f"  1. 浏览器打开 {TIANCHI_PAGE} 并登录（数据集需要同意协议）；")
        log("  2. 点「下载」，复制文件地址（通常是 .tar.gz / .zip / .7z）；")
        log("  3. 重跑：python scripts/download_data.py --out data/aliccp/raw \\")
        log("            --url \"<链接>\" --cookie \"<浏览器 Cookie>\"")
        log("  已登录时 Cookie 一般可省略；若 302/403 再加 --cookie。")
        return 1

    records = []
    for url in args.url:
        name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
        dest = out_dir / name
        if dest.exists():
            log(f"[skip] 已存在 {dest.name}（md5 {md5sum(dest)}）")
        else:
            download(url, dest, args.cookie, args.referer or TIANCHI_PAGE)
        records.append({"file": dest.name, "bytes": dest.stat().st_size, "md5": md5sum(dest), "url": url.split("?")[0]})
        if not args.no_unpack:
            unpack(dest, out_dir)

    manifest = write_manifest(out_dir, records)
    log(f"[ok] manifest 写入 {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
