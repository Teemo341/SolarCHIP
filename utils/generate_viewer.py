#!/usr/bin/env python3
"""扫描当前目录下的数据集文件夹，生成自包含的太阳图像模型对比查看器 solar_viewer.html。

数据集结构示例 (0094/):
    original/                        AIA20240108_0000_0094.png
    ctrl_best_hmi-0094/              sample_20240108_000000.png / sample_cfg_20240108_000000.png
    hmi_aia_dash_pix2pixhd_0094/     sample_20240108_000000.png
    hmi_aia_sdoml_cnn_0094/          sample_20240108_000000.png

hmi/ 子文件夹则是 AIA -> HMI 方向:
    hmi/original/                    hmi.M_720s.20240108_000000_TAI.png
    hmi/ctrl_best_0094-hmi/          sample_20240108_0000.png (+ sample_cfg_)
    hmi/aia_hmi_dannehl_pix2pixcc_0094/  sample_20240108_0000.png
    hmi/aia_hmi_i2iwfilm_0094/       sample_20240108_0000.png

用法:
    python3 generate_viewer.py            仅生成 solar_viewer.html (双击打开; 收藏存入浏览器本地)
    python3 generate_viewer.py --serve    启动本地服务器并打开浏览器 (收藏可写入 best.json)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_RE = re.compile(r"^\d+$")
DATE_RE = re.compile(r"(\d{8})")
BEST_PATH = ROOT / "best.json"


def load_best() -> dict:
    """读取 best.json (按 任务/模态 组织的收藏)."""
    if BEST_PATH.exists():
        try:
            data = json.loads(BEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_best(data: dict) -> None:
    BEST_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

MODEL_LABELS = {
    "ctrl": "Control (best HMI)",
    "dash": "DASH pix2pixHD",
    "sdoml": "SDOML CNN",
    "dannehl": "Dannehl pix2pixCC",
    "i2iwfilm": "I2I W-Film",
}

# 模型展示优先级 (ctrl 最前, 之后按对比模型顺序)
MODEL_PRIORITY = {
    "ctrl": 0,
    "dash": 1,
    "dannehl": 1,
    "sdoml": 2,
    "i2iwfilm": 2,
}


def extract_date(name: str):
    m = DATE_RE.search(Path(name).stem)
    return m.group(1) if m else None


def label_for(dir_name: str) -> str:
    lower = dir_name.lower()
    for key, label in MODEL_LABELS.items():
        if key in lower:
            return label
    return dir_name


def build_dataset(ds_id: str, prefix: str, orig_dir: Path, model_dirs, orig_label: str) -> dict:
    """构建一个数据集条目。

    prefix: 该数据集相对 ROOT 的路径前缀, 如 '0094' 或 'hmi'
    model_dirs: 模型文件夹列表 (对应的 ds_id 已选定)
    """
    dates = set()

    # 1) original
    orig_files = {}
    if orig_dir.is_dir():
        for f in orig_dir.glob("*.png"):
            d = extract_date(f.name)
            if d:
                orig_files[d] = f"{prefix}/original/{f.name}"
                dates.add(d)

    # 2) 模型目录: ctrl 优先, 其余按优先级/名称排序
    def sort_key(d: Path):
        lower = d.name.lower()
        prio = next((p for k, p in MODEL_PRIORITY.items() if k in lower), 3)
        return (prio, d.name)

    slots = []
    for d in sorted(model_dirs, key=sort_key):
        files = {f.name for f in d.glob("*.png")}
        normal = {}
        cfg = {}
        for name in files:
            date = extract_date(name)
            if not date:
                continue
            rel = f"{prefix}/{d.name}/{name}"
            if "cfg" in name.lower():
                cfg[date] = rel
            else:
                normal[date] = rel
            dates.add(date)
        slots.append(
            {
                "label": label_for(d.name),
                "files": normal,
                "cfg": cfg or None,
            }
        )

    return {
        "id": ds_id,
        "dates": sorted(dates),
        "original": {"label": orig_label, "files": orig_files},
        "models": slots,
    }


def scan_hmi_aia() -> list:
    """顶层数字文件夹 (0094/): HMI -> AIA 方向"""
    datasets = []
    for child in sorted(ROOT.iterdir()):
        if not (child.is_dir() and DATASET_RE.match(child.name)):
            continue
        model_dirs = [
            d for d in child.iterdir()
            if d.is_dir() and d.name != "original" and any(d.glob("*.png"))
        ]
        ds = build_dataset(child.name, child.name, child / "original", model_dirs, "Original AIA")
        if ds["dates"]:
            datasets.append(ds)
    return datasets


def scan_aia_hmi() -> list:
    """hmi/ 文件夹: AIA -> HMI 方向。

    hmi/ 下一级目录名 = 数据集 id + 模型名 (如 ctrl_best_0094-hmi,
    aia_hmi_dannehl_pix2pixcc_0094), original/ 是真实 HMI 观测。
    """
    hmi_dir = ROOT / "hmi"
    datasets = []
    if not hmi_dir.is_dir():
        return datasets

    # 从一级目录名提取所有的数据集 id
    ids = set()
    for d in hmi_dir.iterdir():
        if d.is_dir():
            m = re.search(r"(\d{4})", d.name)
            if m:
                ids.add(m.group(1))

    for ds_id in sorted(ids):
        model_dirs = [
            d for d in hmi_dir.iterdir()
            if d.is_dir() and ds_id in d.name and any(d.glob("*.png"))
        ]
        ds = build_dataset(ds_id, "hmi", hmi_dir / "original", model_dirs, "Original HMI")
        if ds["dates"]:
            datasets.append(ds)
    return datasets


def scan_all() -> dict:
    return {
        "directions": [
            {"id": "hmi-aia", "label": "HMI → AIA", "datasets": scan_hmi_aia()},
            {"id": "aia-hmi", "label": "AIA → HMI", "datasets": scan_aia_hmi()},
        ]
    }


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>太阳图像模型对比查看器</title>
<style>
  :root {
    --bg: #0b0e14;
    --panel: #131824;
    --panel-2: #1a2233;
    --border: #2a3550;
    --text: #e6ecf5;
    --muted: #8b98b0;
    --accent: #ffb84d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(1200px 600px at 70% -10%, #1a2340 0%, var(--bg) 55%);
    color: var(--text);
    font-family: -apple-system, "PingFang SC", "Segoe UI", sans-serif;
    min-height: 100vh;
  }
  header {
    position: sticky; top: 0; z-index: 10;
    backdrop-filter: blur(10px);
    background: rgba(11, 14, 20, .82);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
  }
  .head-top { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0; letter-spacing: .5px; }
  h1 .sun { color: var(--accent); }
  .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  label.field { color: var(--muted); font-size: 13px; display: flex; align-items: center; gap: 6px; }
  select, button, .toggle {
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 13px;
    cursor: pointer;
    transition: border-color .15s, background .15s;
  }
  select:hover, button:hover, .toggle:hover { border-color: var(--accent); }
  button:disabled { opacity: .35; cursor: not-allowed; }
  .toggle {
    display: flex; align-items: center; gap: 8px; user-select: none;
  }
  .toggle input { accent-color: var(--accent); cursor: pointer; }
  .seg {
    display: flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  }
  .seg-btn {
    background: var(--panel-2); color: var(--muted);
    border: none; border-radius: 0; padding: 7px 14px; font-size: 13px; cursor: pointer;
    transition: background .15s, color .15s;
  }
  .seg-btn + .seg-btn { border-left: 1px solid var(--border); }
  .seg-btn:hover { color: var(--text); }
  .seg-btn.active { background: var(--accent); color: #231300; font-weight: 600; }
  #favBtn { font-size: 14px; letter-spacing: .5px; }
  #favBtn.faved { color: var(--accent); border-color: var(--accent); }
  #favFilter.active {
    background: var(--accent); color: #231300; font-weight: 600; border-color: var(--accent);
  }
  #toast {
    position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%) translateY(20px);
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 18px; font-size: 13px; opacity: 0; pointer-events: none;
    transition: opacity .25s, transform .25s; z-index: 200; max-width: 80vw;
  }
  #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  #info { margin-left: auto; color: var(--muted); font-size: 13px; }
  main {
    padding: 22px 24px 60px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 18px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform .15s, box-shadow .15s;
  }
  .card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.45); }
  .card.hidden { display: none; }
  .card-head {
    padding: 9px 14px;
    font-size: 13px;
    font-weight: 600;
    background: var(--panel-2);
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .card-head .tag { color: var(--accent); font-weight: 500; font-size: 12px; }
  .img-wrap { background: #000; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; }
  .img-wrap img { width: 100%; height: 100%; object-fit: contain; cursor: zoom-in; display: block; }
  .placeholder {
    color: var(--muted); font-size: 12px; text-align: center; padding: 12px; width: 100%;
  }
  #lightbox {
    position: fixed; inset: 0; z-index: 100;
    background: rgba(0,0,0,.92);
    display: none; align-items: center; justify-content: center;
    cursor: zoom-out;
  }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 94vw; max-height: 94vh; object-fit: contain; border-radius: 6px; }
  #lightbox .lb-label {
    position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,.6); padding: 6px 14px; border-radius: 20px;
    font-size: 13px; color: var(--text);
  }
  @media (max-width: 700px) {
    main { grid-template-columns: 1fr; padding: 14px; }
    #info { margin-left: 0; }
  }
</style>
</head>
<body>
<header>
  <div class="head-top">
    <h1><span class="sun">☀</span> 太阳图像模型对比查看器</h1>
    <div class="controls">
      <div class="seg" id="dirSeg" title="切换生成方向">
        <button class="seg-btn active" data-dir="0">HMI → AIA</button>
        <button class="seg-btn" data-dir="1">AIA → HMI</button>
      </div>
      <label class="field">数据集
        <select id="dataset"></select>
      </label>
      <label class="field">日期
        <select id="date"></select>
      </label>
      <button id="prev" title="上一个日期 (←)">← 上一个日期</button>
      <button id="next" title="下一个日期 (→)">下一个日期 →</button>
      <button id="favBtn" title="收藏/取消收藏当前日期">☆ 收藏</button>
      <button id="favFilter" title="仅显示已收藏的日期">★ 只看收藏</button>
      <label class="toggle" title="是否展示 ctrl 的 CFG 采样">
        <input type="checkbox" id="showCfg" checked> 显示 CFG 采样
      </label>
      <span id="info"></span>
    </div>
  </div>
</header>
<main id="grid"></main>
<div id="lightbox">
  <img id="lb-img" alt="">
  <div class="lb-label" id="lb-label"></div>
</div>
<div id="toast"></div>

<script>
const MANIFEST = __MANIFEST__;
const EMBEDDED_BEST = __BEST__;

const $ = (id) => document.getElementById(id);
let dirIdx = 0, ds = null;
let allDates = [], viewDates = [], dateIdx = 0; // viewDates: 当前可见日期 (全部或收藏)
let favorites = null;   // { [任务]: { [模态]: [日期] } }
let favOnly = false;    // 只看收藏筛选

const curDir = () => MANIFEST.directions[dirIdx];

function fmtDate(d) {
  return d.slice(0,4) + "-" + d.slice(4,6) + "-" + d.slice(6,8);
}

// ---------- 收藏 ----------
function favList() {
  return (favorites[curDir().id] && favorites[curDir().id][ds.id]) || [];
}

function isFav(date) {
  return favList().includes(date);
}

function setFavList(list) {
  if (!favorites[curDir().id]) favorites[curDir().id] = {};
  favorites[curDir().id][ds.id] = list.slice().sort();
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._tm);
  toast._tm = setTimeout(() => t.classList.remove("show"), 2600);
}

async function persistBest() {
  try {
    const r = await fetch("best.json", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(favorites),
    });
    if (!r.ok) throw new Error("http " + r.status);
  } catch (e) {
    // file:// 下无法写文件, 退化为浏览器本地存储
    try { localStorage.setItem("best-favorites", JSON.stringify(favorites)); } catch (_) {}
    toast("注意: 当前以文件方式打开, 收藏已存浏览器本地; 用 python3 generate_viewer.py --serve 运行可写入 best.json");
  }
}

async function loadBest() {
  try {
    const r = await fetch("best.json?_=" + Date.now(), { cache: "no-store" });
    if (r.ok) { favorites = await r.json(); return; }
    throw new Error();
  } catch (e) { /* file:// 下 fetch 本地文件会被阻止 */ }
  try {
    const local = JSON.parse(localStorage.getItem("best-favorites") || "null");
    if (local) { favorites = local; return; }
  } catch (_) {}
  favorites = JSON.parse(JSON.stringify(EMBEDDED_BEST || {}));
}

function toggleFav() {
  const date = viewDates[dateIdx];
  if (!date) return;
  const list = new Set(favList());
  let added;
  if (list.has(date)) { list.delete(date); added = false; }
  else { list.add(date); added = true; }
  setFavList(Array.from(list));
  persistBest();
  updateFavBtn();
  if (favOnly) setView(added ? date : null); // 收藏列表变化后刷新可见日期
  toast(added ? "已收藏 " + fmtDate(date) : "已取消收藏 " + fmtDate(date));
}

// ---------- 日期可见列表 ----------
function setView(keepDate) {
  const cur = keepDate || viewDates[dateIdx];
  viewDates = favOnly ? allDates.filter((d) => favList().includes(d)) : allDates.slice();
  let i = cur ? viewDates.indexOf(cur) : -1;
  if (i < 0) i = viewDates.length - 1;
  dateIdx = Math.max(0, i);
  fillDates();
  render();
  updateFavBtn();
}

// ---------- 初始化 ----------
function fillDatasets(keepId) {
  const dir = curDir();
  const sel = $("dataset");
  const oldId = keepId || (ds ? ds.id : null);
  sel.innerHTML = "";
  dir.datasets.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = "数据集 " + d.id + "（" + d.dates.length + " 个日期）";
    sel.appendChild(opt);
  });
  let idx = oldId ? dir.datasets.findIndex(d => d.id === oldId) : -1;
  if (idx < 0) idx = 0; // 默认第一个数据集
  sel.value = String(idx);
  sel.onchange = () => { onDatasetChange(); };
  ds = dir.datasets[idx];
  allDates = ds.dates;
  setView();
}

function fillDates() {
  const sel = $("date");
  sel.innerHTML = "";
  viewDates.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = fmtDate(d);
    sel.appendChild(opt);
  });
  sel.onchange = () => { dateIdx = Number(sel.value); render(); updateFavBtn(); };
}

function onDatasetChange() {
  ds = curDir().datasets[Number($("dataset").value)];
  allDates = ds.dates;
  setView();
}

function cardEl(label, src, extraTag) {
  const card = document.createElement("div");
  card.className = "card";
  const head = document.createElement("div");
  head.className = "card-head";
  const title = document.createElement("span");
  title.textContent = label;
  head.appendChild(title);
  if (extraTag) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = extraTag;
    head.appendChild(tag);
  }
  const wrap = document.createElement("div");
  wrap.className = "img-wrap";
  if (src) {
    const img = document.createElement("img");
    img.src = src;
    img.loading = "lazy";
    img.alt = label;
    img.onclick = () => openLightbox(src, label);
    img.onerror = () => {
      img.style.display = "none";
      const ph = document.createElement("div");
      ph.className = "placeholder";
      ph.textContent = "⚠ 未找到图片";
      wrap.appendChild(ph);
    };
    wrap.appendChild(img);
  } else {
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.textContent = "— 该日期无此图 —";
    wrap.appendChild(ph);
  }
  card.appendChild(head);
  card.appendChild(wrap);
  return card;
}

function render() {
  const grid = $("grid");
  grid.innerHTML = "";
  const date = viewDates[dateIdx];

  if (!date) {
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.style.padding = "48px 20px";
    ph.style.fontSize = "14px";
    ph.style.gridColumn = "1 / -1";
    ph.textContent = favOnly
      ? "该任务/模态下暂无收藏日期。点击「☆ 收藏」添加。"
      : "暂无数据";
    grid.appendChild(ph);
    $("info").textContent = "";
    $("date").value = "";
    $("prev").disabled = true;
    $("next").disabled = true;
    return;
  }

  const items = [];
  // original
  items.push({ label: ds.original.label, src: ds.original.files[date] || null });
  // ctrl + cfg
  ds.models.forEach((m, i) => {
    items.push({ label: m.label, src: m.files[date] || null, tag: "采样" });
    if (m.cfg) {
      items.push({ label: m.label + " · CFG", src: m.cfg[date] || null, tag: "CFG 采样", isCfg: true });
    }
  });

  const showCfg = $("showCfg").checked;
  items.forEach(it => {
    const card = cardEl(it.label, it.src, it.tag);
    if (it.isCfg && !showCfg) card.classList.add("hidden");
    grid.appendChild(card);
  });

  $("info").textContent =
    (favOnly ? "★ 收藏 · " : "") +
    "第 " + (dateIdx + 1) + " / " + viewDates.length + " 天 · " + fmtDate(date);
  $("date").value = String(dateIdx);
  $("prev").disabled = dateIdx <= 0;
  $("next").disabled = dateIdx >= viewDates.length - 1;
}

function updateFavBtn() {
  const btn = $("favBtn");
  const fav = isFav(viewDates[dateIdx]);
  btn.textContent = fav ? "★ 已收藏" : "☆ 收藏";
  btn.classList.toggle("faved", fav);
}

function step(offset) {
  const next = dateIdx + offset;
  if (next < 0 || next >= viewDates.length) return;
  dateIdx = next;
  render();
  updateFavBtn();
}

// lightbox
function openLightbox(src, label) {
  $("lb-img").src = src;
  $("lb-label").textContent = label;
  $("lightbox").classList.add("open");
}
function closeLightbox() {
  $("lightbox").classList.remove("open");
  $("lb-img").src = "";
}
$("lightbox").onclick = closeLightbox;
document.addEventListener("keydown", (e) => {
  if ($("lightbox").classList.contains("open")) { if (e.key === "Escape") closeLightbox(); return; }
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(1);
});

$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);
$("showCfg").onchange = render;
$("favBtn").onclick = toggleFav;
$("favFilter").onclick = () => {
  favOnly = !favOnly;
  $("favFilter").classList.toggle("active", favOnly);
  setView();
  toast(favOnly
    ? "已开启「只看收藏」: 日期与翻页仅限收藏"
    : "已关闭「只看收藏」");
};

// 方向切换: 保留当前数据集 id
document.querySelectorAll(".seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    dirIdx = Number(btn.dataset.dir);
    document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
    fillDatasets(ds ? ds.id : null);
  });
});

(async () => {
  await loadBest();
  fillDatasets();
})();
</script>
</body>
</html>
"""


def main():
    manifest = scan_all()
    html = TEMPLATE.replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False))
    html = html.replace("__BEST__", json.dumps(load_best(), ensure_ascii=False))
    out = ROOT / "solar_viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}")
    for direc in manifest["directions"]:
        print(f"方向 {direc['label']}: {len(direc['datasets'])} 个数据集")
        for ds in direc["datasets"][:3]:
            print(f"  数据集 {ds['id']}: {len(ds['dates'])} 个日期, "
                  f"模型: {[m['label'] for m in ds['models']]}")


def serve(port: int = 8765):
    """以本地服务器方式运行: 页面可读写 best.json, 收藏持久化到文件。"""
    import webbrowser
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def _send_json(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", ""):
                self.path = "/solar_viewer.html"
            if self.path.rstrip("/") == "/manifest.json":
                self._send_json(scan_all())
                return
            if self.path.split("?")[0] == "/best.json":
                self._send_json(load_best())
                return
            super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] == "/best.json":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    data = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("best.json 必须是 JSON 对象")
                    save_best(data)
                    self._send_json({"ok": True, "best": data})
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_error(405)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"查看器已启动: {url}  (收藏将写入 {BEST_PATH.name}, Ctrl+C 退出)")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止.")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        main()
