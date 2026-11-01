import nbformat as nbf
from pathlib import Path


OUT = Path(r".\privacy_score_quantitative_docker_vscode.ipynb")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}

nb["cells"] = [
    md(
        """
# Privacy Score Quantification

이 노트북은 Docker/VSCode 환경에서 `/workspace/Train` 데이터셋을 자동으로 찾고, 원본 이미지와 변환 이미지의 구조적 차이를 이용해 privacy score를 정량화합니다.

핵심 점수는 `Privacy Score = DSSIM x 100 = ((1 - SSIM) / 2) x 100`입니다. 점수가 높을수록 원본과 구조적으로 더 달라졌다는 뜻입니다. PSNR은 점수식에 섞지 않고 참고 지표로만 함께 저장합니다.
"""
    ),
    md(
        """
## 1. 환경 확인과 라이브러리

아래 셀은 필요한 패키지를 확인합니다. Docker 컨테이너에서 `ModuleNotFoundError`가 나면 터미널에서 `pip install numpy pandas pillow scikit-image matplotlib openpyxl`을 실행한 뒤 커널을 재시작하세요.
"""
    ),
    code(
        r"""
from pathlib import Path
import os
import sys
import math
import time
import json
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize

from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

try:
    from IPython.display import display
except Exception:
    display = print

warnings.filterwarnings("ignore", category=UserWarning)
print("Python:", sys.version.split()[0])
print("Working directory:", Path.cwd())
print("Libraries loaded.")
"""
    ),
    md(
        """
## 2. 실험 설정

Docker/VSCode에서 돌릴 때는 보통 `BASE_TRAIN_OVERRIDE = None` 그대로 두면 `/workspace/Train`, `/workspace`, `/workspaces/...`, 현재 VSCode 폴더를 자동으로 검사합니다. 만약 컨테이너 안 경로가 다르면 `BASE_TRAIN_OVERRIDE = "/your/path"`만 바꾸세요.
"""
    ),
    code(
        r"""
# Set this only if auto-detection fails inside Docker/VSCode.
BASE_TRAIN_OVERRIDE = None

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IMG_SIZE = 512
SEED = 42

# Full experiment from the original notebook.
BLOCK_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
BLUR_SIGMAS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

# Leave as None for the full dataset. Set to a small integer for a quick smoke test.
N_SAMPLES_PER_CLASS = None
MAX_IMAGES_TOTAL = None

# Loading 1097 images at 512x512 uses roughly 0.9 GB plus overhead.
# If your Docker memory is tight, set this to False. It will be slower but lighter.
CACHE_IMAGES_IN_MEMORY = True

# Keep True to evaluate the full block x blur grid.
RUN_COMBO_GRID = True

DEFAULT_OUT_DIR = Path("/workspace/privacy_score_outputs")
if not DEFAULT_OUT_DIR.parent.exists():
    DEFAULT_OUT_DIR = Path.cwd() / "privacy_score_outputs"
OUT_DIR = DEFAULT_OUT_DIR
GRAPH_DIR = OUT_DIR / "graphs"
TABLE_DIR = OUT_DIR / "tables"

OUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("Output directory:", OUT_DIR)
"""
    ),
    md(
        """
## 3. 경로 자동 탐색과 데이터셋 진단

이 셀에서 현재 커널 기준으로 실제 존재하는 이미지 루트를 찾습니다. Docker 안에서 실행 중이면 Windows의 `C:\\...` 경로가 아니라 `/workspace/Train` 같은 컨테이너 내부 경로가 기준입니다.
"""
    ),
    code(
        r"""
def path_exists(path_like):
    try:
        return Path(path_like).exists()
    except Exception:
        return False


def candidate_train_dirs():
    candidates = []
    if BASE_TRAIN_OVERRIDE:
        candidates.append(Path(BASE_TRAIN_OVERRIDE))

    candidates.extend([
        Path("/workspace/Train"),
        Path("/workspace/train"),
        Path("/workspace/Images"),
        Path("/workspace/images"),
        Path("/workspace/data/Train"),
        Path("/workspace/data/train"),
        Path("/workspace/data/Images"),
        Path("/workspace/data/images"),
        Path.cwd() / "Train",
        Path.cwd() / "train",
        Path.cwd() / "Images",
        Path.cwd() / "images",
        Path.cwd() / "data" / "Train",
        Path.cwd() / "data" / "train",
        Path.cwd() / "data" / "Images",
        Path.cwd() / "data" / "images",
    ])

    # VSCode Dev Containers often mount projects under /workspaces/<project>.
    workspaces = Path("/workspaces")
    if workspaces.exists():
        for project_dir in sorted([p for p in workspaces.iterdir() if p.is_dir()]):
            candidates.extend([
                project_dir / "Train",
                project_dir / "train",
                project_dir / "Images",
                project_dir / "images",
                project_dir / "data" / "Train",
                project_dir / "data" / "train",
                project_dir / "data" / "Images",
                project_dir / "data" / "images",
                project_dir,
            ])

    # Fallbacks: useful when opening this notebook outside Docker.
    candidates.extend([
        Path("/workspace"),
        Path.cwd(),
        Path(r".\Train"),
    ])

    unique = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


IGNORE_DIR_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    "privacy_score_outputs",
    "graphs",
    "tables",
}


def should_ignore(path):
    return any(part in IGNORE_DIR_NAMES for part in path.parts)


def count_images_under(path):
    if not path.exists():
        return 0
    return sum(
        1
        for p in path.rglob("*")
        if p.is_file() and not should_ignore(p) and p.suffix.lower() in IMAGE_EXTENSIONS
    )


diagnosis_rows = []
for p in candidate_train_dirs():
    diagnosis_rows.append({
        "candidate_path": str(p),
        "exists": p.exists(),
        "image_count": count_images_under(p) if p.exists() else 0,
    })

path_diagnosis = pd.DataFrame(diagnosis_rows)
display(path_diagnosis)

valid_candidates = [Path(r["candidate_path"]) for r in diagnosis_rows if r["exists"] and r["image_count"] > 0]
if not valid_candidates:
    raise FileNotFoundError(
        "No image dataset was found. In Docker/VSCode, make sure your Train folder is mounted "
        "to /workspace/Train, or set BASE_TRAIN_OVERRIDE to the container path."
    )

BASE_TRAIN = valid_candidates[0]
print("Selected BASE_TRAIN:", BASE_TRAIN)
"""
    ),
    md(
        """
## 4. 이미지 목록 만들기

클래스 폴더 구조가 있으면 부모 폴더명을 클래스명으로 사용합니다. 예: `Train/Normal cases/*.jpg`.
"""
    ),
    code(
        r"""
def collect_image_table(base_dir):
    rows = []
    for p in sorted(base_dir.rglob("*")):
        if should_ignore(p) or not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = p.relative_to(base_dir)
        label = rel.parts[0] if len(rel.parts) > 1 else "root"
        rows.append({
            "path": str(p),
            "relative_path": str(rel),
            "label": label,
            "suffix": p.suffix.lower(),
        })
    return pd.DataFrame(rows)


image_df = collect_image_table(BASE_TRAIN)
if image_df.empty:
    raise RuntimeError(f"No images found under {BASE_TRAIN}")

class_counts = (
    image_df.groupby("label")
    .size()
    .rename("n_images")
    .reset_index()
    .sort_values("label")
)
print("Total images:", len(image_df))
display(class_counts)

rng = np.random.default_rng(SEED)
selected_df = image_df.copy()

if N_SAMPLES_PER_CLASS is not None:
    selected_df = (
        selected_df
        .groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), N_SAMPLES_PER_CLASS), random_state=SEED))
        .reset_index(drop=True)
    )

if MAX_IMAGES_TOTAL is not None and len(selected_df) > MAX_IMAGES_TOTAL:
    selected_df = selected_df.sample(n=MAX_IMAGES_TOTAL, random_state=SEED).reset_index(drop=True)

selected_df = selected_df.sort_values(["label", "relative_path"]).reset_index(drop=True)
selected_df.to_csv(TABLE_DIR / "selected_images.csv", index=False, encoding="utf-8-sig")

print("Selected images:", len(selected_df))
display(selected_df.head())
"""
    ),
    md(
        """
## 5. 한글 폰트와 이미지 변환 함수

그래프 라벨은 호환성을 위해 대부분 영어로 저장합니다. 한글 폰트가 있으면 자동 적용됩니다.
"""
    ),
    code(
        r"""
def setup_matplotlib_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        r"C:\Windows\Fonts\malgun.ttf",
    ]
    font_path = next((Path(p) for p in candidates if Path(p).exists()), None)
    if font_path is None:
        for p in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
            low = p.lower()
            if any(name in low for name in ["nanum", "noto", "malgun", "gulim"]):
                font_path = Path(p)
                break
    if font_path is not None:
        fm.fontManager.addfont(str(font_path))
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()

    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#222222",
        "grid.color": "#dddddd",
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "savefig.bbox": "tight",
    })
    return font_path


font_path = setup_matplotlib_font()
print("Font:", font_path if font_path else "default")

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def load_rgb_image(path, img_size=IMG_SIZE):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((img_size, img_size), RESAMPLE_LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def block_scramble(img_array, seed=SEED, block_size=32):
    h, w, c = img_array.shape
    block_size = int(block_size)
    if block_size >= min(h, w):
        return img_array.copy()
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h or pad_w:
        padded = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        padded = img_array

    ph, pw = padded.shape[:2]
    n_h, n_w = ph // block_size, pw // block_size
    blocks = padded.reshape(n_h, block_size, n_w, block_size, c).swapaxes(1, 2)
    flat_blocks = blocks.reshape(n_h * n_w, block_size, block_size, c)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_h * n_w)
    scrambled = flat_blocks[order].reshape(n_h, n_w, block_size, block_size, c)
    scrambled = scrambled.swapaxes(1, 2).reshape(ph, pw, c)
    return scrambled[:h, :w].astype(np.uint8)


def apply_gaussian_blur(img_array, sigma=0):
    sigma = float(sigma)
    if sigma <= 0:
        return img_array.copy()
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.uint8)


def apply_transform(img_array, method, block_size=None, sigma=None):
    if method == "block":
        return block_scramble(img_array, seed=SEED, block_size=block_size)
    if method == "blur":
        return apply_gaussian_blur(img_array, sigma=sigma)
    if method == "combo":
        x = block_scramble(img_array, seed=SEED, block_size=block_size)
        return apply_gaussian_blur(x, sigma=sigma)
    raise ValueError(f"Unknown method: {method}")

print("Transform functions ready.")
"""
    ),
    md(
        """
## 6. Privacy Score 계산 함수

SSIM 기반 DSSIM만 privacy score로 사용합니다. PSNR은 `psnr_mean` 컬럼에 참고용으로 같이 남깁니다.
"""
    ),
    code(
        r"""
def compute_pair_metrics(original, transformed):
    ssim_value = ssim_fn(original, transformed, channel_axis=-1, data_range=255)
    psnr_value = psnr_fn(original, transformed, data_range=255)
    if not np.isfinite(psnr_value):
        psnr_value = 100.0
    dssim = np.clip((1.0 - float(ssim_value)) / 2.0, 0.0, 1.0)
    privacy_score = dssim * 100.0
    return float(ssim_value), float(psnr_value), float(dssim), float(privacy_score)


def load_image_records(selected):
    records = []
    start = time.time()
    for i, row in selected.iterrows():
        img = load_rgb_image(row["path"])
        records.append({
            "path": row["path"],
            "relative_path": row["relative_path"],
            "label": row["label"],
            "image": img,
        })
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{len(selected)} images")
    print(f"Loaded {len(records)} images in {time.time() - start:.1f}s")
    return records


if CACHE_IMAGES_IN_MEMORY:
    image_records = load_image_records(selected_df)
else:
    image_records = None
    print("Image cache disabled. Images will be loaded from disk for each condition.")


def iter_records():
    if image_records is not None:
        yield from image_records
    else:
        for _, row in selected_df.iterrows():
            yield {
                "path": row["path"],
                "relative_path": row["relative_path"],
                "label": row["label"],
                "image": load_rgb_image(row["path"]),
            }


def evaluate_condition(method, block_size=None, sigma=None):
    ssims, psnrs, dssims, scores = [], [], [], []
    label_scores = {}
    n = 0

    for rec in iter_records():
        original = rec["image"]
        transformed = apply_transform(original, method=method, block_size=block_size, sigma=sigma)
        ssim_value, psnr_value, dssim_value, score_value = compute_pair_metrics(original, transformed)
        ssims.append(ssim_value)
        psnrs.append(psnr_value)
        dssims.append(dssim_value)
        scores.append(score_value)
        label_scores.setdefault(rec["label"], []).append(score_value)
        n += 1

    row = {
        "condition": condition_name(method, block_size, sigma),
        "method": method,
        "block_size": np.nan if block_size is None else int(block_size),
        "sigma": np.nan if sigma is None else float(sigma),
        "n_images": n,
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "dssim_mean": float(np.mean(dssims)),
        "privacy_score": float(np.mean(scores)),
        "privacy_score_std": float(np.std(scores)),
    }

    for label, values in label_scores.items():
        safe_label = str(label).replace(" ", "_").replace("/", "_").replace("\\", "_")
        row[f"privacy_score_{safe_label}"] = float(np.mean(values))
    return row


def condition_name(method, block_size=None, sigma=None):
    if method == "block":
        return f"block_B{int(block_size)}"
    if method == "blur":
        return f"blur_S{format_sigma(sigma)}"
    if method == "combo":
        return f"combo_B{int(block_size)}_S{format_sigma(sigma)}"
    return method


def format_sigma(sigma):
    sigma = float(sigma)
    return str(int(sigma)) if sigma.is_integer() else str(sigma).replace(".", "p")


print("Metric functions ready.")
"""
    ),
    md(
        """
## 7. 실험 실행

전체 조건은 블록 단독 10개, 블러 단독 10개, 조합 100개입니다. Docker 메모리나 시간이 부족하면 위 설정에서 `N_SAMPLES_PER_CLASS`를 먼저 작게 잡아 테스트하세요.
"""
    ),
    code(
        r"""
experiment_plan = []
for bs in BLOCK_SIZES:
    experiment_plan.append(("block", bs, None))
for sg in BLUR_SIGMAS:
    experiment_plan.append(("blur", None, sg))
if RUN_COMBO_GRID:
    for bs in BLOCK_SIZES:
        for sg in BLUR_SIGMAS:
            experiment_plan.append(("combo", bs, sg))

print("Number of conditions:", len(experiment_plan))

rows = []
start_all = time.time()
for idx, (method, bs, sg) in enumerate(experiment_plan, start=1):
    started = time.time()
    row = evaluate_condition(method, block_size=bs, sigma=sg)
    rows.append(row)
    print(
        f"[{idx:03d}/{len(experiment_plan)}] {row['condition']}: "
        f"score={row['privacy_score']:.2f}, ssim={row['ssim_mean']:.4f}, "
        f"psnr={row['psnr_mean']:.2f} dB, time={time.time() - started:.1f}s"
    )

df = pd.DataFrame(rows)
print(f"Done in {(time.time() - start_all) / 60:.1f} minutes")
display(df.head())
"""
    ),
    md(
        """
## 8. 결과 표 저장

CSV와 Excel 파일을 모두 저장합니다. Excel 저장이 실패해도 CSV는 저장됩니다.
"""
    ),
    code(
        r"""
result_csv = TABLE_DIR / "privacy_scores_long.csv"
summary_csv = TABLE_DIR / "privacy_scores_summary_by_method.csv"
best_csv = TABLE_DIR / "privacy_scores_best_conditions.csv"
pivot_csv = TABLE_DIR / "privacy_scores_combo_pivot.csv"
excel_path = OUT_DIR / "privacy_score_results.xlsx"

df_sorted = df.sort_values("privacy_score", ascending=False).reset_index(drop=True)
summary_by_method = (
    df.groupby("method", dropna=False)
    .agg(
        n_conditions=("condition", "count"),
        mean_privacy_score=("privacy_score", "mean"),
        max_privacy_score=("privacy_score", "max"),
        min_privacy_score=("privacy_score", "min"),
        mean_ssim=("ssim_mean", "mean"),
        mean_psnr=("psnr_mean", "mean"),
    )
    .reset_index()
    .sort_values("mean_privacy_score", ascending=False)
)
best_by_method = (
    df.loc[df.groupby("method")["privacy_score"].idxmax()]
    .sort_values("privacy_score", ascending=False)
    .reset_index(drop=True)
)

combo_df = df[df["method"] == "combo"].copy()
if not combo_df.empty:
    combo_pivot = combo_df.pivot(index="block_size", columns="sigma", values="privacy_score")
    combo_pivot = combo_pivot.reindex(index=sorted(BLOCK_SIZES), columns=[float(x) for x in BLUR_SIGMAS])
else:
    combo_pivot = pd.DataFrame()

df_sorted.to_csv(result_csv, index=False, encoding="utf-8-sig")
summary_by_method.to_csv(summary_csv, index=False, encoding="utf-8-sig")
best_by_method.to_csv(best_csv, index=False, encoding="utf-8-sig")
if not combo_pivot.empty:
    combo_pivot.to_csv(pivot_csv, encoding="utf-8-sig")

try:
    with pd.ExcelWriter(excel_path) as writer:
        df_sorted.to_excel(writer, sheet_name="all_scores", index=False)
        summary_by_method.to_excel(writer, sheet_name="summary_by_method", index=False)
        best_by_method.to_excel(writer, sheet_name="best_by_method", index=False)
        if not combo_pivot.empty:
            combo_pivot.to_excel(writer, sheet_name="combo_pivot")
    print("Excel saved:", excel_path)
except Exception as e:
    print("Excel save skipped:", repr(e))

print("CSV saved:")
print(" -", result_csv)
print(" -", summary_csv)
print(" -", best_csv)
if not combo_pivot.empty:
    print(" -", pivot_csv)

display(summary_by_method)
display(df_sorted.head(15))
"""
    ),
    md(
        """
## 9. 그래프 저장

그래프는 `privacy_score_outputs/graphs` 폴더에 PNG로 저장됩니다.
"""
    ),
    code(
        r"""
PALETTE = {
    "score": "#2f9e44",
    "ssim": "#1971c2",
    "psnr": "#e67700",
    "accent": "#9c36b5",
    "muted": "#495057",
}


def save_fig(fig, filename):
    path = GRAPH_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.show()
    print("Saved:", path)
    return path


block_df = df[df["method"] == "block"].copy().sort_values("block_size", ascending=False)
blur_df = df[df["method"] == "blur"].copy().sort_values("sigma")
combo_df = df[df["method"] == "combo"].copy()

# 1. Block-only privacy score
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(block_df["block_size"].astype(int).astype(str), block_df["privacy_score"], marker="o", color=PALETTE["score"], linewidth=2)
ax.set_title("Privacy Score by Block Size")
ax.set_xlabel("Block size")
ax.set_ylabel("Privacy Score (0-100, higher is more private)")
ax.grid(True)
ax.invert_xaxis()
save_fig(fig, "01_block_privacy_score.png")

# 2. Blur-only privacy score
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(blur_df["sigma"].astype(int).astype(str), blur_df["privacy_score"], marker="o", color=PALETTE["score"], linewidth=2)
ax.set_title("Privacy Score by Gaussian Blur Sigma")
ax.set_xlabel("Sigma")
ax.set_ylabel("Privacy Score (0-100, higher is more private)")
ax.grid(True)
save_fig(fig, "02_blur_privacy_score.png")

# 3. Combo heatmap
if not combo_df.empty:
    pivot = combo_df.pivot(index="block_size", columns="sigma", values="privacy_score")
    pivot = pivot.reindex(index=sorted(BLOCK_SIZES), columns=[float(x) for x in BLUR_SIGMAS])
    fig, ax = plt.subplots(figsize=(9, 6.8))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", norm=Normalize(vmin=0, vmax=max(1, np.nanmax(pivot.values))))
    ax.set_title("Block x Blur Privacy Score Heatmap")
    ax.set_xlabel("Blur sigma")
    ax.set_ylabel("Block size")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(x) if float(x).is_integer() else x for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([int(x) for x in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Privacy Score")
    save_fig(fig, "03_combo_privacy_score_heatmap.png")

# 4. SSIM and PSNR reference for block-only
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
x = block_df["block_size"].astype(int).astype(str)
ax1.plot(x, block_df["ssim_mean"], marker="o", color=PALETTE["ssim"], linewidth=2, label="SSIM")
ax2.plot(x, block_df["psnr_mean"], marker="s", color=PALETTE["psnr"], linewidth=2, label="PSNR")
ax1.set_title("Block-only SSIM and PSNR")
ax1.set_xlabel("Block size")
ax1.set_ylabel("SSIM", color=PALETTE["ssim"])
ax2.set_ylabel("PSNR (dB)", color=PALETTE["psnr"])
ax1.invert_xaxis()
ax1.grid(True)
save_fig(fig, "04_block_ssim_psnr_reference.png")

# 5. SSIM and PSNR reference for blur-only
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
x = blur_df["sigma"].astype(int).astype(str)
ax1.plot(x, blur_df["ssim_mean"], marker="o", color=PALETTE["ssim"], linewidth=2, label="SSIM")
ax2.plot(x, blur_df["psnr_mean"], marker="s", color=PALETTE["psnr"], linewidth=2, label="PSNR")
ax1.set_title("Blur-only SSIM and PSNR")
ax1.set_xlabel("Sigma")
ax1.set_ylabel("SSIM", color=PALETTE["ssim"])
ax2.set_ylabel("PSNR (dB)", color=PALETTE["psnr"])
ax1.grid(True)
save_fig(fig, "05_blur_ssim_psnr_reference.png")

# 6. Best conditions
top_n = min(15, len(df_sorted))
top_df = df_sorted.head(top_n).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
colors = [PALETTE["score"] if m == "combo" else PALETTE["accent"] if m == "block" else PALETTE["psnr"] for m in top_df["method"]]
ax.barh(top_df["condition"], top_df["privacy_score"], color=colors)
ax.set_title(f"Top {top_n} Conditions by Privacy Score")
ax.set_xlabel("Privacy Score")
ax.set_ylabel("Condition")
ax.grid(axis="x")
save_fig(fig, "06_top_conditions_privacy_score.png")

# 7. Method distribution
fig, ax = plt.subplots(figsize=(7, 5))
plot_data = [df[df["method"] == m]["privacy_score"].values for m in ["block", "blur", "combo"] if not df[df["method"] == m].empty]
labels = [m for m in ["block", "blur", "combo"] if not df[df["method"] == m].empty]
box = ax.boxplot(plot_data, labels=labels, patch_artist=True)
for patch, color in zip(box["boxes"], [PALETTE["accent"], PALETTE["psnr"], PALETTE["score"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
ax.set_title("Privacy Score Distribution by Method")
ax.set_xlabel("Method")
ax.set_ylabel("Privacy Score")
ax.grid(axis="y")
save_fig(fig, "07_method_distribution_boxplot.png")
"""
    ),
    md(
        """
## 10. 표 이미지 저장

보고서나 발표에 바로 넣을 수 있도록 상위 조건 표를 PNG 이미지로도 저장합니다.
"""
    ),
    code(
        r"""
def save_table_png(table_df, filename, title, max_rows=15):
    view = table_df.head(max_rows).copy()
    keep_cols = ["condition", "method", "block_size", "sigma", "privacy_score", "ssim_mean", "psnr_mean"]
    keep_cols = [c for c in keep_cols if c in view.columns]
    view = view[keep_cols].copy()
    for col in ["privacy_score", "ssim_mean", "psnr_mean"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{x:.3f}")
    for col in ["block_size", "sigma"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.0f}")

    fig_h = max(2.5, 0.38 * len(view) + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    table = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8f9fa")
    path = GRAPH_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved:", path)
    return path


save_table_png(df_sorted, "08_top_conditions_table.png", "Top Conditions by Privacy Score", max_rows=15)
save_table_png(best_by_method, "09_best_by_method_table.png", "Best Condition by Method", max_rows=10)
"""
    ),
    md(
        """
## 11. 샘플 이미지 비교 저장

최고 privacy score 조건 하나를 골라 원본과 변환 결과를 나란히 저장합니다.
"""
    ),
    code(
        r"""
def parse_condition(row):
    method = row["method"]
    block_size = None if pd.isna(row.get("block_size", np.nan)) else int(row["block_size"])
    sigma = None if pd.isna(row.get("sigma", np.nan)) else float(row["sigma"])
    return method, block_size, sigma


sample_path = selected_df.iloc[0]["path"]
sample_original = load_rgb_image(sample_path)
best_row = df_sorted.iloc[0]
best_method, best_bs, best_sg = parse_condition(best_row)
sample_transformed = apply_transform(sample_original, best_method, block_size=best_bs, sigma=best_sg)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].imshow(sample_original)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(sample_transformed)
axes[1].set_title(f"{best_row['condition']}\nScore {best_row['privacy_score']:.2f}")
axes[1].axis("off")
fig.suptitle("Sample Original vs Best Privacy Condition")
sample_compare_path = GRAPH_DIR / "10_sample_original_vs_best_condition.png"
fig.savefig(sample_compare_path, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", sample_compare_path)
"""
    ),
    md(
        """
## 12. 최종 출력 위치

아래 경로의 `tables` 폴더에는 CSV/Excel 표, `graphs` 폴더에는 PNG 그래프가 저장됩니다.
"""
    ),
    code(
        r"""
print("Base train:", BASE_TRAIN)
print("Output:", OUT_DIR)
print("Tables:", TABLE_DIR)
print("Graphs:", GRAPH_DIR)
print("\nImportant files:")
for p in [
    TABLE_DIR / "privacy_scores_long.csv",
    TABLE_DIR / "privacy_scores_summary_by_method.csv",
    TABLE_DIR / "privacy_scores_best_conditions.csv",
    OUT_DIR / "privacy_score_results.xlsx",
    GRAPH_DIR / "03_combo_privacy_score_heatmap.png",
    GRAPH_DIR / "08_top_conditions_table.png",
]:
    print(" -", p)
"""
    ),
]

nbf.write(nb, OUT)
print(OUT)
