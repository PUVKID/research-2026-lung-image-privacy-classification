import copy
import json
import shutil
from datetime import datetime
from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path(r".\privacy_score_ssim_psnr (1).ipynb")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def set_cell_source(nb, index, source):
    nb.cells[index].source = source.strip() + "\n"


backup = NOTEBOOK.with_name(
    NOTEBOOK.stem + "_backup_before_ssim_only_" + datetime.now().strftime("%Y%m%d_%H%M%S") + NOTEBOOK.suffix
)
shutil.copy2(NOTEBOOK, backup)

nb = nbf.read(NOTEBOOK, as_version=4)

set_cell_source(
    nb,
    0,
    """
<div style="background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); border: 2px solid #30363d; border-radius: 12px; padding: 30px; margin-bottom: 20px;">

<h1 style="color: #f0f6fc; text-align: center; font-size: 2em; margin: 0 0 12px 0;">
  Privacy Score 정량화 실험 - SSIM 기반, PSNR 참고 지표
</h1>

<p style="color: #8b949e; text-align: center; margin: 0;">
블록 픽셀 크기별 / 가우시안 블러별 / 블록+블러 조합별로 원본 대비 구조적 차이를 계산하고<br>
SSIM 기반 Privacy Score를 표와 상세 그래프로 저장합니다.
</p>

</div>
""",
)

set_cell_source(
    nb,
    1,
    r"""
## 사용한 Privacy Score 수식

이 노트북의 최종 privacy score는 **SSIM만 사용**합니다. PSNR은 점수 계산에 넣지 않고, 이미지 손상 정도를 해석하기 위한 참고 지표로만 표와 그래프에 함께 표시합니다.

원본 이미지 \(I\), 변환 이미지 \(I'\)에 대해:

\[
SSIM_i = SSIM(I_i, I'_i)
\]

\[
DSSIM_i = \frac{1 - SSIM_i}{2}
\]

\[
PrivacyScore_i = 100 \times DSSIM_i
               = 50 \times (1 - SSIM_i)
\]

조건별 최종 점수는 이미지별 점수의 평균입니다.

\[
PrivacyScore_{condition} = \frac{1}{N}\sum_{i=1}^{N} PrivacyScore_i
\]

- 점수 범위: 이론적으로 0~100점
- 0점에 가까움: 원본과 거의 동일
- 100점에 가까움: 원본과 구조적으로 매우 다름
- PSNR: `psnr_mean`, `psnr_std`로만 저장하고 참고 그래프에만 사용
""",
)

set_cell_source(
    nb,
    3,
    """
# Cell 1: 라이브러리 임포트
import os, glob, json, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

try:
    from IPython.display import display
except Exception:
    display = print
""",
)

set_cell_source(
    nb,
    7,
    """
# Cell 3: 실험 설정 - 경로는 기존 노트북 값을 그대로 사용
BASE_TRAIN   = "/workspace/Train"      # 원본 이미지 루트. Docker/VSCode 컨테이너 기준 경로
IMG_SIZE     = 512
SEED         = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

BLOCK_SIZES  = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
BLUR_SIGMAS  = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

# 속도 조절용. None이면 전체 이미지 사용, 예: 20이면 클래스별 최대 20장만 사용
N_SAMPLES_PER_CLASS = None

# 기존 출력 경로 유지
OUT_DIR   = "./privacy_score_outputs"
GRAPH_DIR = os.path.join(OUT_DIR, "eval_graphs_privacy_score")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
SAMPLE_IMG_DIR = os.path.join(OUT_DIR, "sample_images")

for _dir in (OUT_DIR, GRAPH_DIR, TABLE_DIR, SAMPLE_IMG_DIR):
    os.makedirs(_dir, exist_ok=True)
for _sub in ("block", "blur", "combo"):
    os.makedirs(os.path.join(SAMPLE_IMG_DIR, _sub), exist_ok=True)

print(f"BASE_TRAIN: {BASE_TRAIN}")
print(f"OUT_DIR   : {OUT_DIR}")
print(f"GRAPH_DIR : {GRAPH_DIR}")
print(f"TABLE_DIR : {TABLE_DIR}")
print(f"SAMPLE_IMG_DIR: {SAMPLE_IMG_DIR}")
print(f"실험 조건: 블록 {len(BLOCK_SIZES)}개, 블러 {len(BLUR_SIGMAS)}개, 조합 {len(BLOCK_SIZES) * len(BLUR_SIGMAS)}개")
""",
)

set_cell_source(
    nb,
    11,
    """
# Cell 5: 원본 이미지 로딩
def _is_image_file(path):
    return str(path).lower().endswith(IMAGE_EXTENSIONS)

def _is_ignored_path(path):
    parts = set(os.path.normpath(path).split(os.sep))
    ignored = {"privacy_score_outputs", "eval_graphs_privacy_score", "sample_images", ".ipynb_checkpoints", "__pycache__"}
    return bool(parts & ignored)

def collect_image_files(base_dir):
    records = []
    for root, _, files in os.walk(base_dir):
        if _is_ignored_path(root):
            continue
        for f in sorted(files):
            if not _is_image_file(f):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, base_dir)
            first_part = rel.split(os.sep)[0]
            label = first_part if first_part != os.path.basename(rel) else "root"
            records.append({"path": path, "relative_path": rel, "label": label})
    return pd.DataFrame(records)

def load_original_images(base_dir, n_per_class=None, seed=SEED):
    assert os.path.exists(base_dir), f"BASE_TRAIN 경로가 없습니다: {base_dir}"

    image_table = collect_image_files(base_dir)
    assert len(image_table) > 0, f"이미지 파일을 찾을 수 없습니다: {base_dir}"

    print(f"이미지 파일 발견: {len(image_table)}장")
    class_counts = image_table.groupby("label").size().rename("n_images").reset_index()
    display(class_counts)

    if n_per_class is not None:
        image_table = (
            image_table.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), n_per_class), random_state=seed))
            .reset_index(drop=True)
        )
        print(f"샘플링 후 이미지: {len(image_table)}장")

    image_table = image_table.sort_values(["label", "relative_path"]).reset_index(drop=True)
    image_table.to_csv(os.path.join(TABLE_DIR, "selected_images.csv"), index=False, encoding="utf-8-sig")

    images = []
    failed = []
    for _, row in image_table.iterrows():
        try:
            img = Image.open(row["path"])
            img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.uint8))
        except Exception as e:
            failed.append((row["path"], str(e)))

    if failed:
        print(f"로드 실패 {len(failed)}장")
        for path, err in failed[:10]:
            print(f"  - {path}: {err}")

    print(f"원본 이미지 {len(images)}장 로드 완료 (경로: {base_dir})")
    return images, image_table

images, image_table = load_original_images(BASE_TRAIN, n_per_class=N_SAMPLES_PER_CLASS, seed=SEED)
assert len(images) > 0, f"원본 이미지를 찾을 수 없습니다: {BASE_TRAIN}"

Image.fromarray(images[0]).save(os.path.join(SAMPLE_IMG_DIR, "00_original_sample.png"))
print(f"원본 샘플 이미지 저장: {os.path.join(SAMPLE_IMG_DIR, '00_original_sample.png')}")
""",
)

set_cell_source(
    nb,
    13,
    r"""
# Cell 6: SSIM 기반 Privacy Score + PSNR 참고 지표 계산
# 최종 점수식:
#   DSSIM_i = (1 - SSIM_i) / 2
#   PrivacyScore_i = 100 * DSSIM_i = 50 * (1 - SSIM_i)
# 조건별 최종 점수:
#   PrivacyScore_condition = mean(PrivacyScore_i)
# PSNR은 privacy_score 계산에 사용하지 않고 참고 지표로만 저장한다.
def compute_ssim_psnr(orig, transformed):
    s = ssim_fn(orig, transformed, channel_axis=-1, data_range=255)
    try:
        p = psnr_fn(orig, transformed, data_range=255)
        if np.isinf(p):
            p = 100.0
    except Exception:
        p = 100.0
    return float(s), float(p)

def privacy_score_from_ssim(ssim_value):
    dssim = np.clip((1.0 - float(ssim_value)) / 2.0, 0.0, 1.0)
    return float(dssim * 100.0)

def evaluate_condition(images, transform_fn, **kwargs):
    ssims, psnrs, scores = [], [], []
    for img in images:
        t = transform_fn(img, **kwargs)
        s, p = compute_ssim_psnr(img, t)
        ssims.append(s)
        psnrs.append(p)
        scores.append(privacy_score_from_ssim(s))

    return {
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "dssim_mean": float(np.mean(scores) / 100.0),
        "privacy_score": float(np.mean(scores)),
        "privacy_score_std": float(np.std(scores)),
    }

print("SSIM 기반 Privacy Score 함수 정의 완료")
print("Privacy Score = 100 * ((1 - SSIM) / 2)")
print("PSNR은 psnr_mean/psnr_std 참고 지표로만 저장")
""",
)

set_cell_source(
    nb,
    15,
    """
# Cell 7: 블록 암호화 단독
rows = []
print("블록 암호화 단독 SSIM/PSNR 계산 중...")
for bs in BLOCK_SIZES:
    m = evaluate_condition(images, block_scramble, seed=SEED, block_size=bs)
    rows.append({"condition": f"B{bs}_S0", "type": "block", "block_size": bs, "sigma": 0, **m})
    print(
        f"  Block{bs:>4}: Score={m['privacy_score']:6.2f}/100  "
        f"SSIM={m['ssim_mean']:.4f}  PSNR(ref)={m['psnr_mean']:6.2f}dB"
    )

    sample = block_scramble(images[0], SEED, bs)
    Image.fromarray(sample).save(os.path.join(SAMPLE_IMG_DIR, "block", f"B{bs}_sample.png"))

print(f"블록 샘플 이미지 {len(BLOCK_SIZES)}장 저장: {os.path.join(SAMPLE_IMG_DIR, 'block')}")
""",
)

set_cell_source(
    nb,
    17,
    """
# Cell 8: 가우시안 블러 단독
print("가우시안 블러 단독 SSIM/PSNR 계산 중...")
for sg in BLUR_SIGMAS:
    m = evaluate_condition(images, apply_gaussian_blur, sigma=sg)
    rows.append({"condition": f"B512_S{sg}", "type": "blur", "block_size": 512, "sigma": sg, **m})
    print(
        f"  Sigma{sg:>4}: Score={m['privacy_score']:6.2f}/100  "
        f"SSIM={m['ssim_mean']:.4f}  PSNR(ref)={m['psnr_mean']:6.2f}dB"
    )

    sample = apply_gaussian_blur(images[0], sg)
    Image.fromarray(sample).save(os.path.join(SAMPLE_IMG_DIR, "blur", f"S{sg}_sample.png"))

print(f"블러 샘플 이미지 {len(BLUR_SIGMAS)}장 저장: {os.path.join(SAMPLE_IMG_DIR, 'blur')}")
""",
)

set_cell_source(
    nb,
    19,
    """
# Cell 9: 블록+블러 조합 (10 x 10 = 100개)
print("블록+블러 조합 SSIM/PSNR 계산 중...")
for bs in BLOCK_SIZES:
    for sg in BLUR_SIGMAS:
        def combo_fn(img, block_size=bs, sigma=sg):
            return apply_gaussian_blur(block_scramble(img, SEED, block_size), sigma)

        m = evaluate_condition(images, combo_fn)
        rows.append({"condition": f"B{bs}_S{sg}", "type": "combo", "block_size": bs, "sigma": sg, **m})
        print(
            f"  B{bs:>4}_S{sg:<4}: Score={m['privacy_score']:6.2f}/100  "
            f"SSIM={m['ssim_mean']:.4f}  PSNR(ref)={m['psnr_mean']:6.2f}dB"
        )

        sample = combo_fn(images[0])
        Image.fromarray(sample).save(os.path.join(SAMPLE_IMG_DIR, "combo", f"B{bs}_S{sg}_sample.png"))

print(f"조합 샘플 이미지 {len(BLOCK_SIZES) * len(BLUR_SIGMAS)}장 저장: {os.path.join(SAMPLE_IMG_DIR, 'combo')}")
print(f"전체 {len(rows)}개 조건 계산 완료")
""",
)

set_cell_source(
    nb,
    21,
    """
# Cell 10: 결과 DataFrame 생성 및 저장
df = pd.DataFrame(rows)

preferred_cols = [
    "condition", "type", "block_size", "sigma", "privacy_score", "privacy_score_std",
    "dssim_mean", "ssim_mean", "ssim_std", "psnr_mean", "psnr_std"
]
df = df[[c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]]
df_sorted = df.sort_values("privacy_score", ascending=False).reset_index(drop=True)

summary_by_type = (
    df.groupby("type")
      .agg(
          n_conditions=("condition", "count"),
          score_mean=("privacy_score", "mean"),
          score_max=("privacy_score", "max"),
          score_min=("privacy_score", "min"),
          ssim_mean=("ssim_mean", "mean"),
          psnr_mean_ref=("psnr_mean", "mean"),
      )
      .reset_index()
      .sort_values("score_mean", ascending=False)
)

best_by_type = (
    df.loc[df.groupby("type")["privacy_score"].idxmax()]
      .sort_values("privacy_score", ascending=False)
      .reset_index(drop=True)
)

df_sorted.to_csv(os.path.join(OUT_DIR, "privacy_scores.csv"), index=False, encoding="utf-8-sig")
df_sorted.to_json(os.path.join(OUT_DIR, "privacy_scores.json"), orient="records", force_ascii=False, indent=2)
summary_by_type.to_csv(os.path.join(TABLE_DIR, "summary_by_type.csv"), index=False, encoding="utf-8-sig")
best_by_type.to_csv(os.path.join(TABLE_DIR, "best_by_type.csv"), index=False, encoding="utf-8-sig")

try:
    with pd.ExcelWriter(os.path.join(OUT_DIR, "privacy_scores.xlsx")) as writer:
        df_sorted.to_excel(writer, sheet_name="all_scores", index=False)
        summary_by_type.to_excel(writer, sheet_name="summary_by_type", index=False)
        best_by_type.to_excel(writer, sheet_name="best_by_type", index=False)
    print(f"Excel 저장 완료: {os.path.join(OUT_DIR, 'privacy_scores.xlsx')}")
except Exception as e:
    print(f"Excel 저장은 건너뜀: {e}")

print(f"프라이버시 점수 저장 완료: {os.path.join(OUT_DIR, 'privacy_scores.csv')}")
print(f"요약표 저장 완료: {TABLE_DIR}")
display(summary_by_type)
display(df_sorted.head(15))
""",
)

new_graph_cells = [
    md(
        """
## 11. 상세 그래프 생성

아래 그래프들은 `GRAPH_DIR`에 PNG로 저장됩니다. Privacy Score는 SSIM 기반 점수이며, PSNR은 참고 지표로만 함께 표시합니다.
"""
    ),
    code(
        """
# Cell 11: 상세 그래프 공통 설정
ORANGE, CYAN, GREEN, RED, GRAY, PURPLE, BLUE = "#f08c00", "#1098ad", "#2f9e44", "#e03131", "#868e96", "#7048e8", "#1971c2"

def _save(fig, name):
    path = os.path.join(GRAPH_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"저장: {path}")

def _annotate_bars(ax, fmt="{:.1f}", dy=0.5, fontsize=8):
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )

block_df = df[df["type"] == "block"].sort_values("block_size", ascending=False).copy()
blur_df  = df[df["type"] == "blur"].sort_values("sigma").copy()
combo_df = df[df["type"] == "combo"].copy()

pivot_score = combo_df.pivot(index="block_size", columns="sigma", values="privacy_score").reindex(index=BLOCK_SIZES[::-1], columns=BLUR_SIGMAS)
pivot_ssim  = combo_df.pivot(index="block_size", columns="sigma", values="ssim_mean").reindex(index=BLOCK_SIZES[::-1], columns=BLUR_SIGMAS)
pivot_psnr  = combo_df.pivot(index="block_size", columns="sigma", values="psnr_mean").reindex(index=BLOCK_SIZES[::-1], columns=BLUR_SIGMAS)
"""
    ),
    code(
        """
# 01. 블록 픽셀 크기별 Privacy Score + SSIM + PSNR 참고
fig, ax1 = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(block_df))
labels = block_df["block_size"].astype(int).astype(str)
bars = ax1.bar(x, block_df["privacy_score"], color=GREEN, alpha=0.82, label="Privacy Score")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel("Block size (pixels)")
ax1.set_ylabel("Privacy Score (0-100, SSIM only)", color=GREEN)
ax1.set_ylim(0, max(5, block_df["privacy_score"].max() * 1.18))
ax1.grid(axis="y")
_annotate_bars(ax1, "{:.1f}", dy=max(0.25, block_df["privacy_score"].max() * 0.015))

ax2 = ax1.twinx()
ax2.plot(x, block_df["ssim_mean"], "o-", color=CYAN, linewidth=2, label="SSIM")
ax2.plot(x, block_df["psnr_mean"] / 100.0, "s--", color=ORANGE, linewidth=1.8, label="PSNR/100 ref")
ax2.set_ylabel("SSIM / PSNR divided by 100", color=CYAN)
ax2.set_ylim(0, max(1.05, block_df["ssim_mean"].max() * 1.1))

lines1, names1 = ax1.get_legend_handles_labels()
lines2, names2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, names1 + names2, loc="upper left")
ax1.set_title("01. Block Size Detail: Privacy Score from SSIM, PSNR as Reference")
_save(fig, "01_block_detail_score_ssim_psnr_ref.png")
"""
    ),
    code(
        """
# 02. 블러 sigma별 Privacy Score + SSIM + PSNR 참고
fig, ax1 = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(blur_df))
labels = blur_df["sigma"].astype(int).astype(str)
bars = ax1.bar(x, blur_df["privacy_score"], color=GREEN, alpha=0.82, label="Privacy Score")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel("Gaussian blur sigma")
ax1.set_ylabel("Privacy Score (0-100, SSIM only)", color=GREEN)
ax1.set_ylim(0, max(5, blur_df["privacy_score"].max() * 1.18))
ax1.grid(axis="y")
_annotate_bars(ax1, "{:.1f}", dy=max(0.25, blur_df["privacy_score"].max() * 0.015))

ax2 = ax1.twinx()
ax2.plot(x, blur_df["ssim_mean"], "o-", color=CYAN, linewidth=2, label="SSIM")
ax2.plot(x, blur_df["psnr_mean"] / 100.0, "s--", color=ORANGE, linewidth=1.8, label="PSNR/100 ref")
ax2.set_ylabel("SSIM / PSNR divided by 100", color=CYAN)
ax2.set_ylim(0, max(1.05, blur_df["ssim_mean"].max() * 1.1))

lines1, names1 = ax1.get_legend_handles_labels()
lines2, names2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, names1 + names2, loc="upper left")
ax1.set_title("02. Blur Sigma Detail: Privacy Score from SSIM, PSNR as Reference")
_save(fig, "02_blur_detail_score_ssim_psnr_ref.png")
"""
    ),
    code(
        """
# 03. 블록+블러 조합 Privacy Score 히트맵 - 셀마다 점수 표시
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(pivot_score.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(BLUR_SIGMAS)))
ax.set_xticklabels(BLUR_SIGMAS)
ax.set_yticks(range(len(BLOCK_SIZES)))
ax.set_yticklabels(BLOCK_SIZES[::-1])
ax.set_xlabel("Gaussian blur sigma")
ax.set_ylabel("Block size (pixels)")
ax.set_title("03. Block + Blur Combination Privacy Score Heatmap")
for i in range(pivot_score.shape[0]):
    for j in range(pivot_score.shape[1]):
        value = pivot_score.values[i, j]
        color = "white" if value > np.nanmax(pivot_score.values) * 0.45 else "black"
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color=color)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Privacy Score (0-100, SSIM only)")
_save(fig, "03_combo_privacy_score_heatmap_annotated.png")
"""
    ),
    code(
        """
# 04. 조합 조건 SSIM 히트맵 - score의 원천 지표 확인
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(pivot_ssim.values, cmap="magma_r", aspect="auto")
ax.set_xticks(range(len(BLUR_SIGMAS)))
ax.set_xticklabels(BLUR_SIGMAS)
ax.set_yticks(range(len(BLOCK_SIZES)))
ax.set_yticklabels(BLOCK_SIZES[::-1])
ax.set_xlabel("Gaussian blur sigma")
ax.set_ylabel("Block size (pixels)")
ax.set_title("04. Block + Blur SSIM Mean Heatmap")
for i in range(pivot_ssim.shape[0]):
    for j in range(pivot_ssim.shape[1]):
        value = pivot_ssim.values[i, j]
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.5, color="white" if value < 0.45 else "black")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("SSIM mean")
_save(fig, "04_combo_ssim_heatmap_annotated.png")
"""
    ),
    code(
        """
# 05. 조합 조건 PSNR 참고 히트맵 - 점수에는 미사용
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(pivot_psnr.values, cmap="cividis", aspect="auto")
ax.set_xticks(range(len(BLUR_SIGMAS)))
ax.set_xticklabels(BLUR_SIGMAS)
ax.set_yticks(range(len(BLOCK_SIZES)))
ax.set_yticklabels(BLOCK_SIZES[::-1])
ax.set_xlabel("Gaussian blur sigma")
ax.set_ylabel("Block size (pixels)")
ax.set_title("05. Block + Blur PSNR Reference Heatmap (not used in score)")
for i in range(pivot_psnr.shape[0]):
    for j in range(pivot_psnr.shape[1]):
        value = pivot_psnr.values[i, j]
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.5, color="white" if value < np.nanmedian(pivot_psnr.values) else "black")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("PSNR mean (dB, reference only)")
_save(fig, "05_combo_psnr_reference_heatmap_annotated.png")
"""
    ),
    code(
        """
# 06. 전체 조건 Privacy Score 상위 랭킹
top_n = min(20, len(df_sorted))
top_df = df_sorted.head(top_n).iloc[::-1].copy()
method_colors = {"block": PURPLE, "blur": ORANGE, "combo": GREEN}
fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.36)))
ax.barh(top_df["condition"], top_df["privacy_score"], color=[method_colors.get(t, GRAY) for t in top_df["type"]])
ax.set_xlabel("Privacy Score (0-100, SSIM only)")
ax.set_ylabel("Condition")
ax.set_title(f"06. Top {top_n} Privacy Conditions")
ax.grid(axis="x")
for y, value in enumerate(top_df["privacy_score"]):
    ax.text(value + max(0.2, df_sorted["privacy_score"].max() * 0.01), y, f"{value:.2f}", va="center", fontsize=8)
_save(fig, "06_top_conditions_privacy_score.png")
"""
    ),
    code(
        """
# 07. SSIM과 Privacy Score 관계 - SSIM만으로 점수가 정해짐
fig, ax = plt.subplots(figsize=(8, 6))
for t, color in method_colors.items():
    sub = df[df["type"] == t]
    ax.scatter(sub["ssim_mean"], sub["privacy_score"], label=t, color=color, s=60, alpha=0.78, edgecolors="white")
ax.set_xlabel("SSIM mean")
ax.set_ylabel("Privacy Score = 50 x (1 - SSIM)")
ax.set_title("07. SSIM Determines Privacy Score")
ax.grid(True)
ax.legend(title="type")
for _, r in df_sorted.head(8).iterrows():
    ax.annotate(r["condition"], (r["ssim_mean"], r["privacy_score"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
_save(fig, "07_ssim_vs_privacy_score.png")
"""
    ),
    code(
        """
# 08. PSNR 참고 지표와 Privacy Score 관계 - 점수 계산에는 미사용
fig, ax = plt.subplots(figsize=(8, 6))
for t, color in method_colors.items():
    sub = df[df["type"] == t]
    ax.scatter(sub["psnr_mean"], sub["privacy_score"], label=t, color=color, s=60, alpha=0.78, edgecolors="white")
ax.axvline(40, color=RED, linestyle="--", linewidth=1, label="40 dB reference")
ax.axvline(15, color=BLUE, linestyle="--", linewidth=1, label="15 dB reference")
ax.set_xlabel("PSNR mean (dB, reference only)")
ax.set_ylabel("Privacy Score (0-100, SSIM only)")
ax.set_title("08. PSNR Reference vs Privacy Score")
ax.grid(True)
ax.legend(fontsize=8)
_save(fig, "08_psnr_reference_vs_privacy_score.png")
"""
    ),
    code(
        """
# 09. 변환 유형별 Privacy Score 분포
fig, ax = plt.subplots(figsize=(8, 5.5))
data = [df[df["type"] == t]["privacy_score"].values for t in ["block", "blur", "combo"]]
bp = ax.boxplot(data, labels=["block", "blur", "combo"], patch_artist=True, showmeans=True)
for patch, color in zip(bp["boxes"], [PURPLE, ORANGE, GREEN]):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
ax.set_ylabel("Privacy Score (0-100, SSIM only)")
ax.set_title("09. Privacy Score Distribution by Transform Type")
ax.grid(axis="y")
_save(fig, "09_type_distribution_boxplot.png")
"""
    ),
    code(
        """
# 10. 상위 조건 표를 이미지로 저장
table_df = df_sorted.head(15)[["condition", "type", "block_size", "sigma", "privacy_score", "ssim_mean", "psnr_mean"]].copy()
for col, digits in [("privacy_score", 2), ("ssim_mean", 4), ("psnr_mean", 2)]:
    table_df[col] = table_df[col].map(lambda x: f"{x:.{digits}f}")
for col in ["block_size", "sigma"]:
    table_df[col] = table_df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.0f}")

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.axis("off")
ax.set_title("10. Top 15 Conditions Table", fontsize=14, pad=14)
tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.25)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dee2e6")
        cell.set_text_props(weight="bold")
    else:
        cell.set_facecolor("#ffffff" if r % 2 else "#f8f9fa")
_save(fig, "10_top15_conditions_table.png")
"""
    ),
    code(
        """
# 11. 원본 vs 각 유형 최고 조건 샘플 비교
def _combo_transform(img, block_size, sigma):
    return apply_gaussian_blur(block_scramble(img, SEED, int(block_size)), float(sigma))

sample_original = images[0]
best_rows = best_by_type.sort_values("type").reset_index(drop=True)

fig, axes = plt.subplots(1, len(best_rows) + 1, figsize=(4.2 * (len(best_rows) + 1), 4.5))
axes[0].imshow(sample_original)
axes[0].set_title("Original")
axes[0].axis("off")

for ax, (_, r) in zip(axes[1:], best_rows.iterrows()):
    if r["type"] == "block":
        transformed = block_scramble(sample_original, SEED, int(r["block_size"]))
    elif r["type"] == "blur":
        transformed = apply_gaussian_blur(sample_original, float(r["sigma"]))
    else:
        transformed = _combo_transform(sample_original, r["block_size"], r["sigma"])
    ax.imshow(transformed)
    ax.set_title(f"{r['type']} best\\n{r['condition']}\\nScore {r['privacy_score']:.2f}")
    ax.axis("off")

fig.suptitle("11. Sample Comparison: Original vs Best Conditions", fontsize=14)
_save(fig, "11_sample_original_vs_best_conditions.png")
"""
    ),
    code(
        """
# 12. 종합 대시보드
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig)

ax = fig.add_subplot(gs[0, 0])
ax.plot(block_df["block_size"].astype(str), block_df["privacy_score"], "o-", color=GREEN)
ax.set_title("Block score")
ax.set_xlabel("block size")
ax.set_ylabel("score")
ax.invert_xaxis()
ax.grid(True)

ax = fig.add_subplot(gs[0, 1])
ax.plot(blur_df["sigma"].astype(str), blur_df["privacy_score"], "o-", color=GREEN)
ax.set_title("Blur score")
ax.set_xlabel("sigma")
ax.set_ylabel("score")
ax.grid(True)

ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(pivot_score.values, cmap="viridis", aspect="auto")
ax.set_title("Combo score heatmap")
ax.set_xticks(range(len(BLUR_SIGMAS)))
ax.set_xticklabels(BLUR_SIGMAS, fontsize=7)
ax.set_yticks(range(len(BLOCK_SIZES)))
ax.set_yticklabels(BLOCK_SIZES[::-1], fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(gs[1, 0])
for t, color in method_colors.items():
    sub = df[df["type"] == t]
    ax.scatter(sub["ssim_mean"], sub["privacy_score"], label=t, color=color, s=35, alpha=0.75)
ax.set_title("SSIM vs score")
ax.set_xlabel("SSIM")
ax.set_ylabel("score")
ax.grid(True)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
for t, color in method_colors.items():
    sub = df[df["type"] == t]
    ax.scatter(sub["psnr_mean"], sub["privacy_score"], label=t, color=color, s=35, alpha=0.75)
ax.set_title("PSNR reference vs score")
ax.set_xlabel("PSNR dB")
ax.set_ylabel("score")
ax.grid(True)

ax = fig.add_subplot(gs[1, 2])
ax.hist(combo_df["privacy_score"], bins=18, color=GRAY, edgecolor="white")
ax.set_title("Combo score distribution")
ax.set_xlabel("score")
ax.set_ylabel("count")
ax.grid(axis="y")

fig.suptitle("12. Privacy Score Dashboard - SSIM Only Score, PSNR Reference", fontsize=16)
_save(fig, "12_dashboard_detailed.png")

print(f"상세 그래프 저장 완료: {GRAPH_DIR}")
"""
    ),
]

nb.cells = nb.cells[:22] + new_graph_cells

# Clear old outputs/execution counts so the edited notebook runs cleanly.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

nbf.write(nb, NOTEBOOK)
print(json.dumps({"notebook": str(NOTEBOOK), "backup": str(backup), "cells": len(nb.cells)}, ensure_ascii=False, indent=2))
