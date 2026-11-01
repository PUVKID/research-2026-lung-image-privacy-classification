"""
Block Encryption x Gaussian Blur 100-condition heatmaps
- Accuracy(%) heatmap  (ResNet50)
- Privacy Score heatmap (MS-SSIM 기반)

문서에 너비 70 x 높이 60(mm 기준)으로 삽입해도 글씨가 잘 보이도록
폰트 크기를 크게 키우고, 가로세로 비율(7:6)도 삽입 크기에 맞춰 조정함.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------
# 0. Korean font 등록 (NanumGothic)
# ---------------------------------------------------------
FONT_PATH = "./fonts/NanumGothic-Regular.ttf"
FONT_BOLD_PATH = "./fonts/NanumGothic-Bold.ttf"
fm.fontManager.addfont(FONT_PATH)
fm.fontManager.addfont(FONT_BOLD_PATH)
plt.rcParams["font.family"] = [fm.FontProperties(fname=FONT_PATH).get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------
# 1. 데이터 (표 3, 표 5)
#    행: Block size = 512 -> 1 (위에서 아래로)
#    열: Gaussian Blur sigma = 0,1,2,4,8,16,32,64,128,256
# ---------------------------------------------------------
block_sizes = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
sigmas = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

accuracy = np.array([
    [98.2, 93.6, 82.6, 87.2, 81.7, 67.0, 59.6, 51.4, 51.4, 46.8],  # 512
    [98.2, 93.6, 90.8, 86.2, 80.7, 62.4, 58.7, 51.4, 51.4, 51.4],  # 256
    [94.5, 90.8, 84.4, 84.4, 75.2, 54.1, 52.3, 51.4, 51.4, 51.4],  # 128
    [96.3, 89.0, 86.2, 71.6, 51.4, 51.4, 51.4, 51.4, 51.4, 57.8],  # 64
    [89.9, 84.4, 71.6, 70.6, 51.4, 49.5, 51.4, 51.4, 51.4, 51.4],  # 32
    [83.5, 77.1, 67.0, 57.8, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4],  # 16
    [75.2, 68.8, 60.5, 52.3, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4],  # 8
    [64.2, 57.8, 55.0, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4],  # 4
    [66.1, 56.9, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4],  # 2
    [47.7, 50.5, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4, 51.4],  # 1
])

privacy = np.array([
    [0.0,  0.8,  2.5,  6.1,  11.4, 17.7, 24.8, 30.3, 32.4, 34.6],  # 512
    [49.6, 49.6, 49.6, 49.6, 49.6, 49.7, 49.9, 49.4, 48.5, 35.7],  # 256
    [46.0, 45.9, 45.8, 45.7, 45.8, 46.7, 48.0, 48.3, 41.9, 35.4],  # 128
    [47.7, 47.6, 47.5, 47.4, 47.2, 46.4, 44.4, 39.2, 36.4, 35.2],  # 64
    [47.4, 47.3, 47.1, 46.9, 46.1, 43.8, 39.0, 35.7, 35.3, 35.2],  # 32
    [49.2, 49.1, 49.0, 48.6, 47.5, 43.8, 38.7, 35.7, 35.1, 35.1],  # 16
    [49.6, 49.4, 49.2, 48.6, 46.0, 41.2, 37.3, 36.0, 35.7, 35.2],  # 8
    [49.3, 48.9, 48.1, 45.9, 41.5, 37.3, 35.6, 35.1, 35.1, 35.1],  # 4
    [48.8, 47.7, 45.9, 42.2, 38.1, 36.0, 35.4, 35.2, 35.0, 35.1],  # 2
    [47.1, 44.9, 41.9, 38.2, 36.1, 35.4, 35.4, 35.5, 35.5, 35.2],  # 1
])


def draw_heatmap(data, cmap, vmin, vmax, cbar_label, fname, text_threshold_frac=0.5):
    # 문서 삽입 크기(너비 70 x 높이 60)와 동일한 7:6 비율로 그려서
    # 축소돼도 글자가 비율 그대로 작아지지 않고 레이아웃이 안 깨지게 함
    fig, ax = plt.subplots(figsize=(9.8, 8.4))

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # 축 눈금 (제목 없이, 글씨 크고 볼드하게)
    ax.set_xticks(np.arange(len(sigmas)))
    ax.set_xticklabels(sigmas, fontsize=19, fontweight="bold")
    ax.set_yticks(np.arange(len(block_sizes)))
    ax.set_yticklabels(block_sizes, fontsize=19, fontweight="bold")

    ax.set_xlabel("Gaussian Blur σ", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("Block size (pixels)", fontsize=25, fontweight="bold", labelpad=12)
    # 상단 제목은 표시하지 않음

    # 셀 안에 값 표시 (크고 볼드하게)
    norm_range = vmax - vmin
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            norm_val = (val - vmin) / norm_range if norm_range else 0
            text_color = "white" if norm_val < text_threshold_frac else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                     color=text_color, fontsize=32, fontweight="bold")

    # 컬러바
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=21, fontweight="bold", labelpad=12)
    cbar.ax.tick_params(labelsize=17)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    # 격자선
    ax.set_xticks(np.arange(-0.5, len(sigmas), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(block_sizes), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"saved: {fname}")


# ---------------------------------------------------------
# 2. Accuracy(%) 히트맵  (ResNet50) — 초록(높음) ~ 빨강(낮음)
# ---------------------------------------------------------
draw_heatmap(
    accuracy,
    cmap="RdYlGn",
    vmin=45, vmax=100,
    cbar_label="Accuracy (%)",
    fname="./accuracy_heatmap.png",
    text_threshold_frac=0.35,
)

# ---------------------------------------------------------
# 3. Privacy Score 히트맵 (MS-SSIM 기반) — viridis
# ---------------------------------------------------------
draw_heatmap(
    privacy,
    cmap="viridis",
    vmin=0, vmax=50,
    cbar_label="Privacy Score (0-50)",
    fname="./privacy_heatmap.png",
    text_threshold_frac=0.55,
)
