const fs = require("fs");
const path = require("path");

const workspace = ".";
const inputs = {
  blur: path.join(workspace, "blur_resnet50_same_settings_seq.ipynb"),
  block: path.join(workspace, "block_encryption_resnet50_same_settings_seq.ipynb"),
};
const outputs = {
  blur: path.join(workspace, "blur_resnet50_metric_matched_seq.ipynb"),
  block: path.join(workspace, "block_encryption_resnet50_metric_matched_seq.ipynb"),
};

function asText(source) {
  return Array.isArray(source) ? source.join("") : (source || "");
}

function asSourceArray(text) {
  const parts = text.split("\n");
  if (parts[parts.length - 1] === "") parts.pop();
  return parts.map((line) => `${line}\n`);
}

function setCell(nb, index, text) {
  nb.cells[index].source = asSourceArray(text);
  nb.cells[index].outputs = [];
  nb.cells[index].execution_count = null;
}

const blurConfig = `# Cell 4: 실험 설정
# 블록 변형의 평균 SSIM/PSNR에 맞춰 블러 sigma를 자동 보정
BLOCK_SIZES  = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
DEFAULT_BLUR_SIGMAS = [0 if bs >= 512 else 512 // (2 * bs) for bs in BLOCK_SIZES]
BLUR_SIGMAS  = DEFAULT_BLUR_SIGMAS.copy()

BASE_TRAIN   = "/workspace/Train"
WORKSPACE    = "/workspace"
IMG_SIZE     = 512
BATCH_SIZE   = 4
EPOCHS_S1    = 30
EPOCHS_S2    = 60
NUM_CLASSES  = 3
SEED         = 42
N_FOLDS      = 7
TTA_STEPS    = 20
SWA_START    = 20
TARGET_COUNT = 1000
TEST_RATIO   = 0.10
AUTOTUNE     = tf.data.AUTOTUNE

# 같은 설정으로 새 결과/분할을 만들기 위한 버전 태그
EXPERIMENT_VERSION = "metric_matched_v1"
ORIGINAL_SPLIT_DIR = os.path.join(
    WORKSPACE, f"AllExp_Split_ORIGINAL_seed{SEED}_{EXPERIMENT_VERSION}"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# 지표 매칭 설정: 샘플 수를 늘리면 더 정확하지만 보정 시간이 길어짐
CALIBRATION_MAX_PER_CLASS = 5
CALIBRATION_SIGMA_CANDIDATES = (
    [0] +
    list(range(1, 65)) +
    list(range(66, 129, 2)) +
    list(range(132, 257, 4))
)
CALIBRATION_CACHE_PATH = os.path.join(
    WORKSPACE, f"blur_block_metric_match_{EXPERIMENT_VERSION}.json"
)
BLUR_MATCH_TABLE = []

# 학습 히스토리 저장소 (과적합/과소적합 진단용)
ALL_HISTORIES = {}   # { sigma: { fold: history_dict } }

def set_reproducible(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def list_class_dirs(directory):
    return sorted([d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))])

def list_image_files(directory):
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory)
                   if f.lower().endswith(IMAGE_EXTENSIONS)])

set_reproducible(SEED)

print("┌" + "─" * 62 + "┐")
print("│           ✅ 실험 설정 완료                                  │")
print("├" + "─" * 62 + "┤")
print(f"│  이미지 크기     : {IMG_SIZE}×{IMG_SIZE} (고정){' '*31}│")
print(f"│  블록 순서       : {str(BLOCK_SIZES)[:42]:<42}│")
print(f"│  기본 sigma      : {str(DEFAULT_BLUR_SIGMAS)[:42]:<42}│")
print(f"│  매칭 지표       : SSIM + PSNR{' '*37}│")
print(f"│  보정 샘플       : 클래스별 최대 {CALIBRATION_MAX_PER_CLASS}장{' '*28}│")
print(f"│  클래스별 목표   : {TARGET_COUNT}장{' '*42}│")
print(f"│  K-Fold         : {N_FOLDS}{' '*45}│")
print(f"│  TTA 횟수       : {TTA_STEPS}회{' '*43}│")
print(f"│  배치 크기      : {BATCH_SIZE}{' '*46}│")
print(f"│  공통 원본 split: {os.path.basename(ORIGINAL_SPLIT_DIR)[:42]:<42}│")
print(f"│  실험 버전      : {EXPERIMENT_VERSION:<42}│")
print("└" + "─" * 62 + "┘")
`;

const blockConfig = `# Cell 4: 실험 설정
# 지표 매칭 블러 버전과 비교하기 위한 블록 기준 실험
BLOCK_SIZES  = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

BASE_TRAIN   = "/workspace/Train"       # 원본 학습 데이터 (절대 수정하지 않음)
WORKSPACE    = "/workspace"
IMG_SIZE     = 512          # 이미지 512×512 고정
BATCH_SIZE   = 4
EPOCHS_S1    = 30
EPOCHS_S2    = 60
NUM_CLASSES  = 3
SEED         = 42
N_FOLDS      = 7
TTA_STEPS    = 20
SWA_START    = 20
TARGET_COUNT = 1000         # 각 클래스별 1000장으로 오버샘플링
TEST_RATIO   = 0.10
AUTOTUNE     = tf.data.AUTOTUNE

# 같은 설정으로 새 결과/분할을 만들기 위한 버전 태그
EXPERIMENT_VERSION = "metric_matched_v1"
ORIGINAL_SPLIT_DIR = os.path.join(
    WORKSPACE, f"AllExp_Split_ORIGINAL_seed{SEED}_{EXPERIMENT_VERSION}"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# 학습 히스토리 저장소 (과적합/과소적합 진단용)
ALL_HISTORIES = {}   # { block_size: { fold: history_dict } }

def set_reproducible(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def list_class_dirs(directory):
    return sorted([d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))])

def list_image_files(directory):
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory)
                   if f.lower().endswith(IMAGE_EXTENSIONS)])

set_reproducible(SEED)

print("┌" + "─" * 62 + "┐")
print("│           ✅ 실험 설정 완료                                  │")
print("├" + "─" * 62 + "┤")
print(f"│  이미지 크기     : {IMG_SIZE}×{IMG_SIZE} (고정){' '*31}│")
print(f"│  블록 크기      : {str(BLOCK_SIZES)[:42]:<42}│")
print(f"│  클래스별 목표   : {TARGET_COUNT}장{' '*42}│")
print(f"│  K-Fold         : {N_FOLDS}{' '*45}│")
print(f"│  TTA 횟수       : {TTA_STEPS}회{' '*43}│")
print(f"│  배치 크기      : {BATCH_SIZE}{' '*46}│")
print(f"│  공통 원본 split: {os.path.basename(ORIGINAL_SPLIT_DIR)[:42]:<42}│")
print(f"│  실험 버전      : {EXPERIMENT_VERSION:<42}│")
print("└" + "─" * 62 + "┘")
`;

const blurTransformAndCalibration = `# Cell 5: 블러 처리 함수 + 블록 지표 매칭 함수
def apply_gaussian_blur(img_array, sigma):
    """
    PIL GaussianBlur로 sigma 강도 블러 적용.
    sigma=0이면 원본 그대로 반환 (copy만 수행, 원본 수정 없음).
    """
    if sigma == 0:
        return img_array.copy()
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.array(blurred, dtype=np.uint8)


def block_scramble_for_calibration(img_array, seed, block_size):
    """블록 노트북과 같은 방식으로 메트릭 계산용 블록 변환을 만든다."""
    h, w, c = img_array.shape
    if block_size >= min(h, w):
        return img_array.copy()

    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    ph, pw   = img_array.shape[:2]
    n_h, n_w = ph // block_size, pw // block_size
    blocks   = [img_array[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                for i in range(n_h) for j in range(n_w)]

    rng      = np.random.default_rng(seed)
    order    = rng.permutation(n_h * n_w)
    result   = np.zeros_like(img_array)

    for new_idx, old_idx in enumerate(order):
        i, j = new_idx // n_w, new_idx % n_w
        result[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[old_idx]

    return result[:h, :w]


def generate_blur_dataset(sigma, seed=SEED):
    """
    sigma=0  → 원본(BASE_TRAIN) 그대로 사용 (파일 복사 없음)
    sigma>0  → Train_BlurSigma{sigma}/ 에 블러 이미지 저장
    BASE_TRAIN 원본 파일은 절대 수정하지 않음.
    """
    if sigma == 0:
        print("  Sigma=0: 원본 사용 (원본 건드리지 않음)")
        return BASE_TRAIN

    sigma_label = str(sigma).replace(".", "p")
    out_dir = os.path.join(WORKSPACE, f"Train_BlurSigma{sigma_label}")
    classes = list_class_dirs(BASE_TRAIN)

    total_src = sum(len(list_image_files(os.path.join(BASE_TRAIN, c))) for c in classes)
    total_ex = (
        sum(len(list_image_files(os.path.join(out_dir, c)))
            for c in classes if os.path.exists(os.path.join(out_dir, c)))
        if os.path.exists(out_dir) else 0
    )

    if total_ex >= total_src:
        print(f"  BlurSigma{sigma}: 이미 존재 ({total_ex}장) → 스킵")
        return out_dir

    print(f"  BlurSigma{sigma}: 블러 처리 중 ({total_src}장)...")
    for cls in classes:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        for f in list_image_files(os.path.join(BASE_TRAIN, cls)):
            dst = os.path.join(out_dir, cls, f)
            if os.path.exists(dst):
                continue
            try:
                img = np.array(
                    Image.open(os.path.join(BASE_TRAIN, cls, f)).convert("RGB"),
                    dtype=np.uint8
                )
                Image.fromarray(apply_gaussian_blur(img, sigma)).save(
                    dst, "JPEG", quality=95
                )
            except Exception:
                pass
    print(f"  BlurSigma{sigma}: 완료! (원본 파일 보존됨)")
    return out_dir


def calibration_image_paths(max_per_class=CALIBRATION_MAX_PER_CLASS, seed=SEED):
    rng = random.Random(seed)
    selected = []
    for cls in list_class_dirs(BASE_TRAIN):
        files = list_image_files(os.path.join(BASE_TRAIN, cls))
        rng.shuffle(files)
        selected.extend(os.path.join(BASE_TRAIN, cls, f) for f in files[:max_per_class])
    return selected


def load_metric_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize((IMG_SIZE, IMG_SIZE), resample)
    return np.array(img, dtype=np.uint8)


def average_similarity_metrics(originals, transform_fn):
    ssim_values, psnr_values = [], []
    for idx, arr in enumerate(originals):
        transformed = transform_fn(arr, idx)
        a = tf.convert_to_tensor(arr[None, ...], dtype=tf.float32)
        b = tf.convert_to_tensor(transformed[None, ...], dtype=tf.float32)
        ssim_values.append(float(tf.image.ssim(a, b, max_val=255.0).numpy()[0]))
        psnr_values.append(float(tf.image.psnr(a, b, max_val=255.0).numpy()[0]))
    return {
        "ssim": float(np.mean(ssim_values)),
        "psnr": float(np.mean(psnr_values)),
    }


def choose_sigma_for_target(target_metrics, candidate_metrics, min_sigma):
    candidates = [m for m in candidate_metrics if m["sigma"] >= min_sigma]
    if not candidates:
        candidates = [candidate_metrics[-1]]

    ssim_pool = [m["ssim"] for m in candidates] + [target_metrics["ssim"]]
    psnr_pool = [m["psnr"] for m in candidates] + [target_metrics["psnr"]]
    ssim_scale = max(ssim_pool) - min(ssim_pool) + 1e-8
    psnr_scale = max(psnr_pool) - min(psnr_pool) + 1e-8

    def distance(m):
        return (
            abs(m["ssim"] - target_metrics["ssim"]) / ssim_scale +
            abs(m["psnr"] - target_metrics["psnr"]) / psnr_scale
        )

    return min(candidates, key=distance)


def calibrate_blur_sigmas_to_blocks(force=False):
    """
    블록 변환별 평균 SSIM/PSNR과 가장 가까운 블러 sigma를 찾는다.
    선택 sigma는 블록 강도가 강해지는 순서대로 단조 증가하도록 제한한다.
    """
    if (not force) and os.path.exists(CALIBRATION_CACHE_PATH):
        with open(CALIBRATION_CACHE_PATH) as f:
            cached = json_module.load(f)
        sigmas = cached.get("blur_sigmas")
        table = cached.get("match_table", [])
        if sigmas and len(sigmas) == len(BLOCK_SIZES):
            print(f"✅ 기존 metric-match sigma 재사용: {sigmas}")
            return sigmas, table

    paths = calibration_image_paths()
    if not paths:
        print("⚠️ 보정 샘플을 찾지 못해 기본 로그 매핑을 사용합니다.")
        return DEFAULT_BLUR_SIGMAS.copy(), []

    print("\\n" + "=" * 72)
    print("📏 블록↔블러 지표 매칭 시작 (평균 SSIM/PSNR 기준)")
    print(f"   샘플: {len(paths)}장 | 후보 sigma: {len(CALIBRATION_SIGMA_CANDIDATES)}개")
    print("=" * 72)

    originals = [load_metric_image(p) for p in paths]
    candidate_metrics = []
    for sigma in CALIBRATION_SIGMA_CANDIDATES:
        metrics = average_similarity_metrics(
            originals,
            lambda arr, idx, s=sigma: apply_gaussian_blur(arr, s)
        )
        metrics["sigma"] = int(sigma)
        candidate_metrics.append(metrics)

    sigmas, table = [], []
    previous_sigma = -1
    for block_size in BLOCK_SIZES:
        if block_size >= IMG_SIZE:
            target = {"ssim": 1.0, "psnr": 99.0}
            selected = {"sigma": 0, "ssim": 1.0, "psnr": 99.0}
        else:
            target = average_similarity_metrics(
                originals,
                lambda arr, idx, bs=block_size: block_scramble_for_calibration(arr, SEED, bs)
            )
            selected = choose_sigma_for_target(
                target,
                candidate_metrics,
                min_sigma=max(previous_sigma + 1, 1)
            )

        sigma = int(selected["sigma"])
        sigmas.append(sigma)
        previous_sigma = sigma
        table.append({
            "block_size": int(block_size),
            "sigma": sigma,
            "block_ssim": float(target["ssim"]),
            "block_psnr": float(target["psnr"]),
            "blur_ssim": float(selected["ssim"]),
            "blur_psnr": float(selected["psnr"]),
        })
        print(
            f"  Block{block_size:>3} → sigma={sigma:<3} | "
            f"block SSIM={target['ssim']:.4f}, PSNR={target['psnr']:.2f} | "
            f"blur SSIM={selected['ssim']:.4f}, PSNR={selected['psnr']:.2f}"
        )

    with open(CALIBRATION_CACHE_PATH, "w") as f:
        json_module.dump({
            "experiment_version": EXPERIMENT_VERSION,
            "block_sizes": BLOCK_SIZES,
            "default_blur_sigmas": DEFAULT_BLUR_SIGMAS,
            "blur_sigmas": sigmas,
            "match_table": table,
            "sample_count": len(paths),
            "candidate_sigmas": CALIBRATION_SIGMA_CANDIDATES,
        }, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print(f"✅ metric-match sigma 확정: {sigmas}")
    return sigmas, table
`;

function patchBlurLoop(text) {
  let patched = text;
  patched = patched.replace(
    `print("\\n" + "=" * 60)\nprint("🔬 블러 처리 강도별 실험 시작 (ResNet50)")`,
    `BLUR_SIGMAS, BLUR_MATCH_TABLE = calibrate_blur_sigmas_to_blocks(force=False)\n\nprint("\\n" + "=" * 60)\nprint("🔬 지표 매칭 블러 강도별 실험 시작 (ResNet50)")`,
  );
  patched = patched.replace(
    `result_path = os.path.join(WORKSPACE, f"all_blur_results_resnet50_{EXPERIMENT_VERSION}.json")`,
    `result_path = os.path.join(WORKSPACE, f"all_blur_metric_matched_results_resnet50_{EXPERIMENT_VERSION}.json")`,
  );
  return patched;
}

const blurResults = `# Cell 13: 결과 표 출력
print("\\n\\n" + "=" * 112)
print("📊 지표 매칭 블록 대응 블러 강도별 실험 결과 (ResNet50)")
print("=" * 112)
print(f"  {'단계':>4}  {'Block대응':>10}  {'Sigma':>8}  {'B-SSIM':>7}  {'σ-SSIM':>7}  {'정확도(전체)':>12}  {'ROC-AUC':>8}  {'양성':>8}  {'악성':>8}  {'정상':>8}  {'평가':>6}")
print("-" * 112)
match_by_sigma = {row["sigma"]: row for row in BLUR_MATCH_TABLE}
for step_idx, (block_equiv, sigma) in enumerate(zip(BLOCK_SIZES, BLUR_SIGMAS), start=1):
    if sigma not in all_results: continue
    r   = all_results[sigma]
    b   = r["per_class_acc"].get("Bengin cases",    0) * 100
    m   = r["per_class_acc"].get("Malignant cases", 0) * 100
    n   = r["per_class_acc"].get("Normal cases",    0) * 100
    acc = r["acc"] * 100
    row = match_by_sigma.get(sigma, {})
    block_lbl = "원본" if block_equiv >= IMG_SIZE else f"{block_equiv}x{block_equiv}"
    grade = "🟢 우수" if acc >= 85 else ("🟡 양호" if acc >= 70 else ("🟠 미흡" if acc >= 55 else "🔴 불량"))
    print(f"  {step_idx:>4}  {block_lbl:>10}  {sigma:>8}  {row.get('block_ssim', 0):>7.4f}  {row.get('blur_ssim', 0):>7.4f}  {acc:>11.2f}%  {r['roc_auc']:>8.4f}"
          f"  {b:>7.1f}%  {m:>7.1f}%  {n:>7.1f}%  {grade}")
print("=" * 112)
`;

function loadNotebook(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function writeNotebook(file, nb) {
  fs.writeFileSync(file, `${JSON.stringify(nb, null, 1)}\n`, "utf8");
  JSON.parse(fs.readFileSync(file, "utf8"));
}

const blur = loadNotebook(inputs.blur);
setCell(blur, 8, blurConfig);
setCell(blur, 10, blurTransformAndCalibration);
setCell(blur, 26, patchBlurLoop(asText(blur.cells[26].source)));
setCell(blur, 28, blurResults);
writeNotebook(outputs.blur, blur);
console.log(`blur: wrote ${outputs.blur}`);

const block = loadNotebook(inputs.block);
setCell(block, 8, blockConfig);
let blockLoop = asText(block.cells[26].source);
blockLoop = blockLoop.replace(
  `result_path = os.path.join(WORKSPACE, f"all_block_results_resnet50_{EXPERIMENT_VERSION}.json")`,
  `result_path = os.path.join(WORKSPACE, f"all_block_metric_matched_results_resnet50_{EXPERIMENT_VERSION}.json")`,
);
setCell(block, 26, blockLoop);
writeNotebook(outputs.block, block);
console.log(`block: wrote ${outputs.block}`);
