import { readFile, writeFile } from "node:fs/promises";

const sourcePath = "./blur_resnet50_rebuilt2_KO.ipynb";
const outputPath = "./block_blur_resnet50_all_combinations_KO.ipynb";

const notebook = JSON.parse(await readFile(sourcePath, "utf8"));

function getSource(index) {
  const source = notebook.cells[index].source;
  return Array.isArray(source) ? source.join("") : source;
}

function setSource(index, source) {
  notebook.cells[index].source = source.match(/.*(?:\n|$)/g).filter(Boolean);
}

function replaceOrThrow(source, before, after, label) {
  if (!source.includes(before)) {
    throw new Error(`Missing replacement target: ${label}`);
  }
  return source.replace(before, after);
}

let header = getSource(0);
header = replaceOrThrow(
  header,
  "🌫️ 지표 매칭 블러 실험 — ResNet50 기반 폐암 분류",
  "🧩🌫️ 블록 암호화 + 블러 순차 조합 실험 — ResNet50 기반 폐암 분류",
  "header title",
);
header = replaceOrThrow(
  header,
  "Metric-Matched Gaussian Blur Experiment · Privacy-Preserving Lung Cancer Classification",
  "Sequential Block Encryption + Gaussian Blur Experiment · Privacy-Preserving Lung Cancer Classification",
  "header subtitle",
);
header = replaceOrThrow(
  header,
  '🌫️ <b>블러 Sigma</b></td><td style="padding: 6px;">블록 SSIM/PSNR 기준 자동 매칭',
  '🧩🌫️ <b>조합 순서</b></td><td style="padding: 6px;">블록 원본 → 블러 원본부터 100개 순차 실행',
  "header combination row",
);
header = replaceOrThrow(
  header,
  "| 5 | Cell 5 | 블러 처리 및 지표 매칭 함수 |",
  "| 5 | Cell 5 | 블록 암호화 + 블러 순차 처리 함수 |",
  "toc processing",
);
header = replaceOrThrow(
  header,
  "| 12 | Cell 11 | 단일 Sigma 실험 함수 |",
  "| 12 | Cell 11 | 단일 블록 + 블러 조합 실험 함수 |",
  "toc runner",
);
setSource(0, header);

setSource(8, `# Cell 4: 실험 설정
BLOCK_SIZES  = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
BLUR_SIGMAS  = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
COMBINATIONS = [(block_size, sigma)
                for block_size in BLOCK_SIZES
                for sigma in BLUR_SIGMAS]

BASE_TRAIN   = "/workspace/Train"
WORKSPACE    = "/workspace"
IMG_SIZE     = 512
BATCH_SIZE   = 8
EPOCHS_S1    = 80
EPOCHS_S2    = 200
NUM_CLASSES  = 3
SEED         = 42
N_FOLDS      = 7
TTA_STEPS    = 20
SWA_START    = 40
TARGET_COUNT = 1000
TEST_RATIO   = 0.10
AUTOTUNE     = tf.data.AUTOTUNE

EXPERIMENT_VERSION = "block_blur_all_combinations_v1"
ORIGINAL_SPLIT_DIR = os.path.join(
    WORKSPACE, f"AllExp_Split_ORIGINAL_seed{SEED}_{EXPERIMENT_VERSION}"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

ALL_HISTORIES = {}   # { condition_key: { fold: history_dict } }

def condition_key(block_size, sigma):
    sigma_label = str(sigma).replace(".", "p")
    return f"B{block_size}_S{sigma_label}"

def condition_label(block_size, sigma):
    block_label = "블록 원본" if block_size >= IMG_SIZE else f"Block{block_size}"
    blur_label  = "블러 원본" if sigma == 0 else f"BlurSigma σ={sigma}"
    return f"{block_label} + {blur_label}"

def condition_short_label(block_size, sigma):
    block_label = "원본" if block_size >= IMG_SIZE else f"B{block_size}"
    blur_label  = "원본" if sigma == 0 else f"σ{sigma}"
    return f"{block_label}·{blur_label}"

CONDITION_META = {
    condition_key(block_size, sigma): (block_size, sigma)
    for block_size, sigma in COMBINATIONS
}

def condition_label_from_key(key):
    block_size, sigma = CONDITION_META[key]
    return condition_label(block_size, sigma)

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
print("│          ✅ 실험 설정 완료                                  │")
print("├" + "─" * 62 + "┤")
print(f"│ 이미지 크기     : {IMG_SIZE}×{IMG_SIZE} (고정){' '*31}│")
print(f"│ 블록 크기       : {str(BLOCK_SIZES)[:42]:<42}│")
print(f"│ 블러 Sigma      : {str(BLUR_SIGMAS)[:42]:<42}│")
print(f"│ 전체 조합       : {len(COMBINATIONS):<42}│")
print(f"│ 클래스별 목표   : {TARGET_COUNT}장{' '*42}│")
print(f"│ K-Fold         : {N_FOLDS}{' '*45}│")
print(f"│ TTA 횟수       : {TTA_STEPS}회{' '*43}│")
print(f"│ 배치 크기      : {BATCH_SIZE}{' '*46}│")
print(f"│ 공통 원본 split: {os.path.basename(ORIGINAL_SPLIT_DIR)[:42]:<42}│")
print(f"│ 실험 버전      : {EXPERIMENT_VERSION:<42}│")
print("└" + "─" * 62 + "┘")
`);

setSource(9, "## 5️⃣ 블록 암호화 + 블러 순차 처리 함수\n");
setSource(10, `# Cell 5: 블록 암호화 + 블러 순차 처리 함수
def block_scramble(img_array, seed, block_size):
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

    rng    = np.random.default_rng(seed)
    order  = rng.permutation(n_h * n_w)
    result = np.zeros_like(img_array)

    for new_idx, old_idx in enumerate(order):
        i, j = new_idx // n_w, new_idx % n_w
        result[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[old_idx]

    return result[:h, :w]


def generate_block_dataset(block_size, seed=SEED):
    """
    block_size=512 → 원본(BASE_TRAIN) 그대로 사용 (파일 복사 없음)
    block_size<512 → Train_Block{block_size}/ 에 암호화 이미지 저장
    BASE_TRAIN 원본 파일은 절대 수정하지 않음.
    """
    if block_size >= IMG_SIZE:
        print(f"  Block{block_size}: 원본 사용 (원본 건드리지 않음)")
        return BASE_TRAIN

    out_dir = os.path.join(WORKSPACE, f"Train_Block{block_size}")
    classes = list_class_dirs(BASE_TRAIN)

    total_src = sum(len(list_image_files(os.path.join(BASE_TRAIN, c))) for c in classes)
    total_ex = (
        sum(len(list_image_files(os.path.join(out_dir, c)))
            for c in classes if os.path.exists(os.path.join(out_dir, c)))
        if os.path.exists(out_dir) else 0
    )

    if total_ex >= total_src:
        print(f"  Block{block_size}: 이미 존재 ({total_ex}장) → 스킵")
        return out_dir

    print(f"  Block{block_size}: 블록 암호화 처리 중 ({total_src}장)...")
    for cls in classes:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        for f in list_image_files(os.path.join(BASE_TRAIN, cls)):
            dst = os.path.join(out_dir, cls, f)
            if os.path.exists(dst):
                continue
            try:
                img = Image.open(os.path.join(BASE_TRAIN, cls, f)).convert("RGB")
                img_array = np.array(img, dtype=np.uint8)
                Image.fromarray(block_scramble(img_array, seed, block_size)).save(
                    dst, "JPEG", quality=95
                )
            except Exception:
                pass
    print(f"  Block{block_size}: 완료! (원본 파일 보존됨)")
    return out_dir


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


def generate_blur_dataset(block_size, sigma, source_dir, seed=SEED):
    """
    sigma=0  → 직전 블록 처리 결과 그대로 사용 (파일 복사 없음)
    sigma>0  → Train_Block{block_size}_BlurSigma{sigma}/ 에 블러 이미지 저장
    블록 처리 결과에 블러를 순차 적용하며 BASE_TRAIN 원본은 절대 수정하지 않음.
    """
    if sigma == 0:
        print(f"  Block{block_size} + Sigma=0: 직전 블록 처리 결과 사용")
        return source_dir

    sigma_label = str(sigma).replace(".", "p")
    out_dir     = os.path.join(WORKSPACE, f"Train_Block{block_size}_BlurSigma{sigma_label}")
    classes     = list_class_dirs(source_dir)
    total_src   = sum(len(list_image_files(os.path.join(source_dir, c))) for c in classes)
    total_ex    = (
        sum(len(list_image_files(os.path.join(out_dir, c)))
            for c in classes if os.path.exists(os.path.join(out_dir, c)))
        if os.path.exists(out_dir) else 0
    )
    if total_ex >= total_src:
        print(f"  Block{block_size} + BlurSigma{sigma}: 이미 존재 ({total_ex}장) → 스킵")
        return out_dir

    print(f"  Block{block_size} + BlurSigma{sigma}: 블러 처리 중 ({total_src}장)...")
    for cls in classes:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        for f in list_image_files(os.path.join(source_dir, cls)):
            dst = os.path.join(out_dir, cls, f)
            if os.path.exists(dst):
                continue
            try:
                img = np.array(
                    Image.open(os.path.join(source_dir, cls, f)).convert("RGB"),
                    dtype=np.uint8
                )
                Image.fromarray(apply_gaussian_blur(img, sigma)).save(
                    dst, "JPEG", quality=95
                )
            except Exception:
                pass
    print(f"  Block{block_size} + BlurSigma{sigma}: 완료! (원본 파일 보존됨)")
    return out_dir


def generate_combined_dataset(block_size, sigma, block_dir=None):
    if block_dir is None:
        block_dir = generate_block_dataset(block_size)
    return generate_blur_dataset(block_size, sigma, block_dir)


print("✅ block_scramble / apply_gaussian_blur / generate_combined_dataset 정의 완료")
print(f"   블록: {BLOCK_SIZES}")
print(f"   블러 Sigma: {BLUR_SIGMAS}")
print(f"   실행 조합: {len(COMBINATIONS)}개")
`);

let monitor = getSource(22);
monitor = replaceOrThrow(
  monitor,
  "def print_fold_result(sigma, fold_idx, combined_history, stage=\"\"):",
  "def print_fold_result(condition, fold_idx, combined_history, stage=\"\"):",
  "monitor result signature",
);
monitor = replaceOrThrow(
  monitor,
  "    label = \"원본(σ=0)\" if sigma == 0 else f\"BlurSigma σ={sigma}\"",
  "    label = condition_label_from_key(condition)",
  "monitor result label",
);
monitor = replaceOrThrow(
  monitor,
  "    def __init__(self, sigma, fold_idx, every_n_epochs=5, overfit_gap=0.20):",
  "    def __init__(self, condition, fold_idx, every_n_epochs=5, overfit_gap=0.20):",
  "monitor callback signature",
);
monitor = replaceOrThrow(
  monitor,
  "        self.sigma = sigma",
  "        self.condition = condition",
  "monitor callback field",
);
monitor = replaceOrThrow(
  monitor,
  '                    "sigma": self.sigma,',
  '                    "condition": self.condition,',
  "monitor gpu log",
);
setSource(22, monitor);

let runner = getSource(24);
runner = replaceOrThrow(
  runner,
  "# Cell 11: 단일 블러 Sigma 실험 함수 (누수 없는 올바른 순서)\ndef run_experiment(sigma, blur_dir):\n    exp_idx  = BLUR_SIGMAS.index(sigma) if sigma in BLUR_SIGMAS else 0\n    exp_seed = SEED + exp_idx * 1000\n    set_reproducible(exp_seed)\n\n    label = \"원본(σ=0)\" if sigma == 0 else f\"BlurSigma σ={sigma}\"\n    print(f\"\\n{'='*60}\\n🌫️ {label}  (seed={exp_seed})\\n{'='*60}\")\n\n    split_dir = (ORIGINAL_SPLIT_DIR if sigma == 0 else\n                 os.path.join(WORKSPACE,\n                              f\"AllExp_Split_BlurSigma{sigma}_{EXPERIMENT_VERSION}_seed{SEED}\"))",
  "# Cell 11: 단일 블록 + 블러 조합 실험 함수 (누수 없는 올바른 순서)\ndef run_experiment(block_size, sigma, combined_dir):\n    condition = condition_key(block_size, sigma)\n    exp_idx   = COMBINATIONS.index((block_size, sigma)) if (block_size, sigma) in COMBINATIONS else 0\n    exp_seed  = SEED + exp_idx * 1000\n    set_reproducible(exp_seed)\n\n    label = condition_label(block_size, sigma)\n    print(f\"\\n{'='*60}\\n🧩🌫️ {label}  (seed={exp_seed})\\n{'='*60}\")\n\n    sigma_label = str(sigma).replace(\".\", \"p\")\n    split_dir = (ORIGINAL_SPLIT_DIR if block_size >= IMG_SIZE and sigma == 0 else\n                 os.path.join(WORKSPACE,\n                              f\"AllExp_Split_Block{block_size}_BlurSigma{sigma_label}_{EXPERIMENT_VERSION}_seed{SEED}\"))",
  "runner prelude",
);
runner = replaceOrThrow(
  runner,
  "    prepare_split(blur_dir, split_dir, TEST_RATIO, seed=SEED)",
  "    prepare_split(combined_dir, split_dir, TEST_RATIO, seed=SEED)",
  "runner input directory",
);
runner = replaceOrThrow(
  runner,
  "    if sigma not in ALL_HISTORIES:\n        ALL_HISTORIES[sigma] = {}",
  "    if condition not in ALL_HISTORIES:\n        ALL_HISTORIES[condition] = {}",
  "runner history init",
);
runner = replaceOrThrow(
  runner,
  '        prefix = f"blur_sigma{sigma}_resnet50_f{fold}"',
  '        prefix = f"block{block_size}_blur_sigma{sigma_label}_resnet50_f{fold}"',
  "runner model prefix",
);
runner = replaceOrThrow(
  runner,
  "        monitor_cb = AdvancedMonitorCallback(sigma, fold, every_n_epochs=5)",
  "        monitor_cb = AdvancedMonitorCallback(condition, fold, every_n_epochs=5)",
  "runner monitor condition",
);
runner = replaceOrThrow(
  runner,
  "        ALL_HISTORIES[sigma][fold] = combined\n        print_fold_result(sigma, fold, combined)",
  "        ALL_HISTORIES[condition][fold] = combined\n        print_fold_result(condition, fold, combined)",
  "runner history save",
);
runner = replaceOrThrow(
  runner,
  "    print(f\"\\n  ✅ Sigma={sigma}: {test_acc*100:.2f}% | ROC-AUC={roc_auc:.4f}\")\n\n    return {\n        \"sigma\": sigma,",
  "    print(f\"\\n  ✅ {label}: {test_acc*100:.2f}% | ROC-AUC={roc_auc:.4f}\")\n\n    return {\n        \"condition\": condition,\n        \"block_size\": block_size,\n        \"sigma\": sigma,",
  "runner return payload",
);
runner = replaceOrThrow(
  runner,
  "print(\"✅ Cell 11: run_experiment() 정의 완료\")",
  "print(\"✅ Cell 11: run_experiment() 블록 + 블러 조합 버전 정의 완료\")",
  "runner completion",
);
setSource(23, "## 1️⃣1️⃣ 단일 블록 + 블러 조합 실험 함수\n");
setSource(24, runner);

setSource(25, "## 1️⃣2️⃣ 전체 조합 순차 실험 루프\n");
setSource(26, `# Cell 12: 전체 블록 + 블러 조합 실험 루프 + 조합별 즉시 진단
print("\\n" + "=" * 60)
print("🔬 블록 암호화 + 블러 전체 조합 실험 시작 (ResNet50)")
print(f"   블록: {BLOCK_SIZES}")
print(f"   블러 Sigma: {BLUR_SIGMAS}")
print(f"   순차 조합: {len(COMBINATIONS)}개")
print(f"   순서: 블록 조건 바깥 루프 → 블러 조건 안쪽 루프")
print(f"   K={N_FOLDS} | TTA={TTA_STEPS} | BATCH={BATCH_SIZE} | IMG={IMG_SIZE}")
print(f"   클래스별 오버샘플링: {TARGET_COUNT}장")
print(f"   실험 버전: {EXPERIMENT_VERSION}")
print("=" * 60)

gpu_status("실험 시작 전")

all_results = {}
result_path = os.path.join(
    WORKSPACE, f"all_block_blur_results_resnet50_{EXPERIMENT_VERSION}.json"
)

if os.path.exists(result_path):
    with open(result_path) as f:
        saved = json_module.load(f)
    for k, v in saved.items():
        all_results[k] = v
    print(f"\\n✅ 이전 결과: {len(all_results)}개 조합")

condition_diagnosis = {}

def _diagnose_condition(condition):
    if condition not in ALL_HISTORIES or not ALL_HISTORIES[condition]:
        return

    S_EMOJI = {
        "good": "✅ 정상",
        "overfit": "🔴 과적합",
        "underfit": "🟡 과소적합",
        "unknown": "❓ 미상",
    }

    fold_statuses = []
    lbl = condition_label_from_key(condition)

    print(f"\\n  {'─'*54}")
    print(f"  📋 [{lbl}] 폴드별 과적합 진단")
    print(f"  {'─'*54}")

    for fi, hist in sorted(ALL_HISTORIES[condition].items()):
        st, d = diagnose_fit(
            hist.get("loss", []),
            hist.get("val_loss", []),
            hist.get("accuracy", []),
            hist.get("val_accuracy", []),
        )
        fold_statuses.append(st)

        ta = d.get("train_acc_final", 0) * 100
        va = d.get("val_acc_final", 0) * 100
        ag = d.get("acc_gap", 0) * 100
        lg = d.get("loss_gap", 0)
        ep = len(hist.get("loss", []))

        print(f"  Fold {fi+1}: {S_EMOJI[st]:<12} "
              f"train={ta:.1f}% val={va:.1f}% "
              f"acc_gap={ag:+.1f}% loss_gap={lg:+.4f} ep={ep}")

    cnt = {s: fold_statuses.count(s) for s in ["good", "overfit", "underfit"]}
    dominant = max(cnt, key=cnt.get)

    if dominant == "overfit":
        comment = "→ Dropout 강화 또는 EarlyStopping patience 축소 권장"
    elif dominant == "underfit":
        comment = "→ lr 상향 또는 에포크 추가 권장"
    else:
        comment = "→ 정상 수렴"

    print(f"  {'─'*54}")
    print(f"  종합: {S_EMOJI[dominant]}  "
          f"(정상 {cnt['good']} / 과적합 {cnt['overfit']} / 과소적합 {cnt['underfit']}폴드)")
    print(f"  {comment}")
    print(f"  {'─'*54}")

    condition_diagnosis[condition] = {
        "status": dominant,
        "fold_statuses": fold_statuses,
        "counts": cnt,
    }

for block_idx, block_size in enumerate(BLOCK_SIZES, start=1):
    print(f"\\n{'#'*60}")
    print(f"🧩 블록 단계 {block_idx}/{len(BLOCK_SIZES)} | Block{block_size} 준비")
    print(f"{'#'*60}")
    block_dir = generate_block_dataset(block_size)

    for sigma_idx, sigma in enumerate(BLUR_SIGMAS, start=1):
        step_idx  = (block_idx - 1) * len(BLUR_SIGMAS) + sigma_idx
        condition = condition_key(block_size, sigma)
        lbl       = condition_label(block_size, sigma)

        if condition in all_results:
            print(f"\\n⏭️  Step {step_idx}/{len(COMBINATIONS)} | {lbl}: "
                  f"완료({all_results[condition]['acc']*100:.2f}%) → 스킵")
            if condition in ALL_HISTORIES and ALL_HISTORIES[condition]:
                _diagnose_condition(condition)
            continue

        print(f"\\n📦 Step {step_idx}/{len(COMBINATIONS)} | {lbl} 데이터셋 준비...")

        combined_dir = generate_combined_dataset(block_size, sigma, block_dir)
        result       = run_experiment(block_size, sigma, combined_dir)

        all_results[condition] = result

        with open(result_path, "w") as f:
            json_module.dump(all_results, f, indent=2, ensure_ascii=False)

        print("  💾 중간 저장 완료")
        _diagnose_condition(condition)
        gpu_status(f"{condition} 완료 후")

print("\\n" + "=" * 60)
print("📊 전체 블록 + 블러 조합 실험 최종 요약")
print("=" * 60)

if condition_diagnosis:
    print(f"  {'조합':>24}  {'정확도':>8}  {'ROC-AUC':>8}  {'모델상태':>12}  {'폴드분포'}")
    print(f"  {'-'*78}")
    for block_size, sigma in COMBINATIONS:
        condition = condition_key(block_size, sigma)
        if condition not in all_results:
            continue

        r = all_results[condition]
        d = condition_diagnosis.get(condition, {})
        status = d.get("status", "unknown")
        counts = d.get("counts", {})

        lbl = condition_short_label(block_size, sigma)
        acc = r["acc"] * 100
        auc = r["roc_auc"]

        print(f"  {lbl:>24}  {acc:>7.2f}%  {auc:>8.4f}  "
              f"{status:>12}  "
              f"정상 {counts.get('good',0)} / 과적합 {counts.get('overfit',0)} / 과소적합 {counts.get('underfit',0)}")

print("=" * 60)
`);

setSource(27, "## 1️⃣3️⃣ 전체 블록 + 블러 조합 결과 표 출력\n");
setSource(28, `# Cell 13: 결과 표 출력
print("\\n\\n" + "=" * 108)
print("📊 블록 암호화 + 블러 전체 조합 실험 결과 (ResNet50)")
print("=" * 108)
print(f"  {'블록':>8}  {'블러':>10}  {'정확도(전체)':>12}  {'ROC-AUC':>8}  {'양성(Benign)':>13}  {'악성(Malig)':>12}  {'정상(Normal)':>13}  {'평가':>6}")
print("-" * 108)

for block_size, sigma in COMBINATIONS:
    condition = condition_key(block_size, sigma)
    if condition not in all_results:
        continue

    r   = all_results[condition]
    b   = r["per_class_acc"].get("Bengin cases",    0) * 100
    m   = r["per_class_acc"].get("Malignant cases", 0) * 100
    n   = r["per_class_acc"].get("Normal cases",    0) * 100
    acc = r["acc"] * 100

    block_lbl = "원본" if block_size >= IMG_SIZE else str(block_size)
    blur_lbl  = "원본" if sigma == 0 else f"σ={sigma}"
    grade = ("🟢 우수" if acc >= 85 else
             ("🟡 양호" if acc >= 70 else
              ("🟠 미흡" if acc >= 55 else "🔴 불량")))

    print(f"  {block_lbl:>8}  {blur_lbl:>10}  {acc:>11.2f}%  {r['roc_auc']:>8.4f}"
          f"  {b:>12.1f}%  {m:>11.1f}%  {n:>12.1f}%  {grade}")

print("=" * 108)
`);

setSource(29, `## 🆕 13b. 전체 조합 과적합 / 과소적합 진단 & GPU 타임라인

모든 블록 크기와 블러 Sigma 조합, 폴드에 대해 최종 수렴 상태를 요약합니다.

GPU 로그가 있으면 VRAM·활용률·온도 요약도 함께 출력합니다.
`);
setSource(30, `# Cell 13b: 전체 블록 + 블러 조합 과적합 / 과소적합 진단 요약
print("=" * 75)
print("🩺 전체 과적합 / 과소적합 진단 요약 [ResNet50 블록 + 블러]")
print("=" * 75)

STATUS_EMOJI = {
    "good": "✅ 정상",
    "overfit": "🔴 과적합",
    "underfit": "🟡 과소적합",
    "unknown": "❓ 미상",
}

for block_size, sigma in COMBINATIONS:
    condition = condition_key(block_size, sigma)
    if condition not in ALL_HISTORIES:
        continue

    lbl = condition_label(block_size, sigma)
    print(f"\\n  [{lbl}]")

    statuses = []
    for fold_idx, hist in sorted(ALL_HISTORIES[condition].items()):
        status, detail = diagnose_fit(
            hist.get("loss", []),
            hist.get("val_loss", []),
            hist.get("accuracy", []),
            hist.get("val_accuracy", []),
        )
        statuses.append(status)

        print(f"    Fold {fold_idx+1}: {STATUS_EMOJI[status]:<12}  "
              f"train={detail.get('train_acc_final',0)*100:.1f}%  "
              f"val={detail.get('val_acc_final',0)*100:.1f}%  "
              f"acc_gap={detail.get('acc_gap',0)*100:+.1f}%  "
              f"loss_gap={detail.get('loss_gap',0):+.4f}")

    cnt = {s: statuses.count(s) for s in ["good", "overfit", "underfit"]}
    dominant = max(cnt, key=cnt.get) if statuses else "unknown"

    print(f"    → 종합: {STATUS_EMOJI[dominant]}  "
          f"(정상 {cnt.get('good',0)}폴드 / 과적합 {cnt.get('overfit',0)}폴드 / 과소적합 {cnt.get('underfit',0)}폴드)")

print("=" * 75)

if len(GPU_LOG) > 0:
    used_vals = [g["used"] for g in GPU_LOG]
    util_vals = [g["util"] for g in GPU_LOG]
    temp_vals = [g["temp"] for g in GPU_LOG]

    print("\\n" + "=" * 55)
    print("🖥️  GPU 사용 타임라인 요약")
    print("=" * 55)
    print(f"  기록 수    : {len(GPU_LOG)}개")
    print(f"  VRAM 평균  : {np.mean(used_vals):.0f} MiB | 최대: {np.max(used_vals):.0f} MiB")
    print(f"  활용률 평균: {np.mean(util_vals):.1f}% | 최대: {np.max(util_vals):.1f}%")
    print(f"  온도 평균  : {np.mean(temp_vals):.1f}°C | 최대: {np.max(temp_vals):.1f}°C")
    print("=" * 55)
else:
    print("\\nℹ️  GPU 로그 없음 (GPU 없이 실행되었거나 nvidia-smi 미설치)")
`);

setSource(31, `## 1️⃣4️⃣ 훈련 후 평가 그래프 10개

실험 완료 후 아래 셀을 실행하면 블록 크기와 블러 Sigma 전체 조합의 정확도, 클래스별 성능, ROC-AUC, 혼동 행렬, 학습 곡선, 폴드 히트맵, 과적합 판정, 레이더 차트, GPU 타임라인, 종합 대시보드까지 총 10개 그래프를 생성합니다.

모든 그래프는 기존 디자인과 동일하게 한글 폰트가 적용된 상태로 저장됩니다.
`);

let graph = getSource(32);
graph = replaceOrThrow(
  graph,
  `_SIGMAS = list(BLUR_SIGMAS) if "BLUR_SIGMAS" in globals() else [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
_BLOCK_EQUIV = list(BLOCK_SIZES) if "BLOCK_SIZES" in globals() else [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
_CLASS_NAMES = ["Bengin cases", "Malignant cases", "Normal cases"]
_CLASS_KO    = ["양성(Benign)", "악성(Malignant)", "정상(Normal)"]

def _label_for(idx, sigma):
    if sigma == 0:
        return "원본"
    if idx < len(_BLOCK_EQUIV):
        return f"B{_BLOCK_EQUIV[idx]}·σ{sigma}"
    return f"σ{sigma}"`,
  `_GRAPH_CONDITIONS = list(COMBINATIONS) if "COMBINATIONS" in globals() else [
    (block_size, sigma)
    for block_size in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    for sigma in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
]
_SIGMAS = list(range(len(_GRAPH_CONDITIONS)))
_CLASS_NAMES = ["Bengin cases", "Malignant cases", "Normal cases"]
_CLASS_KO    = ["양성(Benign)", "악성(Malignant)", "정상(Normal)"]

def _label_for(idx, sigma):
    if idx < len(_GRAPH_CONDITIONS):
        block_size, blur_sigma = _GRAPH_CONDITIONS[idx]
        return condition_short_label(block_size, blur_sigma)
    return str(sigma)`,
  "graph prelude",
);
graph = replaceOrThrow(
  graph,
  '            "block_size_equiv": _BLOCK_EQUIV[i] if i < len(_BLOCK_EQUIV) else None,',
  '            "condition": _GRAPH_CONDITIONS[i] if i < len(_GRAPH_CONDITIONS) else None,',
  "graph dummy condition",
);
graph = replaceOrThrow(
  graph,
  `_results   = all_results if ("all_results" in globals() and len(all_results) > 0) else _dummy_results()
_histories = ALL_HISTORIES if ("ALL_HISTORIES" in globals() and len(ALL_HISTORIES) > 0) else _dummy_histories()`,
  `_GRAPH_RESULTS = {
    idx: all_results[condition_key(block_size, sigma)]
    for idx, (block_size, sigma) in enumerate(_GRAPH_CONDITIONS)
    if "all_results" in globals() and condition_key(block_size, sigma) in all_results
}
_GRAPH_HISTORIES = {
    idx: ALL_HISTORIES[condition_key(block_size, sigma)]
    for idx, (block_size, sigma) in enumerate(_GRAPH_CONDITIONS)
    if "ALL_HISTORIES" in globals() and condition_key(block_size, sigma) in ALL_HISTORIES
}
_results   = _GRAPH_RESULTS if len(_GRAPH_RESULTS) > 0 else _dummy_results()
_histories = _GRAPH_HISTORIES if len(_GRAPH_HISTORIES) > 0 else _dummy_histories()`,
  "graph runtime maps",
);
graph = replaceOrThrow(
  graph,
  'SAVE_DIR = os.path.join(WORKSPACE, "eval_graphs_blur_metric_matched")',
  'SAVE_DIR = os.path.join(WORKSPACE, "eval_graphs_block_blur_all_combinations")',
  "graph save directory",
);
graph = replaceOrThrow(
  graph,
  '    lbl_str = "원본(σ=0)" if sigma == 0 else f"Blur σ={sigma}"',
  "    lbl_str = _label_for(sigma, sigma)",
  "graph confusion label",
);
graph = replaceOrThrow(
  graph,
  '    lbl_str = "원본(σ=0)" if sigma == 0 else f"σ={sigma}"',
  "    lbl_str = _label_for(sigma, sigma)",
  "graph radar label",
);
graph = replaceOrThrow(
  graph,
  '    lbl = "원본" if sigma == 0 else f"σ={sigma}"',
  "    lbl = _label_for(sigma, sigma)",
  "graph dashboard table label",
);
graph = graph
  .replaceAll("지표 매칭 블러 Sigma", "블록 암호화 + 블러 조합")
  .replaceAll("블록 대응 · 블러 Sigma", "블록 · 블러 조합")
  .replaceAll("블록 대응 · Sigma", "블록 · 블러 조합")
  .replaceAll("Sigma × 폴드", "조합 × 폴드")
  .replaceAll('ax00.set_xlabel("Sigma")', 'ax00.set_xlabel("조합")')
  .replaceAll('ax01.set_xlabel("Sigma")', 'ax01.set_xlabel("조합")')
  .replaceAll('ax11.set_xlabel("Sigma")', 'ax11.set_xlabel("조합")')
  .replaceAll('col_labels = ["Sigma",', 'col_labels = ["조합",')
  .replaceAll("지표 매칭 블러 ResNet50", "블록 암호화 + 블러 ResNet50")
  .replaceAll("지표 매칭 블러 종합 성능 순위", "블록 + 블러 조합 종합 성능 순위")
  .replaceAll("지표 매칭 블러 성능 비교", "블록 + 블러 조합 성능 비교")
  .replaceAll("지표 매칭 Sigma별", "블록 + 블러 조합별")
  .replaceAll("Sigma별", "조합별");
setSource(32, graph);

notebook.metadata = notebook.metadata || {};
notebook.metadata.block_blur_merge = {
  source_notebooks: [
    "blur_resnet50_rebuilt2_KO.ipynb",
    "block_encryption_resnet50_rebuilt_KO.ipynb",
  ],
  order: "BLOCK_SIZES outer loop, BLUR_SIGMAS inner loop",
  combinations: 100,
};

await writeFile(outputPath, `${JSON.stringify(notebook, null, 1)}\n`, "utf8");
console.log(outputPath);
