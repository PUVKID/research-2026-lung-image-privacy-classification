const fs = require("fs");
const path = require("path");

const workspace = ".";
const notebooks = {
  blur: {
    inPath: "./blur_resnet50_clean.ipynb",
    outPath: path.join(workspace, "blur_resnet50_same_settings_seq.ipynb"),
  },
  block: {
    inPath: "./block_encryption_resnet50_clean.ipynb",
    outPath: path.join(workspace, "block_encryption_resnet50_same_settings_seq.ipynb"),
  },
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

function replaceRequired(text, from, to, label) {
  if (!text.includes(from)) {
    throw new Error(`Missing snippet for ${label}`);
  }
  return text.replace(from, to);
}

function replaceFirstRequired(text, from, to, label) {
  const idx = text.indexOf(from);
  if (idx < 0) throw new Error(`Missing first snippet for ${label}`);
  return text.slice(0, idx) + to + text.slice(idx + from.length);
}

const blurConfig = `# Cell 4: 실험 설정
# 블록 크기 순서와 블러 sigma를 같은 단계로 고정
BLOCK_SIZES  = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
BLUR_SIGMAS  = [0 if bs >= 512 else 512 // (2 * bs) for bs in BLOCK_SIZES]

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
EXPERIMENT_VERSION = "same_settings_v1"
ORIGINAL_SPLIT_DIR = os.path.join(
    WORKSPACE, f"AllExp_Split_ORIGINAL_seed{SEED}_{EXPERIMENT_VERSION}"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

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
print(f"│  블러 sigma      : {str(BLUR_SIGMAS)[:42]:<42}│")
print(f"│  클래스별 목표   : {TARGET_COUNT}장{' '*42}│")
print(f"│  K-Fold         : {N_FOLDS}{' '*45}│")
print(f"│  TTA 횟수       : {TTA_STEPS}회{' '*43}│")
print(f"│  배치 크기      : {BATCH_SIZE}{' '*46}│")
print(f"│  공통 원본 split: {os.path.basename(ORIGINAL_SPLIT_DIR)[:42]:<42}│")
print(f"│  실험 버전      : {EXPERIMENT_VERSION:<42}│")
print("└" + "─" * 62 + "┘")
print("블록 대응 순서:", list(zip(BLOCK_SIZES, BLUR_SIGMAS)))
`;

const blockConfig = `# Cell 4: 실험 설정
# 블러 노트북과 동일한 순서/시드/분할 규칙 사용
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
EXPERIMENT_VERSION = "same_settings_v1"
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

const blurGenerate = `# Cell 5: 블러 처리 함수
def apply_gaussian_blur(img_array, sigma):
    """
    PIL GaussianBlur로 sigma 강도 블러 적용.
    sigma=0이면 원본 그대로 반환 (copy만 수행, 원본 수정 없음).
    """
    if sigma == 0:
        return img_array.copy()
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred, dtype=np.uint8)


def generate_blur_dataset(sigma, seed=SEED):
    """
    sigma=0  → 원본(BASE_TRAIN) 그대로 사용 (파일 복사 없음)
    sigma>0  → Train_BlurSigma{sigma}/ 에 블러 이미지 저장
    BASE_TRAIN 원본 파일은 절대 수정하지 않음.
    """
    if sigma == 0:
        print("  Sigma=0: 원본 사용 (원본 건드리지 않음)")
        return BASE_TRAIN

    out_dir = os.path.join(WORKSPACE, f"Train_BlurSigma{sigma}")
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
`;

const blockGenerate = `# Cell 5: 블록 암호화 함수
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

    rng      = np.random.default_rng(seed)
    order    = rng.permutation(n_h * n_w)
    result   = np.zeros_like(img_array)

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
`;

const oversampleCell = `# Cell 6: 오버샘플링 (각 클래스 1000장, 원본 파일 보존)
def oversample_dir(base_dir, target_count=1000, seed=SEED):
    """원본(base_dir) 파일을 건드리지 않고, 버전이 붙은 oversampled 폴더에 복사+증강."""
    out_dir = base_dir.rstrip("/") + f"_oversampled_{EXPERIMENT_VERSION}_seed{seed}"
    classes = list_class_dirs(base_dir)

    if os.path.exists(out_dir):
        all_done = all(
            len(list_image_files(os.path.join(out_dir, c))) >= target_count
            for c in classes if os.path.exists(os.path.join(out_dir, c))
        ) and all(os.path.exists(os.path.join(out_dir, c)) for c in classes)
        if all_done:
            total = sum(len(list_image_files(os.path.join(out_dir, c))) for c in classes)
            print(f"    오버샘플링 이미 완료 ({total}장) → 재사용")
            return out_dir

    rng = random.Random(seed)

    for cls in classes:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        src_files = [f for f in list_image_files(os.path.join(base_dir, cls))
                     if not f.startswith("aug_")]
        for f in src_files:
            dst = os.path.join(out_dir, cls, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(base_dir, cls, f), dst)

    for cls in classes:
        cls_out = os.path.join(out_dir, cls)
        files = list_image_files(cls_out)
        src_files = [f for f in list_image_files(os.path.join(base_dir, cls))
                     if not f.startswith("aug_")]
        needed = target_count - len(files)
        if needed <= 0:
            print(f"    {cls}: {len(files)}장 (목표 {target_count}장 달성)")
            continue
        if not src_files:
            print(f"    {cls}: 원본 이미지가 없어 증강을 건너뜀")
            continue
        print(f"    {cls}: {len(files)}장 → {target_count}장 ({needed}장 증강 생성)")
        generated = 0
        while generated < needed:
            src_f = rng.choice(src_files)
            try:
                img = Image.open(os.path.join(base_dir, cls, src_f)).convert("RGB")
                img = img.rotate(rng.uniform(-30, 30), expand=False)
                if rng.random() > 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if rng.random() > 0.5: img = img.transpose(Image.FLIP_TOP_BOTTOM)
                from PIL import ImageEnhance
                img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
                img.save(os.path.join(cls_out, f"aug_{generated:05d}_{src_f}"),
                         "JPEG", quality=95)
                generated += 1
            except Exception:
                continue

    total = sum(len(list_image_files(os.path.join(out_dir, c))) for c in classes)
    print(f"    ✅ 오버샘플링 완료: {total}장 → {out_dir}")
    print(f"    ✅ 원본 파일({base_dir}) 보존 확인")
    return out_dir
`;

const splitCell = `# Cell 7: Train/Test 분리
def prepare_split(base_dir, split_dir, test_ratio=0.10, seed=SEED):
    test_dir  = os.path.join(split_dir, "test")
    kfold_dir = os.path.join(split_dir, "kfold_data")
    if os.path.exists(test_dir) and os.path.exists(kfold_dir):
        print(f"    Split 폴더 이미 존재 → 재사용: {split_dir}")
        return

    rng = np.random.default_rng(seed)
    classes = list_class_dirs(base_dir)
    for cls in classes:
        os.makedirs(os.path.join(test_dir,  cls), exist_ok=True)
        os.makedirs(os.path.join(kfold_dir, cls), exist_ok=True)
    for cls in classes:
        files = list_image_files(os.path.join(base_dir, cls))
        rng.shuffle(files)
        n_test = int(len(files) * test_ratio)
        for f in files[:n_test]:
            shutil.copy2(os.path.join(base_dir, cls, f), os.path.join(test_dir, cls, f))
        for f in files[n_test:]:
            shutil.copy2(os.path.join(base_dir, cls, f), os.path.join(kfold_dir, cls, f))
        print(f"    {cls}: kfold={len(files)-n_test}, test={n_test}")
`;

const pipelineCell = `# Cell 8: tf.data 파이프라인 (512×512 고정)
DATA_OPTIONS = tf.data.Options()
DATA_OPTIONS.experimental_deterministic = True

def _deterministic(ds):
    return ds.with_options(DATA_OPTIONS)

def get_paths_labels(directory, class_names):
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(directory, cls)
        if not os.path.exists(cls_dir): continue
        for f in list_image_files(cls_dir):
            paths.append(os.path.join(cls_dir, f))
            one_hot = [0.0] * len(class_names)
            one_hot[idx] = 1.0
            labels.append(one_hot)
    return paths, labels

@tf.function
def load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32), label

@tf.function
def augment_train(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.25 * 255)
    img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=5.0, dtype=tf.float32)
    img = tf.clip_by_value(img + noise, 0.0, 255.0)
    return img, label

@tf.function
def mixup_batch(x, y, alpha=0.3):
    batch_size = tf.shape(x)[0]
    idx = tf.random.shuffle(tf.range(batch_size))
    lam = tf.cast(tf.random.uniform([], minval=alpha, maxval=1.0 - alpha + 0.001), tf.float32)
    x = tf.cast(x, tf.float32); y = tf.cast(y, tf.float32)
    return lam * x + (1 - lam) * tf.gather(x, idx), lam * y + (1 - lam) * tf.gather(y, idx)

@tf.function
def cutmix_batch(x, y, alpha=0.3):
    batch_size = tf.shape(x)[0]
    idx   = tf.random.shuffle(tf.range(batch_size))
    lam   = tf.cast(tf.random.uniform([], alpha, 1.0 - alpha), tf.float32)
    h     = tf.cast(tf.shape(x)[1], tf.float32)
    w     = tf.cast(tf.shape(x)[2], tf.float32)
    cut_h = tf.cast(h * tf.math.sqrt(1.0 - lam), tf.int32)
    cut_w = tf.cast(w * tf.math.sqrt(1.0 - lam), tf.int32)
    cx    = tf.random.uniform([], 0, tf.cast(h, tf.int32), dtype=tf.int32)
    cy    = tf.random.uniform([], 0, tf.cast(w, tf.int32), dtype=tf.int32)
    x1    = tf.clip_by_value(cx - cut_h // 2, 0, tf.cast(h, tf.int32))
    y1    = tf.clip_by_value(cy - cut_w // 2, 0, tf.cast(w, tf.int32))
    x2    = tf.clip_by_value(cx + cut_h // 2, 0, tf.cast(h, tf.int32))
    y2    = tf.clip_by_value(cy + cut_w // 2, 0, tf.cast(w, tf.int32))
    mask  = tf.ones([x2 - x1, y2 - y1, tf.shape(x)[3]])
    paddings = [[x1, tf.shape(x)[1] - x2], [y1, tf.shape(x)[2] - y2], [0, 0]]
    mask  = tf.pad(mask, paddings)
    mask  = tf.expand_dims(mask, 0)
    x     = tf.cast(x, tf.float32)
    mixed = x * (1.0 - mask) + tf.gather(x, idx) * mask
    lam_actual = 1.0 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / (h * w)
    y     = tf.cast(y, tf.float32)
    mixed_y = lam_actual * y + (1.0 - lam_actual) * tf.gather(y, idx)
    return mixed, mixed_y

def apply_mixup(ds, alpha=0.3):
    return _deterministic(ds.map(lambda x, y: mixup_batch(x, y, alpha),
                                 num_parallel_calls=AUTOTUNE)).prefetch(1)

def apply_cutmix(ds, alpha=0.3):
    return _deterministic(ds.map(lambda x, y: cutmix_batch(x, y, alpha),
                                 num_parallel_calls=AUTOTUNE)).prefetch(1)

def apply_mixup_cutmix(ds, alpha=0.3, cutmix_prob=0.5):
    def _apply(x, y):
        use_cutmix = tf.random.uniform([]) < cutmix_prob
        mx, my = mixup_batch(x, y, alpha)
        cx, cy = cutmix_batch(x, y, alpha)
        return tf.cond(use_cutmix, lambda: (cx, cy), lambda: (mx, my))
    return _deterministic(ds.map(_apply, num_parallel_calls=AUTOTUNE)).prefetch(1)

@tf.function
def augment_tta(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15 * 255)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, label

def make_train_ds(directory, class_names, batch_size=BATCH_SIZE):
    paths, labels = get_paths_labels(directory, class_names)
    n  = len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = _deterministic(ds.shuffle(min(n, 1000), seed=SEED, reshuffle_each_iteration=True))
    ds = _deterministic(ds.map(load_image, num_parallel_calls=AUTOTUNE))
    ds = _deterministic(ds.map(augment_train, num_parallel_calls=AUTOTUNE))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(2)
    return ds, n

def make_eval_ds(directory, class_names, batch_size=BATCH_SIZE):
    paths, labels = get_paths_labels(directory, class_names)
    n  = len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = _deterministic(ds.map(load_image, num_parallel_calls=AUTOTUNE))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(1)
    return ds, n, paths

def make_tta_ds(directory, class_names, batch_size=BATCH_SIZE):
    paths, labels = get_paths_labels(directory, class_names)
    n  = len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = _deterministic(ds.map(load_image, num_parallel_calls=AUTOTUNE))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = _deterministic(ds.map(lambda x, y: augment_tta(x, y), num_parallel_calls=AUTOTUNE))
    ds = ds.prefetch(1)
    return ds, n

print(f"✅ tf.data 파이프라인 정의 완료 (이미지 {IMG_SIZE}×{IMG_SIZE} 고정, deterministic=True)")
`;

const blurLoop = `# Cell 12: 전체 실험 루프 (순차 실행)
print("\\n" + "=" * 60)
print("🔬 블러 처리 강도별 실험 시작 (ResNet50)")
print(f"   Block 순서: {BLOCK_SIZES}")
print(f"   Sigma 순서: {BLUR_SIGMAS}")
print(f"   K={N_FOLDS} | TTA={TTA_STEPS} | BATCH={BATCH_SIZE} | IMG={IMG_SIZE}×{IMG_SIZE}")
print(f"   클래스별 오버샘플링: {TARGET_COUNT}장")
print(f"   실험 버전: {EXPERIMENT_VERSION}")
print("=" * 60)
gpu_status("실험 시작 전")

all_results = {}
result_path = os.path.join(WORKSPACE, f"all_blur_results_resnet50_{EXPERIMENT_VERSION}.json")

if os.path.exists(result_path):
    with open(result_path) as f:
        saved = json_module.load(f)
    for k, v in saved.items():
        all_results[int(k)] = v
    print(f"\\n✅ 이전 결과: {sorted(all_results.keys())}")

for step_idx, (block_equiv, sigma) in enumerate(zip(BLOCK_SIZES, BLUR_SIGMAS), start=1):
    if sigma in all_results:
        print(f"\\n⏭️  Step {step_idx}/{len(BLUR_SIGMAS)} | Block{block_equiv}↔BlurSigma{sigma}: 완료({all_results[sigma]['acc']*100:.2f}%) → 스킵")
        continue
    print(f"\\n📦 Step {step_idx}/{len(BLUR_SIGMAS)} | Block{block_equiv} 대응 BlurSigma{sigma} 데이터셋 준비...")
    blur_dir = generate_blur_dataset(sigma)
    result   = run_experiment(sigma, blur_dir)
    all_results[sigma] = result
    with open(result_path, "w") as f:
        json_module.dump({str(k): v for k, v in all_results.items()}, f, indent=2, ensure_ascii=False)
    print(f"  💾 중간 저장 완료")
    gpu_status(f"Sigma{sigma} 완료 후")
`;

const blockLoop = `# Cell 12: 전체 실험 루프 (순차 실행)
print("\\n" + "=" * 60)
print("🔬 블록 암호화 크기별 실험 시작 (ResNet50)")
print(f"   블록: {BLOCK_SIZES}")
print(f"   K={N_FOLDS} | TTA={TTA_STEPS} | BATCH={BATCH_SIZE} | IMG={IMG_SIZE}")
print(f"   클래스별 오버샘플링: {TARGET_COUNT}장")
print(f"   실험 버전: {EXPERIMENT_VERSION}")
print("=" * 60)
gpu_status("실험 시작 전")

all_results = {}
result_path = os.path.join(WORKSPACE, f"all_block_results_resnet50_{EXPERIMENT_VERSION}.json")

if os.path.exists(result_path):
    with open(result_path) as f:
        saved = json_module.load(f)
    for k, v in saved.items():
        all_results[int(k)] = v
    print(f"\\n✅ 이전 결과: {sorted(all_results.keys())}")

for step_idx, block_size in enumerate(BLOCK_SIZES, start=1):
    if block_size in all_results:
        print(f"\\n⏭️  Step {step_idx}/{len(BLOCK_SIZES)} | Block{block_size}: 완료({all_results[block_size]['acc']*100:.2f}%) → 스킵")
        continue
    print(f"\\n📦 Step {step_idx}/{len(BLOCK_SIZES)} | Block{block_size} 데이터셋 준비...")
    block_dir = generate_block_dataset(block_size)
    result    = run_experiment(block_size, block_dir)
    all_results[block_size] = result
    with open(result_path, "w") as f:
        json_module.dump({str(k): v for k, v in all_results.items()}, f, indent=2, ensure_ascii=False)
    print(f"  💾 중간 저장 완료")
    gpu_status(f"Block{block_size} 완료 후")
`;

const blurResults = `# Cell 13: 결과 표 출력
print("\\n\\n" + "=" * 96)
print("📊 블록 대응 블러 강도별 실험 결과 (ResNet50)")
print("=" * 96)
print(f"  {'단계':>4}  {'Block대응':>10}  {'Sigma':>10}  {'정확도(전체)':>12}  {'ROC-AUC':>8}  {'양성(Benign)':>13}  {'악성(Malig)':>12}  {'정상(Normal)':>13}  {'평가':>6}")
print("-" * 96)
for step_idx, (block_equiv, sigma) in enumerate(zip(BLOCK_SIZES, BLUR_SIGMAS), start=1):
    if sigma not in all_results: continue
    r   = all_results[sigma]
    b   = r["per_class_acc"].get("Bengin cases",    0) * 100
    m   = r["per_class_acc"].get("Malignant cases", 0) * 100
    n   = r["per_class_acc"].get("Normal cases",    0) * 100
    acc = r["acc"] * 100
    lbl = "원본(σ=0)" if sigma == 0 else f"σ={sigma}"
    block_lbl = "원본" if block_equiv >= IMG_SIZE else f"{block_equiv}x{block_equiv}"
    grade = "🟢 우수" if acc >= 85 else ("🟡 양호" if acc >= 70 else ("🟠 미흡" if acc >= 55 else "🔴 불량"))
    print(f"  {step_idx:>4}  {block_lbl:>10}  {lbl:>10}  {acc:>11.2f}%  {r['roc_auc']:>8.4f}"
          f"  {b:>12.1f}%  {m:>11.1f}%  {n:>12.1f}%  {grade}")
print("=" * 96)
`;

function patchRunCell(text, kind) {
  if (kind === "blur") {
    text = replaceRequired(
      text,
      `def run_experiment(sigma, blur_dir):\n    label = "원본" if sigma == 0 else f"BlurSigma{sigma}"\n    print(f"\\n{'='*60}\\n🌫️  {label}\\n{'='*60}")`,
      `def run_experiment(sigma, blur_dir):\n    experiment_idx = BLUR_SIGMAS.index(sigma) if sigma in BLUR_SIGMAS else 0\n    block_equiv = BLOCK_SIZES[experiment_idx] if experiment_idx < len(BLOCK_SIZES) else None\n    exp_seed = SEED + experiment_idx * 1000\n    set_reproducible(exp_seed)\n    label = "원본" if sigma == 0 else f"BlurSigma{sigma}"\n    if block_equiv is not None:\n        label = f"{label} | Block{block_equiv} 대응"\n    print(f"\\n{'='*60}\\n🌫️  {label}\\n{'='*60}")`,
      "blur run header",
    );
    text = replaceRequired(
      text,
      `    split_dir = os.path.join(WORKSPACE, f"AllExp_Split_Blur{sigma}")`,
      `    split_dir = (ORIGINAL_SPLIT_DIR if sigma == 0 else\n                 os.path.join(WORKSPACE, f"AllExp_Split_Blur{sigma}_{EXPERIMENT_VERSION}_seed{SEED}"))`,
      "blur split dir",
    );
    text = replaceRequired(
      text,
      `    oversampled_dir = oversample_dir(blur_dir, TARGET_COUNT)`,
      `    oversampled_dir = oversample_dir(blur_dir, TARGET_COUNT, seed=SEED)`,
      "blur oversample seed",
    );
    text = replaceRequired(
      text,
      `    prepare_split(oversampled_dir, split_dir, TEST_RATIO)`,
      `    prepare_split(oversampled_dir, split_dir, TEST_RATIO, seed=SEED)`,
      "blur split seed",
    );
    text = replaceRequired(
      text,
      `    class_names  = sorted([d for d in os.listdir(kfold_dir)\n                            if os.path.isdir(os.path.join(kfold_dir, d))])`,
      `    class_names  = list_class_dirs(kfold_dir)`,
      "blur class dirs",
    );
    text = replaceRequired(
      text,
      `        for f in os.listdir(os.path.join(kfold_dir, cls)):\n            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):\n                all_paths.append(os.path.join(kfold_dir, cls, f))\n                all_labels.append(class_to_idx[cls])`,
      `        for f in list_image_files(os.path.join(kfold_dir, cls)):\n            all_paths.append(os.path.join(kfold_dir, cls, f))\n            all_labels.append(class_to_idx[cls])`,
      "blur kfold file list",
    );
    text = replaceRequired(
      text,
      `          f"{sum(len(os.listdir(os.path.join(test_dir, c))) for c in class_names)}장")`,
      `          f"{sum(len(list_image_files(os.path.join(test_dir, c))) for c in class_names)}장")`,
      "blur test count",
    );
    text = replaceRequired(
      text,
      `        gpu_status(f"Fold{fold+1} 시작")`,
      `        gpu_status(f"Fold{fold+1} 시작")\n        set_reproducible(exp_seed + fold)`,
      "blur fold seed",
    );
    text = replaceRequired(
      text,
      `            for f in os.listdir(os.path.join(kfold_dir, cls)):\n                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):\n                    shutil.copy2(os.path.join(kfold_dir, cls, f),\n                                 os.path.join(pseudo_kfold, cls, f))`,
      `            for f in list_image_files(os.path.join(kfold_dir, cls)):\n                shutil.copy2(os.path.join(kfold_dir, cls, f),\n                             os.path.join(pseudo_kfold, cls, f))`,
      "blur pseudo kfold list",
    );
    text = replaceRequired(
      text,
      `                for f in os.listdir(p):\n                    shutil.copy2(os.path.join(p, f),\n                                 os.path.join(pseudo_kfold, cls, f"pseudo_{f}"))`,
      `                for f in list_image_files(p):\n                    shutil.copy2(os.path.join(p, f),\n                                 os.path.join(pseudo_kfold, cls, f"pseudo_{f}"))`,
      "blur pseudo list",
    );
    text = replaceRequired(
      text,
      `            cnt = len([f for f in os.listdir(os.path.join(pseudo_kfold, cls))\n                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])`,
      `            cnt = len(list_image_files(os.path.join(pseudo_kfold, cls)))`,
      "blur pseudo count",
    );
    text = replaceFirstRequired(
      text,
      `        for t in range(TTA_STEPS):\n            if t == 0:`,
      `        for t in range(TTA_STEPS):\n            set_reproducible(exp_seed + 10000 + fold * 100 + t)\n            if t == 0:`,
      "blur fold tta seed",
    );
    text = replaceFirstRequired(
      text,
      `        for t in range(TTA_STEPS):\n            if t == 0:`,
      `        for t in range(TTA_STEPS):\n            set_reproducible(exp_seed + 30000 + t)\n            if t == 0:`,
      "blur pseudo tta seed",
    );
    text = replaceRequired(
      text,
      `        pm = load_model(fold_models[0], custom_objects={"loss_fn": focal_loss(2.0, 0.05)})`,
      `        set_reproducible(exp_seed + 50000)\n        pm = load_model(fold_models[0], custom_objects={"loss_fn": focal_loss(2.0, 0.05)})`,
      "blur pseudo training seed",
    );
    text = replaceRequired(
      text,
      `    return {"sigma": sigma,\n            "acc": test_acc, "roc_auc": roc_auc,`,
      `    return {"sigma": sigma, "block_size_equiv": block_equiv,\n            "experiment_seed": exp_seed,\n            "acc": test_acc, "roc_auc": roc_auc,`,
      "blur return",
    );
    return text;
  }

  text = replaceRequired(
    text,
    `def run_experiment(block_size, block_dir):\n    label = "원본" if block_size >= 512 else f"Block{block_size}"\n    print(f"\\n{'='*60}\\n🧩 {label}\\n{'='*60}")`,
    `def run_experiment(block_size, block_dir):\n    experiment_idx = BLOCK_SIZES.index(block_size) if block_size in BLOCK_SIZES else 0\n    exp_seed = SEED + experiment_idx * 1000\n    set_reproducible(exp_seed)\n    label = "원본" if block_size >= IMG_SIZE else f"Block{block_size}"\n    print(f"\\n{'='*60}\\n🧩 {label}\\n{'='*60}")`,
    "block run header",
  );
  text = replaceRequired(
    text,
    `    split_dir = os.path.join(WORKSPACE, f"AllExp_Split_{block_size}")`,
    `    split_dir = (ORIGINAL_SPLIT_DIR if block_size >= IMG_SIZE else\n                 os.path.join(WORKSPACE, f"AllExp_Split_Block{block_size}_{EXPERIMENT_VERSION}_seed{SEED}"))`,
    "block split dir",
  );
  text = replaceRequired(
    text,
    `    oversampled_dir = oversample_dir(block_dir, TARGET_COUNT)`,
    `    oversampled_dir = oversample_dir(block_dir, TARGET_COUNT, seed=SEED)`,
    "block oversample seed",
  );
  text = replaceRequired(
    text,
    `    prepare_split(oversampled_dir, split_dir, TEST_RATIO)`,
    `    prepare_split(oversampled_dir, split_dir, TEST_RATIO, seed=SEED)`,
    "block split seed",
  );
  text = replaceRequired(
    text,
    `    class_names  = sorted([d for d in os.listdir(kfold_dir)\n                            if os.path.isdir(os.path.join(kfold_dir, d))])`,
    `    class_names  = list_class_dirs(kfold_dir)`,
    "block class dirs",
  );
  text = replaceRequired(
    text,
    `        for f in os.listdir(os.path.join(kfold_dir, cls)):\n            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):\n                all_paths.append(os.path.join(kfold_dir, cls, f))\n                all_labels.append(class_to_idx[cls])`,
    `        for f in list_image_files(os.path.join(kfold_dir, cls)):\n            all_paths.append(os.path.join(kfold_dir, cls, f))\n            all_labels.append(class_to_idx[cls])`,
    "block kfold file list",
  );
  text = replaceRequired(
    text,
    `          f"{sum(len(os.listdir(os.path.join(test_dir, c))) for c in class_names)}장")`,
    `          f"{sum(len(list_image_files(os.path.join(test_dir, c))) for c in class_names)}장")`,
    "block test count",
  );
  text = replaceRequired(
    text,
    `        gpu_status(f"Fold{fold+1} 시작")  # GPU 상태 출력`,
    `        gpu_status(f"Fold{fold+1} 시작")  # GPU 상태 출력\n        set_reproducible(exp_seed + fold)`,
    "block fold seed",
  );
  text = replaceRequired(
    text,
    `            for f in os.listdir(os.path.join(kfold_dir, cls)):\n                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):\n                    shutil.copy2(os.path.join(kfold_dir, cls, f),\n                                 os.path.join(pseudo_kfold, cls, f))`,
    `            for f in list_image_files(os.path.join(kfold_dir, cls)):\n                shutil.copy2(os.path.join(kfold_dir, cls, f),\n                             os.path.join(pseudo_kfold, cls, f))`,
    "block pseudo kfold list",
  );
  text = replaceRequired(
    text,
    `                for f in os.listdir(p):\n                    shutil.copy2(os.path.join(p, f),\n                                 os.path.join(pseudo_kfold, cls, f"pseudo_{f}"))`,
    `                for f in list_image_files(p):\n                    shutil.copy2(os.path.join(p, f),\n                                 os.path.join(pseudo_kfold, cls, f"pseudo_{f}"))`,
    "block pseudo list",
  );
  text = replaceRequired(
    text,
    `            cnt = len([f for f in os.listdir(os.path.join(pseudo_kfold, cls))\n                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])`,
    `            cnt = len(list_image_files(os.path.join(pseudo_kfold, cls)))`,
    "block pseudo count",
  );
  text = replaceFirstRequired(
    text,
    `        for t in range(TTA_STEPS):\n            if t == 0:`,
    `        for t in range(TTA_STEPS):\n            set_reproducible(exp_seed + 10000 + fold * 100 + t)\n            if t == 0:`,
    "block fold tta seed",
  );
  text = replaceFirstRequired(
    text,
    `        for t in range(TTA_STEPS):\n            if t == 0:`,
    `        for t in range(TTA_STEPS):\n            set_reproducible(exp_seed + 30000 + t)\n            if t == 0:`,
    "block pseudo tta seed",
  );
  text = replaceRequired(
    text,
    `        pm = load_model(fold_models[0], custom_objects={"loss_fn": focal_loss(2.0, 0.05)})`,
    `        set_reproducible(exp_seed + 50000)\n        pm = load_model(fold_models[0], custom_objects={"loss_fn": focal_loss(2.0, 0.05)})`,
    "block pseudo training seed",
  );
  text = replaceRequired(
    text,
    `    return {"block_size": block_size,\n            "acc": test_acc, "roc_auc": roc_auc,`,
    `    return {"block_size": block_size,\n            "experiment_seed": exp_seed,\n            "acc": test_acc, "roc_auc": roc_auc,`,
    "block return",
  );
  return text;
}

function patchNotebook(kind, cfg) {
  const nb = JSON.parse(fs.readFileSync(cfg.inPath, "utf8"));
  setCell(nb, 8, kind === "blur" ? blurConfig : blockConfig);
  setCell(nb, 10, kind === "blur" ? blurGenerate : blockGenerate);
  setCell(nb, 12, oversampleCell);
  setCell(nb, 14, splitCell);
  setCell(nb, 16, pipelineCell);
  setCell(nb, 24, patchRunCell(asText(nb.cells[24].source), kind));
  setCell(nb, 26, kind === "blur" ? blurLoop : blockLoop);
  if (kind === "blur") setCell(nb, 28, blurResults);

  fs.writeFileSync(cfg.outPath, `${JSON.stringify(nb, null, 1)}\n`, "utf8");
  JSON.parse(fs.readFileSync(cfg.outPath, "utf8"));
  console.log(`${kind}: wrote ${cfg.outPath}`);
}

patchNotebook("blur", notebooks.blur);
patchNotebook("block", notebooks.block);
