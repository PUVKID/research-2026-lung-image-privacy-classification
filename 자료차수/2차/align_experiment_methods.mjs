import { copyFile, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";

const paths = {
  blur: "./blur_resnet50_rebuilt2_KO.ipynb",
  block: "./block_encryption_resnet50_rebuilt_KO.ipynb",
  combo: "./block_blur_resnet50_all_combinations_KO.ipynb",
  blockBackup: "./block_encryption_resnet50_rebuilt_KO_before_alignment.ipynb",
};

const books = {};
for (const name of ["blur", "block", "combo"]) {
  books[name] = JSON.parse(await readFile(paths[name], "utf8"));
}

function getSource(book, index) {
  const source = book.cells[index].source;
  return Array.isArray(source) ? source.join("") : source;
}

function setSource(book, index, source) {
  book.cells[index].source = source.match(/.*(?:\n|$)/g).filter(Boolean);
}

function replaceOrThrow(source, before, after, label) {
  if (!source.includes(before)) {
    throw new Error(`Missing replacement target: ${label}`);
  }
  return source.replace(before, after);
}

function addGpuLogCompatibility(source) {
  if (source.includes('"used_mib": used,')) {
    return source;
  }
  return replaceOrThrow(
    source,
    `                GPU_LOG.append({
                    "sigma": self.sigma,
                    "fold": self.fold_idx,
                    "epoch": epoch + 1,
                    "used": used,
                    "total": total,
                    "util": util,
                    "temp": temp,
                })`,
    `                GPU_LOG.append({
                    "time": time.time(),
                    "sigma": self.sigma,
                    "fold": self.fold_idx,
                    "epoch": epoch + 1,
                    "used": used,
                    "total": total,
                    "util": util,
                    "temp": temp,
                    "used_mib": used,
                    "total_mib": total,
                    "util_pct": util,
                    "temp_c": temp,
                })`,
    "blur GPU log compatibility",
  );
}

function addComboGpuLogCompatibility(source) {
  if (source.includes('"used_mib": used,')) {
    return source;
  }
  return replaceOrThrow(
    source,
    `                GPU_LOG.append({
                    "condition": self.condition,
                    "fold": self.fold_idx,
                    "epoch": epoch + 1,
                    "used": used,
                    "total": total,
                    "util": util,
                    "temp": temp,
                })`,
    `                GPU_LOG.append({
                    "time": time.time(),
                    "condition": self.condition,
                    "fold": self.fold_idx,
                    "epoch": epoch + 1,
                    "used": used,
                    "total": total,
                    "util": util,
                    "temp": temp,
                    "used_mib": used,
                    "total_mib": total,
                    "util_pct": util,
                    "temp_c": temp,
                })`,
    "combo GPU log compatibility",
  );
}

function makeGpuGraphRobust(source) {
  const old = `    times     = [(g["time"] - _log[0]["time"]) / 60 for g in _log]
    used_vals = [g["used_mib"] for g in _log]
    util_vals = [g["util_pct"] for g in _log]
    temp_vals = [g["temp_c"]   for g in _log]
    total_mib = _log[0]["total_mib"]`;
  const replacement = `    if all("time" in g for g in _log):
        times = [(g["time"] - _log[0]["time"]) / 60 for g in _log]
    else:
        times = [i * 5 for i in range(len(_log))]
    used_vals = [g.get("used_mib", g.get("used", 0)) for g in _log]
    util_vals = [g.get("util_pct", g.get("util", 0)) for g in _log]
    temp_vals = [g.get("temp_c", g.get("temp", 0)) for g in _log]
    total_mib = _log[0].get("total_mib", _log[0].get("total", 0))`;
  if (source.includes(replacement)) {
    return source;
  }
  return replaceOrThrow(source, old, replacement, "GPU graph compatibility");
}

function makeGpuSummaryRobust(source) {
  const old = `    used_vals = [g["used"] for g in GPU_LOG]
    util_vals = [g["util"] for g in GPU_LOG]
    temp_vals = [g["temp"] for g in GPU_LOG]`;
  const oldAlternate = `    used_vals = [g["used_mib"] for g in GPU_LOG]
    util_vals = [g["util_pct"] for g in GPU_LOG]
    temp_vals = [g["temp_c"]   for g in GPU_LOG]`;
  const replacement = `    used_vals = [g.get("used", g.get("used_mib", 0)) for g in GPU_LOG]
    util_vals = [g.get("util", g.get("util_pct", 0)) for g in GPU_LOG]
    temp_vals = [g.get("temp", g.get("temp_c", 0)) for g in GPU_LOG]`;
  if (source.includes(replacement)) {
    return source;
  }
  if (source.includes(old)) {
    return source.replace(old, replacement);
  }
  if (source.includes(oldAlternate)) {
    return source.replace(oldAlternate, replacement);
  }
  return source;
}

// Keep the running blur design intact while making its monitor output compatible
// with its own graph cell.
setSource(books.blur, 22, addGpuLogCompatibility(getSource(books.blur, 22)));
setSource(books.blur, 30, makeGpuSummaryRobust(getSource(books.blur, 30)));
setSource(books.blur, 32, makeGpuGraphRobust(getSource(books.blur, 32)));

// The combined notebook already follows the blur training method. Add the same
// backward-compatible GPU fields so reporting also stays aligned.
setSource(books.combo, 22, addComboGpuLogCompatibility(getSource(books.combo, 22)));
setSource(books.combo, 30, makeGpuSummaryRobust(getSource(books.combo, 30)));
setSource(books.combo, 32, makeGpuGraphRobust(getSource(books.combo, 32)));

// Rebuild the block experiment's common pipeline from the blur experiment so
// data splitting, oversampling, steps, callbacks, TTA, and pseudo training match.
let blockSettings = getSource(books.block, 8);
blockSettings = replaceOrThrow(
  blockSettings,
  'EXPERIMENT_VERSION = "same_settings_gpu4090_v2"',
  'EXPERIMENT_VERSION = "same_settings_gpu4090_v3_aligned"',
  "block experiment version",
);
setSource(books.block, 8, blockSettings);

setSource(books.block, 12, getSource(books.blur, 12));
setSource(books.block, 14, getSource(books.blur, 14));

let blockMonitor = getSource(books.blur, 22);
blockMonitor = replaceOrThrow(
  blockMonitor,
  "def print_fold_result(sigma, fold_idx, combined_history, stage=\"\"):",
  "def print_fold_result(block_size, fold_idx, combined_history, stage=\"\"):",
  "block fold result signature",
);
blockMonitor = replaceOrThrow(
  blockMonitor,
  '    label = "원본(σ=0)" if sigma == 0 else f"BlurSigma σ={sigma}"',
  '    label = "원본" if block_size >= IMG_SIZE else f"Block{block_size}"',
  "block fold label",
);
blockMonitor = replaceOrThrow(
  blockMonitor,
  "    def __init__(self, sigma, fold_idx, every_n_epochs=5, overfit_gap=0.20):",
  "    def __init__(self, block_size, fold_idx, every_n_epochs=5, overfit_gap=0.20):",
  "block monitor signature",
);
blockMonitor = replaceOrThrow(
  blockMonitor,
  "        self.sigma = sigma",
  "        self.block_size = block_size",
  "block monitor field",
);
blockMonitor = replaceOrThrow(
  blockMonitor,
  '                    "sigma": self.sigma,',
  '                    "block_size": self.block_size,',
  "block GPU log key",
);
setSource(books.block, 22, blockMonitor);

let blockRunner = getSource(books.blur, 24);
blockRunner = replaceOrThrow(
  blockRunner,
  `# Cell 11: 단일 블러 Sigma 실험 함수 (누수 없는 올바른 순서)
def run_experiment(sigma, blur_dir):
    exp_idx  = BLUR_SIGMAS.index(sigma) if sigma in BLUR_SIGMAS else 0
    exp_seed = SEED + exp_idx * 1000
    set_reproducible(exp_seed)

    label = "원본(σ=0)" if sigma == 0 else f"BlurSigma σ={sigma}"
    print(f"\\n{'='*60}\\n🌫️ {label}  (seed={exp_seed})\\n{'='*60}")

    split_dir = (ORIGINAL_SPLIT_DIR if sigma == 0 else
                 os.path.join(WORKSPACE,
                              f"AllExp_Split_BlurSigma{sigma}_{EXPERIMENT_VERSION}_seed{SEED}"))`,
  `# Cell 11: 단일 블록 크기 실험 함수 (누수 없는 올바른 순서)
def run_experiment(block_size, block_dir):
    exp_idx  = BLOCK_SIZES.index(block_size) if block_size in BLOCK_SIZES else 0
    exp_seed = SEED + exp_idx * 1000
    set_reproducible(exp_seed)

    label = "원본" if block_size >= IMG_SIZE else f"Block{block_size}"
    print(f"\\n{'='*60}\\n🧩 {label}  (seed={exp_seed})\\n{'='*60}")

    split_dir = (ORIGINAL_SPLIT_DIR if block_size >= IMG_SIZE else
                 os.path.join(WORKSPACE,
                              f"AllExp_Split_Block{block_size}_{EXPERIMENT_VERSION}_seed{SEED}"))`,
  "block runner prelude",
);
blockRunner = replaceOrThrow(
  blockRunner,
  "    prepare_split(blur_dir, split_dir, TEST_RATIO, seed=SEED)",
  "    prepare_split(block_dir, split_dir, TEST_RATIO, seed=SEED)",
  "block split input",
);
blockRunner = replaceOrThrow(
  blockRunner,
  "    if sigma not in ALL_HISTORIES:\n        ALL_HISTORIES[sigma] = {}",
  "    if block_size not in ALL_HISTORIES:\n        ALL_HISTORIES[block_size] = {}",
  "block history init",
);
blockRunner = replaceOrThrow(
  blockRunner,
  '        prefix = f"blur_sigma{sigma}_resnet50_f{fold}"',
  '        prefix = f"block{block_size}_resnet50_f{fold}"',
  "block model prefix",
);
blockRunner = replaceOrThrow(
  blockRunner,
  "        monitor_cb = AdvancedMonitorCallback(sigma, fold, every_n_epochs=5)",
  "        monitor_cb = AdvancedMonitorCallback(block_size, fold, every_n_epochs=5)",
  "block monitor call",
);
blockRunner = replaceOrThrow(
  blockRunner,
  "        ALL_HISTORIES[sigma][fold] = combined\n        print_fold_result(sigma, fold, combined)",
  "        ALL_HISTORIES[block_size][fold] = combined\n        print_fold_result(block_size, fold, combined)",
  "block history save",
);
blockRunner = replaceOrThrow(
  blockRunner,
  "    print(f\"\\n  ✅ Sigma={sigma}: {test_acc*100:.2f}% | ROC-AUC={roc_auc:.4f}\")\n\n    return {\n        \"sigma\": sigma,",
  "    print(f\"\\n  ✅ Block{block_size}: {test_acc*100:.2f}% | ROC-AUC={roc_auc:.4f}\")\n\n    return {\n        \"block_size\": block_size,",
  "block result payload",
);
blockRunner = replaceOrThrow(
  blockRunner,
  "print(\"✅ Cell 11: run_experiment() 정의 완료\")",
  "print(\"✅ Cell 11: run_experiment() 블록 버전 정의 완료\")",
  "block runner completion",
);
setSource(books.block, 24, blockRunner);
setSource(books.block, 30, makeGpuSummaryRobust(getSource(books.block, 30)));
setSource(books.block, 32, makeGpuGraphRobust(getSource(books.block, 32)));

for (const [name, book] of Object.entries(books)) {
  book.metadata = book.metadata || {};
  book.metadata.experiment_method_alignment = {
    canonical_method: "blur_resnet50_rebuilt2_KO.ipynb leak-free pipeline",
    split_order: "transform -> split original-only test/kfold -> oversample train-only",
    training: "ResNet50 + Focal Loss + MixUp/CutMix + Cosine LR + SWA + 7-Fold + TTA20 + pseudo labeling",
    aligned_at: "2026-06-02",
  };
}

if (!existsSync(paths.blockBackup)) {
  await copyFile(paths.block, paths.blockBackup);
}

for (const name of ["blur", "block", "combo"]) {
  await writeFile(paths[name], `${JSON.stringify(books[name], null, 1)}\n`, "utf8");
  console.log(`UPDATED ${paths[name]}`);
}
console.log(`BACKUP ${paths.blockBackup}`);
