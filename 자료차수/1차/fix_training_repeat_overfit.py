import ast
import json
from pathlib import Path


WORKDIR = Path(r".")
FILES = [
    Path(r".\blur_resnet50_rebuilt_KO.ipynb"),
    Path(r".\block_encryption_resnet50_rebuilt_KO.ipynb"),
]


def cell_text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_cell_text(cell, src):
    cell["source"] = [line if line.endswith("\n") else line + "\n" for line in src.splitlines()]
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def replace_once(src, old, new, label):
    if old not in src:
        raise RuntimeError(f"snippet not found: {label}")
    return src.replace(old, new, 1)


def patch_notebook(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    name = path.name
    is_blur = name.startswith("blur_")

    # Cell 1: callback imports only.
    src = cell_text(nb["cells"][2])
    src = replace_once(
        src,
        "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback",
        "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TerminateOnNaN, ReduceLROnPlateau",
        f"{name} imports",
    )
    set_cell_text(nb["cells"][2], src)

    # Cell 4: use more of RTX 4090 and allow more useful epochs.
    src = cell_text(nb["cells"][8])
    src = src.replace("BATCH_SIZE   = 4", "BATCH_SIZE   = 8")
    src = src.replace("EPOCHS_S1    = 30", "EPOCHS_S1    = 80")
    src = src.replace("EPOCHS_S2    = 60", "EPOCHS_S2    = 200")
    src = src.replace("SWA_START    = 20", "SWA_START    = 40")
    set_cell_text(nb["cells"][8], src)

    # Cell 8: make train dataset infinite for fit(); eval/TTA stays finite.
    src = cell_text(nb["cells"][16])
    src = replace_once(
        src,
        "    ds = ds.batch(batch_size, drop_remainder=True)\n    ds = ds.prefetch(2)\n    return ds, n",
        "    ds = ds.batch(batch_size, drop_remainder=True)\n    ds = ds.repeat()\n    ds = ds.prefetch(AUTOTUNE)\n    return ds, n",
        f"{name} train repeat",
    )
    set_cell_text(nb["cells"][16], src)

    # Cell 10b: add actual overfit stop callback without changing markdown/design.
    src = cell_text(nb["cells"][22])
    marker = "\ndef plot_training_diagnosis("
    if "class OverfitControlCallback" not in src:
        overfit_callback = r'''

class OverfitControlCallback(Callback):
    """과적합이 여러 에폭 연속 지속되면 현재 학습 단계를 조기 종료한다."""
    def __init__(self, min_epochs=20, patience=8, acc_gap=0.18, loss_gap=0.20):
        super().__init__()
        self.min_epochs = min_epochs
        self.patience = patience
        self.acc_gap = acc_gap
        self.loss_gap = loss_gap
        self.bad_count = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch + 1 < self.min_epochs:
            return
        train_acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if None in (train_acc, val_acc, train_loss, val_loss):
            return
        is_overfit = (train_acc - val_acc > self.acc_gap) and (val_loss - train_loss > self.loss_gap)
        self.bad_count = self.bad_count + 1 if is_overfit else max(0, self.bad_count - 1)
        if self.bad_count >= self.patience:
            print(f"\n  🛑 과적합 방지: {self.bad_count}회 연속 과적합 신호 → 현재 학습 단계 종료")
            self.model.stop_training = True
'''
        src = src.replace(marker, overfit_callback + marker, 1)
    set_cell_text(nb["cells"][22], src)

    # Cell 11: fix steps, repeat validation only inside fit, add anti-overfit callbacks.
    src = cell_text(nb["cells"][24])
    src = src.replace("train_steps = math.ceil(train_n / BATCH_SIZE)", "train_steps = max(1, train_n // BATCH_SIZE)")
    src = src.replace("steps_per_epoch=math.ceil(p_n / BATCH_SIZE)", "steps_per_epoch=max(1, p_n // BATCH_SIZE)")

    src = src.replace("validation_data=val_ds, validation_steps=val_steps", "validation_data=val_ds.repeat(), validation_steps=val_steps")
    src = src.replace("validation_data=p_val_ds, validation_steps=math.ceil(p_vn / BATCH_SIZE)", "validation_data=p_val_ds.repeat(), validation_steps=math.ceil(p_vn / BATCH_SIZE)")

    src = src.replace(
        'EarlyStopping(monitor="val_loss", patience=10,\n                              restore_best_weights=True, verbose=0),',
        'EarlyStopping(monitor="val_loss", patience=20, min_delta=1e-4,\n                              restore_best_weights=True, verbose=0),\n                TerminateOnNaN(),\n                OverfitControlCallback(min_epochs=20, patience=8, acc_gap=0.18, loss_gap=0.20),',
    )
    src = src.replace(
        'EarlyStopping(monitor="val_loss", patience=20,\n                              restore_best_weights=False, verbose=0),',
        'EarlyStopping(monitor="val_loss", patience=45, min_delta=1e-4,\n                              restore_best_weights=True, verbose=0),\n                TerminateOnNaN(),\n                OverfitControlCallback(min_epochs=35, patience=10, acc_gap=0.16, loss_gap=0.18),',
    )
    src = src.replace(
        'callbacks=[EarlyStopping(monitor="val_loss", patience=7,\n                                         restore_best_weights=True, verbose=0)],',
        'callbacks=[EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4,\n                                         restore_best_weights=True, verbose=0),\n                           ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,\n                                             min_lr=1e-7, verbose=0),\n                           TerminateOnNaN(),\n                           OverfitControlCallback(min_epochs=8, patience=4, acc_gap=0.18, loss_gap=0.20)],',
    )

    # The pseudo-label validation replacement has a spacing difference between files.
    src = src.replace("validation_data=p_val_ds, validation_steps=math.ceil(p_vn / BATCH_SIZE)", "validation_data=p_val_ds.repeat(), validation_steps=math.ceil(p_vn / BATCH_SIZE)")

    set_cell_text(nb["cells"][24], src)

    # Validate code cells.
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse(cell_text(cell), filename=f"{name}:cell{i}")

    out = WORKDIR / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.loads(out.read_text(encoding="utf-8"))
    return out


for file_path in FILES:
    print(patch_notebook(file_path))
