import ast
import json
from pathlib import Path


WORKDIR = Path(r".")
FILES = [
    WORKDIR / "blur_resnet50_rebuilt_KO.ipynb",
    WORKDIR / "block_encryption_resnet50_rebuilt_KO.ipynb",
]


def text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_text(cell, src):
    cell["source"] = [line if line.endswith("\n") else line + "\n" for line in src.splitlines()]
    cell["outputs"] = []
    cell["execution_count"] = None


old = '''callbacks=[EarlyStopping(monitor="val_loss", patience=7,
                                         restore_best_weights=True, verbose=0)],'''
new = '''callbacks=[EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4,
                                         restore_best_weights=True, verbose=0),
                           ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                             min_lr=1e-7, verbose=0),
                           TerminateOnNaN(),
                           OverfitControlCallback(min_epochs=8, patience=4, acc_gap=0.18, loss_gap=0.20)],'''

for path in FILES:
    nb = json.loads(path.read_text(encoding="utf-8"))
    src = text(nb["cells"][24])
    if old not in src:
        raise RuntimeError(f"pseudo callback snippet not found: {path.name}")
    src = src.replace(old, new, 1)
    set_text(nb["cells"][24], src)
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse(text(cell), filename=f"{path.name}:cell{i}")
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
