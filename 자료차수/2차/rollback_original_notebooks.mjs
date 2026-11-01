import { copyFile, readFile, writeFile } from "node:fs/promises";

const paths = {
  blur: "./blur_resnet50_rebuilt2_KO.ipynb",
  block: "./block_encryption_resnet50_rebuilt_KO.ipynb",
  blockBackup: "./block_encryption_resnet50_rebuilt_KO_before_alignment.ipynb",
};

function getSource(book, index) {
  const source = book.cells[index].source;
  return Array.isArray(source) ? source.join("") : source;
}

function setSource(book, index, source) {
  book.cells[index].source = source.match(/.*(?:\n|$)/g).filter(Boolean);
}

function replaceOrThrow(source, before, after, label) {
  if (!source.includes(before)) {
    throw new Error(`Missing rollback target: ${label}`);
  }
  return source.replace(before, after);
}

const blur = JSON.parse(await readFile(paths.blur, "utf8"));

let monitor = getSource(blur, 22);
monitor = replaceOrThrow(
  monitor,
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
  `                GPU_LOG.append({
                    "sigma": self.sigma,
                    "fold": self.fold_idx,
                    "epoch": epoch + 1,
                    "used": used,
                    "total": total,
                    "util": util,
                    "temp": temp,
                })`,
  "blur monitor log",
);
setSource(blur, 22, monitor);

let summary = getSource(blur, 30);
summary = replaceOrThrow(
  summary,
  `    used_vals = [g.get("used", g.get("used_mib", 0)) for g in GPU_LOG]
    util_vals = [g.get("util", g.get("util_pct", 0)) for g in GPU_LOG]
    temp_vals = [g.get("temp", g.get("temp_c", 0)) for g in GPU_LOG]`,
  `    used_vals = [g["used"] for g in GPU_LOG]
    util_vals = [g["util"] for g in GPU_LOG]
    temp_vals = [g["temp"] for g in GPU_LOG]`,
  "blur summary log",
);
setSource(blur, 30, summary);

let graph = getSource(blur, 32);
graph = replaceOrThrow(
  graph,
  `    if all("time" in g for g in _log):
        times = [(g["time"] - _log[0]["time"]) / 60 for g in _log]
    else:
        times = [i * 5 for i in range(len(_log))]
    used_vals = [g.get("used_mib", g.get("used", 0)) for g in _log]
    util_vals = [g.get("util_pct", g.get("util", 0)) for g in _log]
    temp_vals = [g.get("temp_c", g.get("temp", 0)) for g in _log]
    total_mib = _log[0].get("total_mib", _log[0].get("total", 0))`,
  `    times     = [(g["time"] - _log[0]["time"]) / 60 for g in _log]
    used_vals = [g["used_mib"] for g in _log]
    util_vals = [g["util_pct"] for g in _log]
    temp_vals = [g["temp_c"]   for g in _log]
    total_mib = _log[0]["total_mib"]`,
  "blur graph log",
);
setSource(blur, 32, graph);

delete blur.metadata?.experiment_method_alignment;
await writeFile(paths.blur, `${JSON.stringify(blur, null, 1)}\n`, "utf8");

await copyFile(paths.blockBackup, paths.block);

console.log(`ROLLED_BACK ${paths.blur}`);
console.log(`ROLLED_BACK ${paths.block}`);
