"""Export benchmark comparisons as standalone, presentation-ready PNG figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finish(fig, folder, name):
    path = folder / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _bar(rows, dataset, folder, slug):
    names = [row["name"] for row in rows]
    ranks = [100 * row["datasets"][dataset]["rank1"] for row in rows]
    maps = [100 * row["datasets"][dataset]["map"] for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.57 * len(rows) + 1.8)))
    y = np.arange(len(rows))
    ax.barh(y - .18, ranks, height=.34, label="Rank-1", color="#087e8b")
    ax.barh(y + .18, maps, height=.34, label="mAP", color="#e46c28")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Retrieval accuracy (%)")
    ax.set_title(f"{dataset}: same query and gallery crops" +
                 (" (exploratory test-only)" if slug == "bucktales" else ""))
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _finish(fig, folder, f"{slug}_rank1_map.png")


def _leaderboard(rows, dataset, folder, slug):
    ranked = sorted(rows, key=lambda row: row["datasets"][dataset]["rank1"], reverse=True)
    names = [f"{index}. {row['name']}" for index, row in enumerate(ranked, 1)]
    scores = [100 * row["datasets"][dataset]["rank1"] for row in ranked]
    colors = ["#e46c28" if "ImageNet features" in row["name"] else "#087e8b"
              for row in ranked]
    fig, ax = plt.subplots(figsize=(12, max(5, .62 * len(rows) + 1.8)))
    bars = ax.barh(names, scores, color=colors)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set(xlim=(0, max(scores) + 12), xlabel="Rank-1 (%)",
           title=f"{dataset}: local ReID leaderboard" +
           (" (exploratory test-only)" if slug == "bucktales" else ""))
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=(Patch(color="#087e8b", label="Project checkpoints"),
                       Patch(color="#e46c28", label="Frozen ImageNet references")),
              loc="lower right")
    fig.tight_layout()
    return _finish(fig, folder, f"{slug}_leaderboard.png")


def _cmc(rows, dataset, folder, slug):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    for row, color in zip(rows, colors):
        ax.plot(range(1, 11), np.array(row["datasets"][dataset]["cmc"]) * 100,
                marker="o", markersize=3, linewidth=2, label=row["name"], color=color)
    ax.set(xlim=(1, 10), ylim=(0, 100), xlabel="Gallery rank",
           ylabel="Correct identity found by rank (%)",
           title=f"{dataset}: CMC" + (" (exploratory)" if slug == "bucktales" else ""))
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    return _finish(fig, folder, f"{slug}_cmc.png")


def _pair_heatmap(rows, folder, key, name):
    pairs = sorted(rows[0]["bucktales_pairs"])
    values = np.array([[100 * row["bucktales_pairs"][pair][key] for pair in pairs]
                       for row in rows])
    fig, ax = plt.subplots(figsize=(max(12, len(pairs) * 1.1), max(5, len(rows) * .6 + 1.5)))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pairs)), pairs, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [row["name"] for row in rows])
    ax.set_title(f"BuckTales by drone pair: {name} (%) — exploratory test-only")
    for i in range(len(rows)):
        for j in range(len(pairs)):
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center",
                    color="white" if values[i, j] > 55 else "#172a37", fontsize=8)
    fig.colorbar(image, ax=ax, label=f"{name} (%)")
    fig.tight_layout()
    return _finish(fig, folder, f"bucktales_pairs_{key}.png")


def _pooling(rows, folder):
    indexed = {row["name"]: row for row in rows}
    pairs = [(row["name"], row["name"] + " (global pool)") for row in rows
             if row["name"] + " (global pool)" in indexed]
    if not pairs:
        return None
    datasets = list(rows[0]["datasets"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        x = np.arange(len(pairs))
        masked = [100 * indexed[name]["datasets"][dataset]["rank1"] for name, _ in pairs]
        global_scores = [100 * indexed[other]["datasets"][dataset]["rank1"] for _, other in pairs]
        ax.bar(x - .2, masked, .4, label="Mask-guided", color="#087e8b")
        ax.bar(x + .2, global_scores, .4, label="Global", color="#e46c28")
        ax.set_xticks(x, [name.removeprefix("Joint ") for name, _ in pairs])
        ax.set(title=dataset, ylabel="Rank-1 (%)", ylim=(0, 100))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0][0].legend()
    fig.suptitle("Pooling ablation: identical checkpoint weights")
    fig.tight_layout()
    return _finish(fig, folder, "pooling_ablation_rank1.png")


def _matched(rows, folder):
    indexed = {row["name"]: row for row in rows}
    names = ("ReID only matched", "Joint matched")
    if not all(name in indexed for name in names):
        return None
    datasets = list(rows[0]["datasets"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        x = np.arange(2)
        for offset, key, color in ((-.2, "rank1", "#087e8b"), (.2, "map", "#e46c28")):
            ax.bar(x + offset, [100 * indexed[name]["datasets"][dataset][key] for name in names],
                   .4, label="Rank-1" if key == "rank1" else "mAP", color=color)
        ax.set_xticks(x, ["ReID only", "Joint"])
        ax.set(title=dataset, ylabel="Retrieval accuracy (%)", ylim=(0, 100))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0][0].legend()
    fig.suptitle("Matched start, identity data, and ReID update count")
    fig.tight_layout()
    return _finish(fig, folder, "matched_reid_vs_joint.png")


def _voc(rows, folder):
    present = [row for row in rows if row["voc_miou"] is not None and
               "(global pool)" not in row["name"]]
    fig, ax = plt.subplots(figsize=(10, max(4, .55 * len(present) + 1.8)))
    ax.barh([row["name"] for row in present], [100 * row["voc_miou"] for row in present],
            color="#4e9b55")
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Mean intersection over union (%)",
           title="VOC 2012 validation: 21-class semantic masks")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _finish(fig, folder, "voc_mask_miou.png")


def _efficiency(rows, dataset, folder, slug):
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    for row, color in zip(rows, colors):
        ax.scatter(row["latency_ms"], 100 * row["datasets"][dataset]["rank1"],
                   s=max(50, row["params"] / 100_000), color=color,
                   edgecolor="white", linewidth=1, label=row["name"])
    ax.set(xlabel="PC CPU time per 224×224 crop (ms, lower is better)",
           ylabel="Rank-1 (%)", title=f"{dataset}: speed versus retrieval accuracy" +
           (" (exploratory)" if slug == "bucktales" else ""))
    ax.grid(alpha=.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    return _finish(fig, folder, f"{slug}_speed_accuracy.png")


def _tradeoff(rows, dataset, folder, slug, max_latency=None):
    """ReID accuracy versus measured PC latency, with size variants linked by family."""
    indexed = {row["name"]: row for row in rows}
    families = (
        ("ShuffleNetV2", ("ShuffleNetV2 x1.0", "ShuffleNetV2 x1.5"), "#6a5acd"),
        ("MobileNetV3", ("MobileNetV3-Small", "MobileNetV3-Large"), "#16a085"),
        ("EfficientNet", ("EfficientNet-B0", "EfficientNet-B2"), "#e67e22"),
        ("MobileNetV2", ("MobileNetV2",), "#95a000"),
        ("RegNetY", ("RegNetY-400MF",), "#8e44ad"),
        ("ConvNeXt", ("ConvNeXt-Tiny",), "#7f8c8d"),
        ("Swin", ("Swin-T",), "#c0392b"),
        ("ResNet", ("ResNet-18",), "#a67c52"),
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    for family, variants, color in families:
        selected = [indexed[f"{variant} ImageNet features"] for variant in variants
                    if f"{variant} ImageNet features" in indexed and
                    (max_latency is None or indexed[f"{variant} ImageNet features"]["latency_ms"] <= max_latency)]
        if selected:
            selected.sort(key=lambda row: row["latency_ms"])
            ax.plot([row["latency_ms"] for row in selected],
                    [100 * row["datasets"][dataset]["rank1"] for row in selected],
                    marker="o", markersize=8, linewidth=2, label=f"{family} (frozen ImageNet)",
                    color=color)
    project = [row for row in rows if "ImageNet features" not in row["name"]]
    label_offsets = {
        "Joint balanced": (8, -12),
        "Joint balanced (global pool)": (8, 10),
        "ReID only matched": (8, 14),
        "Joint multiview": (8, 7),
        "Joint multiview (global pool)": (8, -13),
    }
    for row in project:
        ax.scatter(row["latency_ms"], 100 * row["datasets"][dataset]["rank1"],
                   marker="*" if "matched" in row["name"] else "D", s=90,
                   color="#087e8b", edgecolor="white", linewidth=.8, zorder=5)
        label = row["name"] if row["name"] == "Joint matched" else row["name"].removeprefix("Joint ")
        ax.annotate(label,
                    (row["latency_ms"], 100 * row["datasets"][dataset]["rank1"]),
                    xytext=label_offsets.get(row["name"], (8, 5)),
                    textcoords="offset points", fontsize=8)
    ax.set(xlabel="PC CPU latency per 224×224 crop (ms)",
           ylabel="ReID Rank-1 (%)",
           title=f"{dataset}: ReID accuracy versus measured latency" +
           (f" (≤{max_latency:g} ms)" if max_latency is not None else "") +
           (" (exploratory)" if slug == "bucktales" else ""))
    from matplotlib.ticker import MultipleLocator
    if max_latency is None:
        limit = max(row["latency_ms"] for row in rows)
        ax.set_xlim(0, max(10, limit * 1.12))
        ax.xaxis.set_major_locator(MultipleLocator(10 if limit < 120 else 20))
    else:
        ax.set_xlim(0, max_latency)
        ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(alpha=.25, which="both")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Line2D([], [], marker="D", linestyle="None", color="#087e8b", label="Project checkpoints"))
    labels.insert(0, "Project checkpoints")
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    suffix = "_zoom" if max_latency is not None else ""
    return _finish(fig, folder, f"{slug}_accuracy_latency{suffix}.png")


def _parameters(rows, folder):
    fig, ax = plt.subplots(figsize=(10, max(5, .57 * len(rows) + 1.8)))
    ax.barh([row["name"] for row in rows], [row["params"] / 1e6 for row in rows],
            color="#7755aa")
    ax.invert_yaxis()
    ax.set(xlabel="Active parameters (millions)", title="Model size at inference")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _finish(fig, folder, "active_parameters.png")


def save_figures(rows, datasets, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, dataset in enumerate(datasets):
        slug = "aicity" if index == 0 else "bucktales"
        paths.extend((_leaderboard(rows, dataset, folder, slug),
                      _bar(rows, dataset, folder, slug), _cmc(rows, dataset, folder, slug),
                      _efficiency(rows, dataset, folder, slug),
                      _tradeoff(rows, dataset, folder, slug),
                      _tradeoff(rows, dataset, folder, slug, max_latency=25)))
    paths.extend((_pair_heatmap(rows, folder, "rank1", "Rank-1"),
                  _pair_heatmap(rows, folder, "map", "mAP"), _voc(rows, folder),
                  _parameters(rows, folder)))
    for path in (_pooling(rows, folder), _matched(rows, folder)):
        if path is not None:
            paths.append(path)
    return paths
