"""Render the local same-data benchmark as a self-contained HTML report."""

import html
import math
import platform

COLORS = ("#087e8b", "#e46c28", "#7755aa", "#4e9b55", "#c44e62", "#af8b16", "#426eb1")


def _bar_svg(rows, dataset):
    width = 960
    height = 52 + 62 * len(rows)
    plot_left, plot_width = 250, 630
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Rank-1 and mAP on {html.escape(dataset)}">']
    for tick in (0, 25, 50, 75, 100):
        x = plot_left + tick * plot_width / 100
        parts.append(f'<line x1="{x}" y1="28" x2="{x}" y2="{height - 12}" stroke="#e2e8ee"/>')
        parts.append(f'<text x="{x}" y="19" text-anchor="middle" class="tick">{tick}%</text>')
    for index, row in enumerate(rows):
        top = 38 + index * 62
        color = COLORS[index % len(COLORS)]
        name = html.escape(row["name"])
        metrics = row["datasets"][dataset]
        parts.append(f'<text x="8" y="{top + 19}" class="label">{name}</text>')
        for offset, key, opacity in ((0, "rank1", 1), (23, "map", .5)):
            value = metrics[key] * 100
            bar_width = value * plot_width / 100
            parts.append(f'<rect x="{plot_left}" y="{top + offset}" width="{bar_width:.1f}" height="17" rx="3" fill="{color}" opacity="{opacity}"/>')
            parts.append(f'<text x="{plot_left + bar_width + 7:.1f}" y="{top + offset + 13}" class="value">{value:.1f}</text>')
    parts.append('</svg>')
    return "".join(parts)


def _cmc_svg(rows, dataset):
    width, height = 960, 340
    left, right, top, bottom = 65, 935, 25, 285
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="CMC curves on {html.escape(dataset)}">']
    for tick in (0, 25, 50, 75, 100):
        y = bottom - tick * (bottom - top) / 100
        parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#e2e8ee"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" class="tick">{tick}%</text>')
    for rank in (1, 3, 5, 7, 10):
        x = left + (rank - 1) * (right - left) / 9
        parts.append(f'<text x="{x}" y="{bottom + 21}" text-anchor="middle" class="tick">{rank}</text>')
    for index, row in enumerate(rows):
        points = " ".join(f"{left + rank * (right - left) / 9:.1f},"
                          f"{bottom - value * (bottom - top):.1f}"
                          for rank, value in enumerate(row["datasets"][dataset]["cmc"]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[index % len(COLORS)]}" stroke-width="3"/>')
    parts.append(f'<text x="{(left + right) / 2}" y="{height - 10}" text-anchor="middle" class="tick">Rank</text></svg>')
    return "".join(parts)


def _mask_svg(rows):
    present = [row for row in rows if row["voc_miou"] is not None]
    width, height = 960, 40 + 42 * len(present)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="VOC mean IoU comparison">']
    for index, row in enumerate(present):
        y = 12 + 42 * index
        value = row["voc_miou"] * 100
        parts.append(f'<text x="8" y="{y + 15}" class="label">{html.escape(row["name"])}</text>')
        parts.append(f'<rect x="250" y="{y}" width="{value * 6:.1f}" height="20" rx="3" fill="{COLORS[index % len(COLORS)]}"/>')
        parts.append(f'<text x="{257 + value * 6:.1f}" y="{y + 15}" class="value">{value:.1f}%</text>')
    parts.append('</svg>')
    return "".join(parts)


def _speed_svg(rows, dataset):
    width, height = 960, 340
    left, right, top, bottom = 72, 935, 24, 280
    max_latency = max(35, math.ceil(max(row["latency_ms"] for row in rows) / 10) * 10)
    max_rank1 = max(.4, math.ceil(max(row["datasets"][dataset]["rank1"] for row in rows) * 10) / 10)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="CPU speed versus Rank-1 on {html.escape(dataset)}">']
    for step in range(5):
        x = left + step * (right - left) / 4
        y = bottom - step * (bottom - top) / 4
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#e2e8ee"/>')
        parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#e2e8ee"/>')
        parts.append(f'<text x="{x}" y="{bottom + 20}" text-anchor="middle" class="tick">{step * max_latency / 4:.0f}</text>')
        parts.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" class="tick">{step * max_rank1 * 25:.0f}%</text>')
    for index, row in enumerate(rows):
        x = left + row["latency_ms"] / max_latency * (right - left)
        y = bottom - row["datasets"][dataset]["rank1"] / max_rank1 * (bottom - top)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{COLORS[index % len(COLORS)]}" stroke="white" stroke-width="2"/>')
        parts.append(f'<text x="{x + 11:.1f}" y="{y - 8 + (index % 3) * 13:.1f}" class="value">{html.escape(row["name"])}</text>')
    parts.append(f'<text x="{(left + right) / 2}" y="{height - 10}" text-anchor="middle" class="tick">CPU milliseconds per 224×224 crop (lower is faster)</text></svg>')
    return "".join(parts)


def _model_type(name):
    if "ImageNet features" in name:
        return "Frozen ImageNet features"
    if "(global pool)" in name:
        return "Same-weights pooling ablation"
    if "matched" in name:
        return "Matched training experiment"
    return "Earlier joint checkpoint"


def _leaderboard_html(rows, dataset):
    ranked = sorted(rows, key=lambda row: (-row["datasets"][dataset]["rank1"],
                                          -row["datasets"][dataset]["map"]))
    body = []
    for rank, row in enumerate(ranked, 1):
        result = row["datasets"][dataset]
        cells = (str(rank), html.escape(row["name"]), _model_type(row["name"]),
                 f'{result["rank1"]:.1%}', f'{result["map"]:.1%}',
                 f'{result["rank5"]:.1%}', f'{row["params"] / 1e6:.2f} M',
                 f'{row["latency_ms"]:.1f} ms')
        body.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    return ('<div class="table-scroll"><table><thead><tr><th>#</th><th>Model</th><th>Comparison type</th>'
            '<th>Rank-1 ↓</th><th>mAP</th><th>Rank-5</th><th>Parameters</th><th>CPU/crop</th>'
            '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


def _report(rows, datasets, device):
    first = rows[0]
    legend = "".join(f'<span><i style="background:{COLORS[i % len(COLORS)]}"></i>{html.escape(row["name"])}</span>'
                     for i, row in enumerate(rows))
    sections = []
    for name, (queries, gallery) in datasets.items():
        sections.append(f'<section><h2>{html.escape(name)}</h2><p>{len(queries)} query crops, {len(gallery)} gallery crops. '
                        'Same-camera gallery entries are excluded for each query.</p>'
                        '<h3>Local leaderboard: ranked by Rank-1</h3>'
                        f'{_leaderboard_html(rows, name)}'
                        '<p>Solid bars: Rank-1. Pale bars: mAP.</p>'
                        f'{_bar_svg(rows, name)}<h3>CMC: chance the correct identity appears by rank k</h3>'
                        f'{_cmc_svg(rows, name)}</section>')
    first_dataset = next(iter(datasets))
    table_rows = []
    for row in rows:
        cells = [html.escape(row["name"]), f'{row["params"] / 1e6:.2f} M',
                 f'{row["latency_ms"]:.1f} ms',
                 f'{row["voc_miou"] * 100:.1f}%' if row["voc_miou"] is not None else 'n/a']
        table_rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    pair_tables = []
    for row in rows:
        cells = []
        for pair, metrics in row["bucktales_pairs"].items():
            cells.append('<tr>' + ''.join(f'<td>{value}</td>' for value in
                (html.escape(pair), str(metrics["queries"]),
                 f'{metrics["rank1"]:.1%}', f'{metrics["rank5"]:.1%}',
                 f'{metrics["map"]:.1%}')) + '</tr>')
        pair_tables.append(f'<details><summary>{html.escape(row["name"])}</summary>'
                           '<table><thead><tr><th>Pair</th><th>Valid queries</th>'
                           '<th>Rank-1</th><th>Rank-5</th><th>mAP</th></tr></thead>'
                           f'<tbody>{"".join(cells)}</tbody></table></details>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReID model benchmark</title><style>
body{{font:16px system-ui,sans-serif;color:#1d2a35;background:#f5f7f9;margin:0;padding:28px;line-height:1.45}}
main{{max-width:1050px;margin:auto}}section{{background:white;border:1px solid #dce3e9;border-radius:12px;padding:24px;margin:22px 0}}
h1,h2,h3{{line-height:1.2}}p{{max-width:850px}}svg{{width:100%;height:auto;display:block}}.tick{{fill:#586976;font:13px system-ui,sans-serif}}
.label{{fill:#233540;font:13px system-ui,sans-serif}}.value{{fill:#233540;font:12px system-ui,sans-serif}}
.legend{{display:flex;flex-wrap:wrap;gap:10px 20px}}.legend span{{display:flex;align-items:center;gap:6px}}.legend i{{display:inline-block;width:12px;height:12px;border-radius:3px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce3e9}}small{{color:#586976}}
.table-scroll{{overflow-x:auto}}.table-scroll table{{min-width:900px}}.table-scroll tbody tr:first-child{{background:#e9f7f2}}
details{{margin:12px 0}}summary{{cursor:pointer;font-weight:600}}
</style></head><body><main><h1>ReID model comparison</h1>
<p>All curves here come from the same local test crops and cosine similarity. Rank-1, Rank-5, Rank-10, mAP, and CMC follow cross-camera ReID evaluation. The ImageNet feature baselines were trained for classification, not ReID; they provide architecture references rather than training-matched competitors. “Global pool” uses identical joint weights with mask guidance switched off.</p>
<div class="legend">{legend}</div>{''.join(sections)}
<section><h2>BuckTales by drone pair</h2><p>Each query searches only the other drone in its own pair. The aggregate graphs pool valid queries across pairs, so larger pairs carry more weight. Pair identities are never mixed between galleries.</p>{''.join(pair_tables)}</section>
<section><h2>Accuracy versus CPU inference time</h2><p>Each dot is one model. Higher Rank-1 and lower per-crop latency are better; these PC timings are not Raspberry Pi or IMX500 measurements.</p>{_speed_svg(rows, first_dataset)}</section>
<section><h2>VOC segmentation</h2><p>21-class mean IoU on VOC 2012 validation at the model's 28×28 mask output. This does not measure mask quality on aerial animals or vehicles.</p>{_mask_svg(rows)}</section>
<section><h2>Size and inference time</h2><table><thead><tr><th>Model</th><th>Active parameters</th><th>CPU time per 224×224 crop</th><th>VOC mIoU</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p><small>Latency is the median of 25 warm-cache, batch-one PyTorch CPU runs with one thread on {html.escape(platform.platform())}. It excludes image decoding, object detection, tracking, and camera transfer. Evaluation embeddings used {html.escape(str(device))}. Parameter counts exclude the unused ShuffleNet classification layer.</small></p></section>
<section><h2>How to read this</h2><p>BuckTales contains {len(rows[0]['bucktales_pairs'])} prepared drone pairs and is reserved for test-only, zero-shot evaluation. Some pairs reuse the same video, and animal identities are not verified across recording days. This report has already been inspected during development, so later scores on it are exploratory rather than a blind final test. AICity is a vehicle cross-camera test. Neither is a standard person ReID leaderboard. Published scores from Market-1501 or other datasets should not be placed on these graphs as if they used the same images or protocol.</p>
<p>The <a href="https://papers.neurips.cc/paper_files/paper/2024/file/95286b5d4cd5b7953bd2bbe717300fe0-Paper-Datasets_and_Benchmarks_Track.pdf">BuckTales paper</a> explicitly provides no standalone ReID baseline. Its published tracking HOTA and IDF1 values include detection and temporal association and cannot be ranked against crop retrieval Rank-1. The <a href="https://eval.aicitychallenge.org/aicity2022/submission/leaderboard">official AI City Track 1 leaderboard</a> also ranks full multi-camera tracking by IDF1, whereas this report evaluates a local crop retrieval split. Source for standard ReID metrics: <a href="https://openaccess.thecvf.com/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf">Market-1501 benchmark paper</a>.</p></section>
</main></body></html>'''
