"""Render readable, static HTML scorecards from the vehicle benchmark CSVs."""

import csv
from html import escape
from pathlib import Path


METRICS = Path(__file__).resolve().parents[2] / "metrics"
JOINT = "Vehicle warm-start joint"
CONTROL = "Vehicle warm-start ReID only"

STYLE = """
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
color:#17253a;background:#f4f7fb;font-size:16px}*{box-sizing:border-box}body{margin:0}
a{color:#155db2}a:hover{text-decoration:none}.wrap{max-width:1160px;margin:auto;padding:28px 24px 70px}
header{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:34px}
.brand{font-size:.9rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#31577f}
nav{display:flex;gap:8px}nav a{padding:9px 14px;border-radius:9px;text-decoration:none;font-weight:650}
nav a.active{background:#113b66;color:white}nav a:not(.active){background:white;color:#244766}
h1{font-size:clamp(2rem,4vw,3.1rem);line-height:1.08;letter-spacing:-.035em;margin:0 0 12px}
h2{font-size:1.35rem;margin:0 0 16px;letter-spacing:-.02em}h3{margin:0 0 8px;font-size:1.07rem}
p{line-height:1.55;margin:0 0 14px}.lead{font-size:1.12rem;color:#43566c;max-width:780px}
.hero{margin-bottom:34px}.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-weight:800;font-size:.75rem;color:#4c6991;margin:0 0 11px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.card,.panel{background:white;border:1px solid #dfe7f0;border-radius:16px;box-shadow:0 5px 22px #173d6010}
.card{padding:22px}.panel{padding:26px;margin:18px 0 24px}.muted{color:#5e6d80}.metric{font-size:2rem;font-weight:800;letter-spacing:-.04em;margin:6px 0}
.metric small{font-size:.53em;letter-spacing:0}.delta{font-weight:800;font-size:1.03rem}.bad{color:#b3333e}.good{color:#087c59}
.pill{display:inline-block;padding:5px 9px;border-radius:999px;font-size:.77rem;font-weight:800;background:#e5eef9;color:#214e80}
.pill.warn{background:#fff1d8;color:#82500b}.pill.good{background:#dff5ea;color:#12694a}
.subhead{margin:20px 0 10px;font-size:.91rem;font-weight:800;color:#43566c;text-transform:uppercase;letter-spacing:.055em}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:660px}th{text-align:left;color:#52657b;font-size:.79rem;letter-spacing:.035em;text-transform:uppercase}
th,td{padding:11px 12px;border-bottom:1px solid #e9edf3}tr:last-child td{border-bottom:0}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.ours{background:#edf7ff}tr.control{background:#f6f8fb}.name{font-weight:650}.track{background:#e8eef5;height:8px;width:110px;border-radius:99px;overflow:hidden;display:inline-block;vertical-align:middle}
.fill{height:100%;background:#3a89d7;border-radius:99px}.ours .fill{background:#087caa}.note{border-left:4px solid #6d94bd;padding:11px 14px;background:#f2f7fc;border-radius:0 8px 8px 0;color:#43566c;margin-top:16px}
.steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.step{border:1px solid #dfe7f0;border-radius:12px;padding:17px;background:#fbfdff}
.step b{color:#113b66}.step p{font-size:.94rem;margin:8px 0 0}.big-target{font-size:1.35rem;font-weight:800;color:#113b66}
.footer{font-size:.9rem;color:#607186;margin-top:35px}.compact{font-size:.94rem}.nowrap{white-space:nowrap}
@media(max-width:760px){.grid,.grid.three,.steps{grid-template-columns:1fr}.wrap{padding:20px 15px 50px}.panel{padding:18px}.metric{font-size:1.8rem}}
"""


def _rows(name):
    with (METRICS / name).open(newline="") as file:
        return list(csv.DictReader(file))


def _value(row, key):
    return float(row[key])


def _percent(value):
    return f"{100 * value:.2f}%"


def _html_page(title, active, content):
    navigation = "".join(
        f'<a class="{"active" if active == label else ""}" href="{filename}">{label}</a>'
        for label, filename in (("Overview", "index.html"), ("Full rankings", "rankings.html"))
    )
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{escape(title)} · Vehicle ReID metrics</title><style>{STYLE}</style></head>"
            f"<body><div class=\"wrap\"><header><div class=\"brand\">Capstone / vehicle ReID</div>"
            f"<nav aria-label=\"Metrics pages\">{navigation}</nav></header><main>{content}</main>"
            "<footer class=\"footer\">Generated from the CSV files in this folder. "
            "Run <code>python benchmark.py metrics</code> to refresh these pages.</footer></div></body></html>")


def _summary_card(label, rows, detail):
    by_name = {row["model"]: row for row in rows}
    joint, control = by_name.get(JOINT), by_name.get(CONTROL)
    if not joint or not control:
        return f'<div class="card"><h3>{escape(label)}</h3><p class="muted">Matched checkpoints not yet measured.</p></div>'
    gap = 100 * (_value(joint, "map") - _value(control, "map"))
    rank_gap = 100 * (_value(joint, "rank1") - _value(control, "rank1"))
    return (f'<div class="card"><div class="eyebrow">{escape(label)}</div>'
            f'<div class="metric">{_percent(_value(joint, "map"))} <small>mAP</small></div>'
            f'<p>Joint model · Rank-1 {_percent(_value(joint, "rank1"))}</p>'
            f'<div class="delta {"good" if gap > 0 else "bad"}">{gap:+.2f} points mAP vs ReID-only</div>'
            f'<p class="muted compact">Rank-1 gap: {rank_gap:+.2f} points. '
            f'Control: {_percent(_value(control, "map"))} mAP / '
            f'{_percent(_value(control, "rank1"))} Rank-1.</p>'
            f'<p class="muted compact">{escape(detail)}</p></div>')


def _target_table(targets, vru, vrai):
    body = ""
    for row in targets:
        results = vru if row["dataset"].startswith("VRU") else vrai
        live_joint = next((item for item in results if item["model"] == JOINT), None)
        metric_key = "rank1" if row["metric"] == "Rank-1" else "map"
        current = (_percent(_value(live_joint, metric_key)) if live_joint
                   else f'{row["current_joint_percent"]}%')
        body += ("<tr>"
                 f'<td class="name">{escape(row["dataset"])}</td>'
                 f'<td>{escape(row["metric"])}</td>'
                 f'<td class="num"><b>&gt; {escape(row["threshold_percent"])}%</b></td>'
                 f'<td class="num">{current}</td>'
                 "</tr>")
    return ("<div class=\"table-scroll\"><table><thead><tr><th>Test</th><th>Metric</th>"
            "<th class=\"num\">Beat this control</th><th class=\"num\">Current joint</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def _paper_table(papers):
    body = ""
    for row in papers:
        link = escape(row["source"], quote=True)
        body += (f'<tr><td class="name"><a href="{link}">{escape(row["paper"])}</a></td>'
                 f'<td>{escape(row["dataset"])}</td>'
                 f'<td class="num">{escape(row["rank1_threshold_percent"])}%</td>'
                 f'<td class="num">{escape(row["map_threshold_percent"])}%</td></tr>')
    return ("<div class=\"table-scroll\"><table><thead><tr><th>Published method</th>"
            "<th>Official test</th><th class=\"num\">Rank-1</th><th class=\"num\">mAP</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def _ranking_table(rows):
    ranked = sorted(rows, key=lambda row: _value(row, "map"), reverse=True)
    body = ""
    for place, row in enumerate(ranked, 1):
        name = row["model"]
        row_class = "ours" if name == JOINT else "control" if name == CONTROL else ""
        width = min(100, max(0, 100 * _value(row, "map")))
        parameters = f'{int(row["parameters"])/1_000_000:.2f} M' if row["parameters"] else "—"
        latency = f'{float(row["cpu_ms"]):.1f}' if row["cpu_ms"] else "—"
        body += (f'<tr class="{row_class}"><td class="num">{place}</td>'
                 f'<td class="name">{escape(name)}</td>'
                 f'<td class="num">{_percent(_value(row, "rank1"))}</td>'
                 f'<td class="num"><b>{_percent(_value(row, "map"))}</b> '
                 f'<span class="track" aria-hidden="true"><span class="fill" style="display:block;width:{width:.1f}%"></span></span></td>'
                 f'<td class="num">{parameters}</td><td class="num">{latency}</td></tr>')
    return ("<div class=\"table-scroll\"><table><thead><tr>"
            "<th class=\"num\">#</th><th>Model</th><th class=\"num\">Rank-1</th>"
            "<th class=\"num\">mAP</th><th class=\"num\">Parameters</th>"
            f"<th class=\"num\">CPU ms/crop</th></tr></thead><tbody>{body}</tbody></table></div>")


def render():
    """Regenerate both HTML pages after a benchmark, without rerunning a model."""
    vru = _rows("vru_benchmark.csv")
    vrai = _rows("vrai_local_benchmark.csv")
    targets = _rows("reid_targets.csv")
    papers = _rows("published_targets.csv")

    overview = (
        '<section class="hero"><p class="eyebrow">Current state and next goals</p>'
        '<h1>Crop-model ReID scorecard</h1>'
        '<p class="lead">The first question: does the mask improve vehicle matching enough '
        'to justify its extra compute? Start by beating the matched ReID-only ShuffleNetV2.</p>'
        '<div class="note"><b>Scope:</b> these models receive an already cropped vehicle. '
        'They do not detect cars or produce separate masks and embeddings from a full frame. '
        'The <a href="../FULL_FRAME_MOTS.md">new full-frame MOTS target</a> adds those tasks.</div></section>'
        '<div class="grid">'
        + _summary_card("VRU · aerial retrieval", vru, "One gallery image per identity; no verified camera IDs.")
        + _summary_card("VRAI · local cross-camera", vrai,
                        "Custom identity-disjoint split of official training images, not the official test.")
        + '</div><section class="panel"><h2>First thresholds to beat</h2>'
        '<p class="muted">Both Rank-1 and mAP should exceed the matched ReID-only model on the same split. '
        'Higher is better. These are local project thresholds, not paper leaderboard scores.</p>'
        + _target_table(targets, vru, vrai)
        + '</section><section class="panel"><p class="eyebrow">Suggested research bar</p>'
        '<h2>What would make a strong thesis result?</h2>'
        '<p>There is no universal publication cutoff. These are <b>proposed ReID-branch goals</b> '
        'for the new full-frame model. A complete result also needs detection AP, instance-mask AP, '
        'tracking association metrics, and end-to-end frame latency. Predeclare a fresh identity '
        'holdout before the final experiment.</p>'
        '<div class="steps"><div class="step"><b>1 · Retrieval benefit</b>'
        '<p><span class="big-target">+2 points mAP</span> over matched ReID-only on a fresh '
        'cross-camera holdout. Repeat with at least 3 training seeds and report a paired '
        'confidence interval for the difference.</p></div>'
        '<div class="step"><b>2 · Real masks</b>'
        '<p><span class="big-target">≥70% foreground IoU</span> on at least 200 aerial vehicle '
        'crops from unseen scenes. Report small vehicles separately. This is a project goal, '
        'not an official benchmark record.</p></div>'
        '<div class="step"><b>3 · Efficiency</b>'
        '<p><span class="big-target">Faster than two models</span> at comparable retrieval accuracy: '
        'measure joint inference against a separate segmenter + ReID pipeline on the same '
        'hardware and input size. Count detection and tracking for any full-system claim.</p></div></div>'
        '<div class="note">The local VRU and VRAI results have already been inspected during '
        'development. They are useful for improvement, but a final claim needs untouched identities. '
        'VRU mAP equals mean reciprocal rank because the gallery has one positive per identity.</div>'
        '</section><section class="panel"><h2>Published aerial targets</h2>'
        '<p class="muted">Stretch goals on each paper’s exact official protocol. '
        'Do not compare local split percentages directly with these numbers.</p>'
        + _paper_table(papers)
        + '</section><section class="panel"><h2>What to compare next</h2>'
        '<p>Train OSNet and TransReID on the same vehicle IDs. Compare them with both '
        'ShuffleNetV2 arms and a separate segmenter + ReID pipeline on one fixed query/gallery '
        'split. <a href="rankings.html">See all current model rankings →</a></p>'
        '<p class="compact muted">Architecture references: '
        '<a href="https://ksp.etri.re.kr/ksp/article/file/67007.pdf">MG-MOTS (lightweight mask-guided ReID)</a> '
        'and <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Porzi_Learning_Multi-Object_Tracking_and_Segmentation_From_Automatic_Annotations_CVPR_2020_paper.pdf">MOTSNet (mask pooling)</a>. '
        'Their same-video tracking scores use a different task protocol.</p></section>'
    )
    rankings = (
        '<section class="hero"><p class="eyebrow">Measured local results</p>'
        '<h1>Full model rankings</h1><p class="lead">Each table is sorted by mAP. '
        'Blue highlights the current joint model; gray highlights its matched ReID-only '
        'control. Scores from the two tables are separate protocols.</p></section>'
        '<section class="panel"><h2>VRU · aerial vehicle retrieval</h2>'
        '<p class="muted">Fixed local gallery: 8,000 identities; 83,595 queries. '
        'One positive per query, so mAP is mean reciprocal rank. No verified cross-drone IDs.</p>'
        + _ranking_table(vru)
        + '<div class="note">MSINet uses the authors’ trained checkpoint and different '
        'preprocessing. Its CPU time is measured locally but is not a matched training-budget '
        'comparison. These numbers include per-crop model inference, not video decoding, '
        'detection, or tracking.</div></section>'
        '<section class="panel"><h2>VRAI · local cross-camera retrieval</h2>'
        '<p class="muted">1,223 queries and 2,199 gallery images from a custom, '
        'identity-disjoint split of VRAI training images. This is not the official VRAI test.</p>'
        + _ranking_table(vrai)
        + '<div class="note">The warm-start joint model improves over its own global-pooling '
        'ablation, but the matched ReID-only model remains ahead.</div></section>'
        '<section class="panel"><h2>Read these rankings correctly</h2>'
        '<p>Rank-1 is the fraction of queries whose first match has the right vehicle ID. '
        'mAP rewards placing all correct gallery images early in the ranked list. '
        'Parameters are active inference parameters. CPU latency is milliseconds per '
        'crop on the machine used for these CSVs; it is not Raspberry Pi AI Camera latency.</p>'
        '<p><a href="index.html">← Back to thresholds and thesis goal</a></p></section>'
    )
    for filename, title, active, body in (
        ("index.html", "Overview", "Overview", overview),
        ("rankings.html", "Full rankings", "Full rankings", rankings),
    ):
        path = METRICS / filename
        path.write_text(_html_page(title, active, body))
        print(f"Saved {path}")


if __name__ == "__main__":
    render()
