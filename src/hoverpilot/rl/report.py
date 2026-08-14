"""Generate a self-contained HTML report from training artifacts."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .checkpoints import load_policy_checkpoint

REWARD_TAGS = (
    "train/episode_reward",
    "train/avg_reward",
    "eval/avg_reward",
)
ERROR_TAGS = (
    "train/state/abs_inclination_error_deg",
    "train/state/abs_roll_error_deg",
    "train/state/abs_rudder_angle_error_deg",
    "train/state/altitude_error_m",
    "train/state/longitudinal_error_m",
    "train/state/radial_distance_m",
    "eval/attitude_error_deg",
    "eval/position_error_m",
    "eval/altitude_error_m",
)


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Report generation requires TensorBoard; install the RL extra."
        ) from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [(event.step, float(event.value)) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def _polyline(points: Sequence[tuple[int, float]], width: int, height: int) -> str:
    finite = [(float(x), float(y)) for x, y in points if math.isfinite(y)]
    if not finite:
        return ""
    x_values = [point[0] for point in finite]
    y_values = [point[1] for point in finite]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0e-9, y_max - y_min)
    coordinates = " ".join(
        f"{20 + (x - x_min) / x_span * (width - 40):.1f},"
        f"{10 + (y_max - y) / y_span * (height - 30):.1f}"
        for x, y in finite
    )
    return f'<polyline points="{coordinates}" fill="none" stroke="currentColor" stroke-width="2" />'


def _line_chart(title: str, series: Mapping[str, Sequence[tuple[int, float]]]) -> str:
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2")
    visible = [(name, points) for name, points in series.items() if points]
    if not visible:
        return ""
    lines = []
    legend = []
    for index, (name, points) in enumerate(visible):
        color = colors[index % len(colors)]
        lines.append(f'<g style="color:{color}">{_polyline(points, 760, 240)}</g>')
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(name)}</span>'
        )
    return (
        '<section class="chart"><h2>'
        + html.escape(title)
        + '</h2><svg viewBox="0 0 760 240" role="img">'
        + '<path d="M20 10V220H740" class="axis" />'
        + "".join(lines)
        + '</svg><div class="legend">'
        + "".join(legend)
        + "</div></section>"
    )


def _trajectory_chart(scalars: Mapping[str, Sequence[tuple[int, float]]]) -> str:
    x_points = scalars.get("train/state/x_m", ())
    y_points = scalars.get("train/state/y_m", ())
    y_by_step = dict(y_points)
    points = [(x, y_by_step[step]) for step, x in x_points if step in y_by_step]
    if not points:
        return ""
    xs, ys = zip(*points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(1.0e-9, x_max - x_min)
    y_span = max(1.0e-9, y_max - y_min)
    coordinates = " ".join(
        f"{20 + (x - x_min) / x_span * 720:.1f},"
        f"{10 + (y_max - y) / y_span * 210:.1f}"
        for x, y in points
    )
    return (
        '<section class="chart"><h2>Flight trajectory (X/Y)</h2>'
        '<svg viewBox="0 0 760 240" role="img"><path d="M20 10V220H740" class="axis" />'
        f'<polyline points="{coordinates}" fill="none" stroke="#7c3aed" stroke-width="2" />'
        '</svg></section>'
    )


def _latest_evaluation(checkpoint) -> Mapping[str, object]:
    return checkpoint.evaluation_history[-1] if checkpoint.evaluation_history else {}


def _termination_counts(checkpoint, scalars) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evaluation in checkpoint.evaluation_history if checkpoint else ():
        reasons = evaluation.get("termination_counts", {})
        if isinstance(reasons, Mapping):
            for reason, count in reasons.items():
                counts[str(reason)] = counts.get(str(reason), 0) + int(count)
    if counts:
        return counts
    for tag, points in scalars.items():
        prefix = "eval/termination/"
        if tag.startswith(prefix) and points:
            counts[tag.removeprefix(prefix)] = int(points[-1][1])
    return counts


def _termination_table(counts: Mapping[str, int]) -> str:
    if not counts:
        return ""
    total = max(1, sum(counts.values()))
    rows = "".join(
        f"<tr><td>{html.escape(reason)}</td><td>{count}</td><td>{count / total:.1%}</td></tr>"
        for reason, count in sorted(counts.items(), key=lambda item: -item[1])
    )
    return (
        '<section><h2>Termination reasons</h2><table><thead><tr>'
        '<th>Reason</th><th>Count</th><th>Rate</th></tr></thead><tbody>'
        + rows
        + "</tbody></table></section>"
    )


def _comparison_table(current, baseline) -> str:
    if not current or not baseline:
        return ""
    current_eval = _latest_evaluation(current)
    baseline_eval = _latest_evaluation(baseline)
    metrics = (
        ("Mean reward", "mean_reward", True),
        ("Survival time (s)", "mean_survival_time_s", True),
        ("Position error (m)", "mean_position_error_m", False),
        ("Altitude error (m)", "mean_altitude_error_m", False),
        ("Attitude error (deg)", "mean_attitude_error_deg", False),
    )
    rows = []
    for label, key, higher_is_better in metrics:
        if key not in current_eval or key not in baseline_eval:
            continue
        current_value = float(current_eval[key])
        baseline_value = float(baseline_eval[key])
        delta = current_value - baseline_value
        improved = delta > 0 if higher_is_better else delta < 0
        rows.append(
            f'<tr><td>{label}</td><td>{baseline_value:.3f}</td>'
            f'<td>{current_value:.3f}</td><td class="{"good" if improved else "bad"}">{delta:+.3f}</td></tr>'
        )
    if not rows:
        return ""
    return (
        '<section><h2>Checkpoint comparison</h2><table><thead><tr>'
        '<th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _discover_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("*.pt"), key=lambda path: path.stat().st_mtime)
    best = [path for path in candidates if ".best." in path.name]
    return (best or candidates)[-1] if candidates else None


def generate_training_report(
    run_dir: str,
    *,
    checkpoint_path: str | None = None,
    compare_to: str | None = None,
    output_path: str | None = None,
    video_path: str | None = None,
) -> Path:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise ValueError(f"Training run directory does not exist: {run_path}")
    scalars = _load_scalars(run_path)
    resolved_checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path
        else _discover_checkpoint(run_path)
    )
    checkpoint = (
        load_policy_checkpoint(str(resolved_checkpoint)) if resolved_checkpoint else None
    )
    baseline = load_policy_checkpoint(compare_to) if compare_to else None
    output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else run_path / "report.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    latest = _latest_evaluation(checkpoint) if checkpoint else {}
    summary_items = {
        "Run": run_path.name,
        "Checkpoint": str(resolved_checkpoint) if resolved_checkpoint else "not found",
        "Training step": checkpoint.training_step if checkpoint else "unknown",
        "Best mean reward": checkpoint.best_mean_reward if checkpoint else "unknown",
        "Latest mean reward": latest.get("mean_reward", "unknown"),
        "Latest survival time": latest.get("mean_survival_time_s", "unknown"),
    }
    cards = "".join(
        f'<div class="card"><strong>{html.escape(str(label))}</strong><span>{html.escape(str(value))}</span></div>'
        for label, value in summary_items.items()
    )
    reward_chart = _line_chart(
        "Reward trend", {tag: scalars.get(tag, ()) for tag in REWARD_TAGS}
    )
    error_chart = _line_chart(
        "Control errors", {tag: scalars.get(tag, ()) for tag in ERROR_TAGS}
    )
    trajectory = _trajectory_chart(scalars)
    terminations = _termination_table(_termination_counts(checkpoint, scalars))
    comparison = _comparison_table(checkpoint, baseline)
    metadata = checkpoint.experiment_metadata if checkpoint else {}
    configuration = html.escape(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    video = ""
    if video_path:
        video_source = html.escape(str(Path(video_path).expanduser().resolve()))
        video = f'<section><h2>Evaluation video</h2><video controls src="{video_source}"></video></section>'

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>HoverPilot training report</title><style>
body{{font:15px system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:32px;color:#172033;background:#f5f7fb}}
h1,h2{{color:#111827}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card,section{{background:white;border:1px solid #dbe2ea;border-radius:10px;padding:18px;margin:16px 0}}
.card{{display:flex;flex-direction:column;gap:8px;margin:0}}.card span{{font-size:1.1rem;overflow-wrap:anywhere}}
svg{{width:100%;height:auto;background:#fbfdff}}.axis{{fill:none;stroke:#94a3b8;stroke-width:1}}
.legend{{display:flex;gap:16px;flex-wrap:wrap}}.legend i{{display:inline-block;width:10px;height:10px;margin-right:5px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #e5e7eb}}
.good{{color:#047857}}.bad{{color:#b91c1c}}pre{{max-height:500px;overflow:auto;background:#111827;color:#e5e7eb;padding:16px;border-radius:8px}}
video{{max-width:100%}}</style></head><body>
<h1>HoverPilot training report</h1><div class="cards">{cards}</div>
{reward_chart}{error_chart}{trajectory}{terminations}{comparison}{video}
<section><h2>Reproducibility metadata</h2><pre>{configuration}</pre></section>
</body></html>"""
    output.write_text(document, encoding="utf-8")
    return output
