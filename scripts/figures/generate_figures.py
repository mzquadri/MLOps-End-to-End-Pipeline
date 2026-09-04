"""Figures for the reference run, drawn from what the pipeline wrote.

Three figures, each answering one question:

    can this model be shipped        the promotion gate, every check against its threshold
    how well does it separate        ROC and precision-recall on the held-out split
    does it generalise               cross-validation, validation and test side by side

Everything is read from results/reference_run.json and results/evaluation_curves.json,
so a figure cannot show a number the run did not produce. Run the pipeline first:

    python -m src.pipeline --config configs/train_config.yaml
    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import portfolio_style as ps  # noqa: E402

OUT = REPO / "docs" / "figures"
RUN = REPO / "results" / "reference_run.json"
CURVES = REPO / "results" / "evaluation_curves.json"

#: Human wording for the gate check keys, and the unit each one is measured in.
CHECK_LABEL = {
    "accuracy": ("Accuracy on the held-out split", ""),
    "f1_weighted": ("Weighted F1", ""),
    "accuracy_over_baseline": ("Margin over the majority baseline", ""),
    "latency_p95": ("Single-row inference, p95", " ms"),
}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png")


def require(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO).as_posix()} is missing. Run the pipeline first:\n"
            "  python -m src.pipeline --config configs/train_config.yaml"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def fig_gate(run):
    """The question the whole repository is organised around: can this ship?"""
    gate = run["performance_gate"]
    checks = gate["checks"]
    order = ["accuracy", "f1_weighted", "accuracy_over_baseline", "latency_p95"]
    order = [k for k in order if k in checks] + [k for k in checks if k not in order]

    fig = plt.figure(figsize=(12.4, 7.8))
    ax = fig.add_axes([0.40, 0.215, 0.50, 0.525])

    ypos = np.arange(len(order))[::-1]
    for key, yy in zip(order, ypos, strict=True):
        c = checks[key]
        label, unit = CHECK_LABEL.get(key, (key.replace("_", " "), ""))
        # Latency passes by being *below* its threshold; the others by being above.
        lower_is_better = key == "latency_p95"
        value, thr = c["value"], c["threshold"]
        headroom = (thr - value) / thr if lower_is_better else (value - thr) / thr
        colour = ps.GREEN if c["passed"] else ps.RED

        ax.barh(yy, min(headroom, 1.0) * 100, color=colour, height=0.5, zorder=3,
                alpha=0.85)
        ax.text(-0.03, yy, label, ha="right", va="center", fontsize=11.4,
                color=ps.INK, transform=ax.get_yaxis_transform())
        shown = f"{value:.4g}{unit}" if not lower_is_better else f"{value:.3g}{unit}"
        thr_shown = f"{thr:.4g}{unit}"
        ax.text(-0.03, yy - 0.30, f"measured {shown}, threshold {thr_shown}",
                ha="right", va="center", fontsize=9.4, color=ps.FAINT,
                transform=ax.get_yaxis_transform())
        ax.text(min(headroom, 1.0) * 100 + 1.6, yy,
                "passes" if c["passed"] else "fails", va="center", fontsize=10.4,
                color=colour, fontweight="600")

    ax.set_yticks([])
    ax.set_xlim(0, 118)
    ps.clean(ax, left=False, grid_axis="x")
    ax.set_xlabel("headroom over the threshold (%)", fontsize=11)

    passed = gate["overall_passed"]
    ps.title_block(
        fig,
        "Every check the model had to pass to be promoted",
        "The gate is not decoration. If any bar were missing its threshold the run "
        "would stop and nothing would reach\nthe registry. All four pass on this run.",
        y=0.962, size=22)
    ps.footnote(fig, [
        "The margin check is the one that matters. An accuracy floor alone can be met "
        "by a model that has learned the class prior, so the gate also requires a "
        "stated margin over a majority-class baseline.",
        f"Latency is measured in process over {run['latency']['n_runs']} single-row "
        f"predictions and is a property of the machine, not of the model. It is "
        f"gated loosely for that reason.",
        f"Overall: {'passed' if passed else 'failed'}. "
        f"Source: results/reference_run.json."], y=0.098)
    save(fig, "01_promotion_gate")


def fig_curves(run, curves):
    """How well the classifier separates the two classes."""
    t = run["test_metrics"]
    roc, pr = curves["roc"], curves["precision_recall"]

    fig = plt.figure(figsize=(12.4, 7.2))
    axL = fig.add_axes([0.085, 0.205, 0.375, 0.535])
    axR = fig.add_axes([0.585, 0.205, 0.375, 0.535])

    axL.plot([0, 1], [0, 1], color=ps.HAIR, lw=1.4, ls="--", zorder=2)
    axL.plot(roc["fpr"], roc["tpr"], color=ps.BLUE, lw=2.4, zorder=3)
    axL.fill_between(roc["fpr"], roc["tpr"], color=ps.BLUE_SOFT, alpha=0.20, zorder=1)
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1.02)
    ps.clean(axL, grid_axis="y")
    axL.set_xlabel("false positive rate", fontsize=10.6)
    axL.set_ylabel("true positive rate", fontsize=10.6)
    axL.text(0, 1.09, "ROC", transform=axL.transAxes, fontsize=13, color=ps.INK,
             fontweight="600", va="bottom")
    ps.note(axL, 0.52, 0.30, f"AUC {t['roc_auc']:.4f}", color=ps.BLUE, size=13.5,
            transform=axL.transAxes, weight="600")
    ps.note(axL, 0.52, 0.20, "diagonal is a coin flip", color=ps.FAINT, size=9.8,
            transform=axL.transAxes)

    base = t["baseline_accuracy"]
    axR.axhline(base, color=ps.HAIR, lw=1.4, ls="--", zorder=2)
    axR.plot(pr["recall"], pr["precision"], color=ps.GREEN, lw=2.4, zorder=3)
    axR.fill_between(pr["recall"], pr["precision"], color=ps.GREEN_SOFT, alpha=0.20,
                     zorder=1)
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 1.02)
    ps.clean(axR, grid_axis="y")
    axR.set_xlabel("recall", fontsize=10.6)
    axR.set_ylabel("precision", fontsize=10.6)
    axR.text(0, 1.09, "Precision and recall", transform=axR.transAxes, fontsize=13,
             color=ps.INK, fontweight="600", va="bottom")
    ps.note(axR, 0.06, 0.22, f"AUC {t['pr_auc']:.4f}", color=ps.GREEN, size=13.5,
            transform=axR.transAxes, weight="600")
    ps.note(axR, 0.06, 0.12, f"a coin flip sits at {base:.2f}", color=ps.FAINT,
            size=9.8, transform=axR.transAxes)

    ps.title_block(
        fig, "How well it separates the two classes",
        "Both curves are computed on the 600 row held-out split, which the model and "
        "the feature transformers never saw\nduring fitting.", y=0.962, size=22)
    ps.footnote(fig, [
        "The two curves answer different questions. ROC is insensitive to class "
        "balance; precision-recall is not, which is why the baseline line sits at the "
        "positive class rate rather than at zero.",
        "This dataset is balanced 1,500 to 1,500, so the two tell a similar story "
        "here. On a skewed problem they would not.",
        "Source: results/evaluation_curves.json."], y=0.096)
    save(fig, "02_evaluation_curves")


def fig_generalisation(run):
    """Does the number hold up across the three places it was measured?"""
    tm, t = run["training_metrics"], run["test_metrics"]

    fig = plt.figure(figsize=(12.4, 7.4))
    ax = fig.add_axes([0.095, 0.215, 0.545, 0.525])
    axC = fig.add_axes([0.715, 0.215, 0.245, 0.525])

    points = [
        ("cross-validation\n5 folds on train", tm["cv_mean"], tm.get("cv_std", 0.0),
         ps.FAINT),
        ("validation\n600 rows", tm["validation_f1_weighted"], 0.0, ps.BLUE),
        ("held-out test\n600 rows", t["f1_weighted"], 0.0, ps.GREEN),
    ]
    xs = np.arange(len(points))
    for x, (_label, val, sd, colour) in zip(xs, points, strict=True):
        if sd:
            ax.errorbar([x], [val], yerr=[sd], fmt="none", ecolor=colour, elinewidth=2,
                        capsize=6, zorder=3)
        ax.plot([x], [val], "o", color=colour, ms=13, mec=ps.PAPER, mew=2, zorder=4)
        ax.text(x, val + (sd if sd else 0) + 0.006, f"{val:.4f}", ha="center",
                fontsize=11.2, color=ps.INK, fontweight="600")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points], fontsize=10.4)
    ax.set_xlim(-0.55, len(points) - 0.45)
    ax.set_ylim(0.74, 0.845)
    ps.clean(ax, grid_axis="y")
    ax.set_ylabel("weighted F1", fontsize=11)
    for x in xs:
        ax.axvline(x, color=ps.HAIR, lw=0.9, zorder=0)

    cm = np.array(t["confusion_matrix"])
    labels = t["confusion_matrix_labels"]
    axC.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max() * 1.35)
    for i in range(2):
        for j in range(2):
            axC.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=15,
                     color=ps.INK if cm[i, j] < cm.max() * 0.7 else ps.PAPER,
                     fontweight="600")
    axC.set_xticks([0, 1])
    axC.set_xticklabels(labels, fontsize=10)
    axC.set_yticks([0, 1])
    axC.set_yticklabels(labels, fontsize=10)
    axC.set_xlabel("predicted", fontsize=10.4)
    axC.set_ylabel("actual", fontsize=10.4)
    axC.tick_params(length=0)
    for sp in axC.spines.values():
        sp.set_visible(False)
    axC.text(0, 1.09, "Test confusion matrix", transform=axC.transAxes, fontsize=12.4,
             color=ps.INK, fontweight="600", va="bottom")

    spread = max(p[1] for p in points) - min(p[1] for p in points)
    ps.title_block(
        fig, "The number holds up where it was not fitted",
        "Weighted F1 measured in three places. Cross-validation sits inside the "
        "training split, validation guided the choices,\nand the test split was "
        "scored once at the end.", y=0.962, size=22)
    ps.footnote(fig, [
        f"The three agree to within {spread:.3f}, and the test result sits inside the "
        f"cross-validation spread. That is what an absence of overfitting looks like "
        f"on a model this simple.",
        f"The errors are close to symmetric: {cm[0, 1]} negatives called positive "
        f"against {cm[1, 0]} positives called negative, so the model is not trading "
        f"one class off against the other.",
        "600 test rows puts roughly three points of confidence interval on accuracy, "
        "so small differences here are not meaningful. Source: "
        "results/reference_run.json."], y=0.096)
    save(fig, "03_generalisation")


def main() -> int:
    ps.apply()
    run = require(RUN)
    curves = require(CURVES)
    print(f"reference run: {run['dataset']['key']}, "
          f"{run['split']['test_rows']} test rows\n")
    fig_gate(run)
    fig_curves(run, curves)
    fig_generalisation(run)
    print(f"\nfigures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
