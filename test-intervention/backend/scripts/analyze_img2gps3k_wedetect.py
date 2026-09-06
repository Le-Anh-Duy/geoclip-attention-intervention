"""Summarize and visualize Img2GPS3K WeDetect intervention results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402


OUTCOME_ORDER = ["improved", "worsened", "unchanged"]
OUTCOME_LABELS = {
    "improved": "Tốt hơn",
    "worsened": "Xấu hơn",
    "unchanged": "Không đổi",
}
COLORS = {
    "improved": "#2a9d8f",
    "worsened": "#e76f51",
    "unchanged": "#8d99ae",
}
PROPOSAL_BIN_ORDER = ["0", "1", "2", "3", "4–5", "6+"]


def classify(delta: pd.Series, tolerance: float) -> pd.Series:
    return pd.Series(
        np.select(
            [delta > tolerance, delta < -tolerance],
            ["improved", "worsened"],
            default="unchanged",
        ),
        index=delta.index,
    )


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def safe_correlation(function, x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return {"coefficient": None, "p_value": None}
    result = function(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))
    return {"coefficient": float(result.statistic), "p_value": float(result.pvalue)}


def build_statistics(frame: pd.DataFrame, tolerance: float) -> tuple[dict, pd.DataFrame]:
    frame = frame.copy()
    frame["outcome"] = classify(frame["union_delta_km"], tolerance)
    frame["proposal_bin"] = pd.cut(
        frame["proposal_count"],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, np.inf],
        labels=PROPOSAL_BIN_ORDER,
        ordered=True,
    )
    proposal_frame = frame.loc[frame["proposal_count"] > 0].copy()

    counts = frame["outcome"].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    proposal_counts = (
        proposal_frame["outcome"].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    )
    exact = (
        frame.groupby("proposal_count", observed=True)
        .agg(
            images=("name", "size"),
            mean_improvement_km=("union_delta_km", "mean"),
            median_improvement_km=("union_delta_km", "median"),
            q25_improvement_km=("union_delta_km", lambda values: values.quantile(0.25)),
            q75_improvement_km=("union_delta_km", lambda values: values.quantile(0.75)),
            improved_rate=("outcome", lambda values: (values == "improved").mean()),
            worsened_rate=("outcome", lambda values: (values == "worsened").mean()),
            unchanged_rate=("outcome", lambda values: (values == "unchanged").mean()),
            mean_union_coverage=("union_coverage", "mean"),
        )
        .reset_index()
    )

    quantiles = proposal_frame["union_delta_km"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    summary = {
        "evaluated_images": int(len(frame)),
        "tolerance_km": tolerance,
        "outcome_all_images": {
            outcome: {
                "count": int(counts[outcome]),
                "rate": float(counts[outcome] / len(frame)),
            }
            for outcome in OUTCOME_ORDER
        },
        "images_with_proposals": int(len(proposal_frame)),
        "images_without_proposals": int((frame["proposal_count"] == 0).sum()),
        "outcome_images_with_proposals": {
            outcome: {
                "count": int(proposal_counts[outcome]),
                "rate": float(proposal_counts[outcome] / len(proposal_frame)),
            }
            for outcome in OUTCOME_ORDER
        },
        "improvement_km_images_with_proposals": {
            "mean": float(proposal_frame["union_delta_km"].mean()),
            "quantiles": {str(index): float(value) for index, value in quantiles.items()},
        },
        "proposal_count_vs_improvement_km": {
            "spearman": safe_correlation(
                spearmanr,
                proposal_frame["proposal_count"],
                proposal_frame["union_delta_km"],
            ),
            "pearson": safe_correlation(
                pearsonr,
                proposal_frame["proposal_count"],
                proposal_frame["union_delta_km"],
            ),
        },
    }
    return summary, exact


def plot_outcome_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    counts = frame["outcome"].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    proposal_frame = frame.loc[frame["proposal_count"] > 0]
    proposal_counts = (
        proposal_frame["outcome"].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    bars = axes[0].bar(
        [OUTCOME_LABELS[value] for value in OUTCOME_ORDER],
        counts.to_numpy(),
        color=[COLORS[value] for value in OUTCOME_ORDER],
    )
    axes[0].set_title("Kết quả trên toàn bộ ảnh")
    axes[0].set_ylabel("Số ảnh")
    for bar, count in zip(bars, counts, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n{count / len(frame):.1%}",
            ha="center",
            va="bottom",
        )

    bars = axes[1].bar(
        [OUTCOME_LABELS[value] for value in OUTCOME_ORDER],
        proposal_counts.to_numpy(),
        color=[COLORS[value] for value in OUTCOME_ORDER],
    )
    axes[1].set_title("Chỉ ảnh có proposal")
    axes[1].set_ylabel("Số ảnh")
    for bar, count in zip(bars, proposal_counts, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n{count / len(proposal_frame):.1%}",
            ha="center",
            va="bottom",
        )

    for outcome in ("improved", "worsened"):
        signed = proposal_frame.loc[proposal_frame["outcome"] == outcome, "union_delta_km"]
        magnitudes, probability = ecdf(signed.abs().to_numpy())
        axes[2].plot(
            magnitudes,
            probability,
            color=COLORS[outcome],
            linewidth=2,
            label=OUTCOME_LABELS[outcome],
        )
    axes[2].set_xscale("log")
    axes[2].set_title("Độ lớn thay đổi khoảng cách")
    axes[2].set_xlabel("|Baseline − intervention| (km, log scale)")
    axes[2].set_ylabel("Tỷ lệ tích lũy")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)

    for axis in axes[:2]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_proposal_relationship(frame: pd.DataFrame, exact: pd.DataFrame, output_path: Path) -> None:
    bin_counts = pd.crosstab(frame["proposal_bin"], frame["outcome"], normalize="index")
    bin_counts = bin_counts.reindex(index=PROPOSAL_BIN_ORDER, columns=OUTCOME_ORDER, fill_value=0)
    bin_sizes = frame["proposal_bin"].value_counts().reindex(PROPOSAL_BIN_ORDER, fill_value=0)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    left = np.zeros(len(bin_counts))
    for outcome in OUTCOME_ORDER:
        values = bin_counts[outcome].to_numpy()
        axes[0].barh(
            range(len(bin_counts)),
            values,
            left=left,
            color=COLORS[outcome],
            label=OUTCOME_LABELS[outcome],
        )
        left += values
    axes[0].set_yticks(
        range(len(bin_counts)),
        [f"{label}  (n={bin_sizes[label]:,})" for label in PROPOSAL_BIN_ORDER],
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Tỷ lệ ảnh")
    axes[0].set_ylabel("Số proposal")
    axes[0].set_title("Kết quả theo số proposal")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    proposal_frame = frame.loc[frame["proposal_count"] > 0].copy()
    proposal_bins = PROPOSAL_BIN_ORDER[1:]
    proposal_frame["proposal_bin"] = proposal_frame["proposal_bin"].cat.remove_unused_categories()
    x_lookup = {label: index for index, label in enumerate(proposal_bins)}
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.16, 0.16, len(proposal_frame))
    axes[1].scatter(
        proposal_frame["proposal_bin"].astype(str).map(x_lookup).to_numpy() + jitter,
        proposal_frame["union_delta_km"],
        c=[COLORS[value] for value in proposal_frame["outcome"]],
        s=10,
        alpha=0.25,
        linewidths=0,
    )
    grouped = (
        proposal_frame.groupby("proposal_bin", observed=True)["union_delta_km"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reindex(proposal_bins)
    )
    axes[1].errorbar(
        range(len(proposal_bins)),
        grouped["median"],
        yerr=np.vstack(
            [
                grouped["median"] - grouped["q25"],
                grouped["q75"] - grouped["median"],
            ]
        ),
        color="#264653",
        marker="o",
        markersize=5,
        linewidth=1.5,
        capsize=3,
        label="Median và IQR",
    )
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_xticks(
        range(len(proposal_bins)),
        [
            f"{label}\n(n={(proposal_frame['proposal_bin'].astype(str) == label).sum():,})"
            for label in proposal_bins
        ],
    )
    axes[1].set_xlabel("Số proposal")
    axes[1].set_ylabel("Cải thiện khoảng cách (km)\n(dương = tốt hơn, symlog)")
    axes[1].set_title("Số proposal và mức cải thiện")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def geographic_grid_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    geographic = frame.copy()
    geographic["longitude_bin"] = pd.cut(
        geographic["target_lon"], np.arange(-180, 181, 20), include_lowest=True
    )
    geographic["latitude_bin"] = pd.cut(
        geographic["target_lat"], np.arange(-90, 91, 15), include_lowest=True
    )
    grid = (
        geographic.groupby(["latitude_bin", "longitude_bin"], observed=True)
        .agg(
            images=("name", "size"),
            improved=("outcome", lambda values: (values == "improved").sum()),
            worsened=("outcome", lambda values: (values == "worsened").sum()),
            unchanged=("outcome", lambda values: (values == "unchanged").sum()),
            mean_improvement_km=("union_delta_km", "mean"),
        )
        .reset_index()
    )
    grid["latitude"] = grid["latitude_bin"].map(lambda interval: interval.mid)
    grid["longitude"] = grid["longitude_bin"].map(lambda interval: interval.mid)
    grid["improved_rate"] = grid["improved"] / grid["images"]
    grid["worsened_rate"] = grid["worsened"] / grid["images"]
    grid["unchanged_rate"] = grid["unchanged"] / grid["images"]
    grid["net_outcome_rate"] = (grid["improved"] - grid["worsened"]) / grid["images"]
    return grid.drop(columns=["latitude_bin", "longitude_bin"])


def plot_geographic_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    projection = ccrs.Robinson()
    plate_carree = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    def prepare(axis) -> None:
        axis.set_global()
        axis.add_feature(cfeature.LAND, facecolor="#f2efe9", zorder=0)
        axis.add_feature(cfeature.OCEAN, facecolor="#eaf3f8", zorder=0)
        axis.coastlines(linewidth=0.45, color="#666666")
        axis.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#999999")

    for axis, outcome in zip(axes.flat[:3], OUTCOME_ORDER, strict=True):
        prepare(axis)
        subset = frame.loc[frame["outcome"] == outcome]
        axis.scatter(
            subset["target_lon"],
            subset["target_lat"],
            s=7,
            alpha=0.38,
            color=COLORS[outcome],
            edgecolors="none",
            transform=plate_carree,
        )
        axis.set_title(f"{OUTCOME_LABELS[outcome]} · n={len(subset):,}")

    net_axis = axes.flat[3]
    prepare(net_axis)
    score = frame["outcome"].map({"improved": 1.0, "worsened": -1.0, "unchanged": 0.0})
    collection = net_axis.hexbin(
        frame["target_lon"],
        frame["target_lat"],
        C=score,
        reduce_C_function=np.mean,
        gridsize=42,
        mincnt=3,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        alpha=0.82,
        transform=plate_carree,
    )
    net_axis.set_title("Xu hướng theo vùng · xanh tốt hơn, đỏ xấu hơn")
    colorbar = figure.colorbar(collection, ax=net_axis, orientation="horizontal", pad=0.035)
    colorbar.set_label("Mean outcome: −1 xấu hơn · 0 không đổi · +1 tốt hơn")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tolerance-km", type=float, default=1e-9)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    required = {
        "name",
        "proposal_count",
        "union_coverage",
        "baseline_distance_km",
        "union_distance_km",
        "union_delta_km",
        "target_lat",
        "target_lon",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["name"].duplicated().any():
        raise ValueError("Input contains duplicate image names")

    frame["outcome"] = classify(frame["union_delta_km"], args.tolerance_km)
    frame["proposal_bin"] = pd.cut(
        frame["proposal_count"],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, np.inf],
        labels=PROPOSAL_BIN_ORDER,
        ordered=True,
    )
    summary, exact = build_statistics(frame, args.tolerance_km)
    geographic_grid = geographic_grid_statistics(frame)
    binned = (
        frame.groupby("proposal_bin", observed=True)
        .agg(
            images=("name", "size"),
            mean_improvement_km=("union_delta_km", "mean"),
            median_improvement_km=("union_delta_km", "median"),
            improved_rate=("outcome", lambda values: (values == "improved").mean()),
            worsened_rate=("outcome", lambda values: (values == "worsened").mean()),
            unchanged_rate=("outcome", lambda values: (values == "unchanged").mean()),
        )
        .reindex(PROPOSAL_BIN_ORDER)
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact.to_csv(args.output_dir / "proposal_count_statistics.csv", index=False)
    binned.to_csv(args.output_dir / "proposal_bin_statistics.csv", index=False)
    geographic_grid.to_csv(args.output_dir / "geographic_grid_statistics.csv", index=False)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_outcome_distribution(frame, args.output_dir / "outcome_distribution.png")
    plot_proposal_relationship(frame, exact, args.output_dir / "proposal_relationship.png")
    plot_geographic_distribution(frame, args.output_dir / "geographic_distribution.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
