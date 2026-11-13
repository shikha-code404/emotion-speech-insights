"""Exploratory Data Analysis (EDA) for Speech Emotion Recognition Manifest.

Analyzes ml/data_manifest/audio_manifest.csv and produces visual plots
and statistical tables in ml/reports/:
1. Class distribution per corpus
2. Duration histogram & summary statistics
3. Clips per speaker distribution
4. Sample-rate and audio channel distribution
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd


def generate_eda_reports(
    manifest_path: Union[str, pathlib.Path] = "ml/data_manifest/audio_manifest.csv",
    reports_dir: Union[str, pathlib.Path] = "ml/reports",
) -> None:
    """Generate EDA charts and statistical summaries from audio manifest."""
    manifest_file = pathlib.Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_file}")

    reports_path = pathlib.Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_file)
    print(f"Loaded manifest with {len(df)} clips across corpora: {df['corpus'].unique().tolist()}")

    # Set visualization styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # -------------------------------------------------------------------------
    # 1. Class Distribution Per Corpus
    # -------------------------------------------------------------------------
    class_table = (
        pd.crosstab(df["emotion"], df["corpus"], margins=True, margins_name="Total")
    )
    class_csv = reports_path / "class_distribution.csv"
    class_table.to_csv(class_csv)
    print(f"Saved: {class_csv}")

    corpora = sorted(df["corpus"].unique())
    fig, axes = plt.subplots(1, len(corpora), figsize=(12, 5), sharey=False)
    if len(corpora) == 1:
        axes = [axes]

    colors = {
        "anger": "#e74c3c",
        "boredom": "#95a5a6",
        "disgust": "#8e44ad",
        "fear": "#34495e",
        "happiness": "#f39c12",
        "neutral": "#3498db",
        "sadness": "#2ecc71",
        "surprise": "#1abc9c",
    }

    for ax, corpus in zip(axes, corpora):
        sub_df = df[df["corpus"] == corpus]
        counts = sub_df["emotion"].value_counts().sort_values(ascending=False)
        bar_colors = [colors.get(e, "#7f8c8d") for e in counts.index]
        bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="black", alpha=0.85)
        ax.set_title(f"{corpus} Class Distribution (N={len(sub_df)})", fontweight="bold")
        ax.set_ylabel("Number of Clips")
        ax.set_xlabel("Vocal Emotion Class")
        ax.tick_params(axis="x", rotation=35)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    class_plot = reports_path / "class_distribution.png"
    plt.savefig(class_plot, dpi=300)
    plt.close()
    print(f"Saved: {class_plot}")

    # -------------------------------------------------------------------------
    # 2. Duration Histogram & Statistics
    # -------------------------------------------------------------------------
    duration_stats = df.groupby("corpus")["duration_seconds"].describe().round(3)
    duration_csv = reports_path / "duration_summary.csv"
    duration_stats.to_csv(duration_csv)
    print(f"Saved: {duration_csv}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for corpus in corpora:
        sub_df = df[df["corpus"] == corpus]
        ax.hist(
            sub_df["duration_seconds"],
            bins=25,
            alpha=0.6,
            edgecolor="black",
            label=f"{corpus} (mean={sub_df['duration_seconds'].mean():.2f}s, min={sub_df['duration_seconds'].min():.2f}s, max={sub_df['duration_seconds'].max():.2f}s)",
        )
    ax.set_title("Clip Duration Distribution by Corpus", fontweight="bold")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Clip Count")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    duration_plot = reports_path / "duration_histogram.png"
    plt.savefig(duration_plot, dpi=300)
    plt.close()
    print(f"Saved: {duration_plot}")

    # -------------------------------------------------------------------------
    # 3. Clips Per Speaker Distribution
    # -------------------------------------------------------------------------
    speaker_counts = (
        df.groupby(["corpus", "speaker_id"]).size().reset_index(name="clip_count")
    )
    speaker_csv = reports_path / "clips_per_speaker.csv"
    speaker_counts.to_csv(speaker_csv, index=False)
    print(f"Saved: {speaker_csv}")

    fig, axes = plt.subplots(1, len(corpora), figsize=(12, 5))
    if len(corpora) == 1:
        axes = [axes]

    for ax, corpus in zip(axes, corpora):
        sub_spk = speaker_counts[speaker_counts["corpus"] == corpus].sort_values(
            by="speaker_id"
        )
        bars = ax.bar(
            sub_spk["speaker_id"].astype(str),
            sub_spk["clip_count"],
            color="#2980b9",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title(f"{corpus} Clips per Speaker", fontweight="bold")
        ax.set_xlabel("Speaker ID")
        ax.set_ylabel("Number of Clips")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    speaker_plot = reports_path / "clips_per_speaker.png"
    plt.savefig(speaker_plot, dpi=300)
    plt.close()
    print(f"Saved: {speaker_plot}")

    # -------------------------------------------------------------------------
    # 4. Sample-Rate Distribution
    # -------------------------------------------------------------------------
    sr_counts = (
        df.groupby(["corpus", "sample_rate"]).size().reset_index(name="clip_count")
    )
    sr_csv = reports_path / "sample_rate_distribution.csv"
    sr_counts.to_csv(sr_csv, index=False)
    print(f"Saved: {sr_csv}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sr_summary = df.groupby(["sample_rate", "corpus"]).size().unstack(fill_value=0)
    sr_summary.plot(kind="bar", stacked=True, ax=ax, edgecolor="black", alpha=0.85)
    ax.set_title("Audio Sample-Rate Distribution by Corpus", fontweight="bold")
    ax.set_xlabel("Sample Rate (Hz)")
    ax.set_ylabel("Number of Clips")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Corpus")
    ax.grid(True, linestyle="--", alpha=0.7)

    sr_plot = reports_path / "sample_rate_distribution.png"
    plt.savefig(sr_plot, dpi=300)
    plt.close()
    print(f"Saved: {sr_plot}")

    print("\nAll EDA reports generated successfully in ml/reports/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on audio manifest")
    parser.add_argument("--manifest", default="ml/data_manifest/audio_manifest.csv", help="Manifest CSV path")
    parser.add_argument("--reports-dir", default="ml/reports", help="Reports output directory")
    args = parser.parse_args()

    try:
        generate_eda_reports(manifest_path=args.manifest, reports_dir=args.reports_dir)
    except Exception as e:
        print(f"Error generating EDA reports: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

