import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark results for single-level speculative decoding.")
    parser.add_argument("--summary", default="results/bench_summary.csv")
    parser.add_argument("--prompt_id", default="ALL")
    parser.add_argument("--out_dir", default="results/plots")
    parser.add_argument("--target_model", default="")
    parser.add_argument("--draft_model", default="")
    parser.add_argument("--synced_root", default="results")
    parser.add_argument("--plot_synced_strategy_comparison", action="store_true")
    parser.add_argument("--plot_greedy_throughput_vs_k_across_pairs", action="store_true")
    parser.add_argument("--plot_greedy_prefix_vs_k_across_pairs", action="store_true")
    parser.add_argument("--plot_greedy_acceptance_vs_k_across_pairs", action="store_true")
    parser.add_argument("--plot_best_greedy_throughput_comparison", action="store_true")
    parser.add_argument("--plot_baseline_vs_best_greedy_speculative", action="store_true")
    parser.add_argument("--plot_greedy_cost_breakdown", action="store_true")
    parser.add_argument("--plot_greedy_cost_breakdown_lines", action="store_true")
    parser.add_argument("--plot_greedy_absolute_cost_breakdown", action="store_true")
    return parser.parse_args()


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in [
        "candidate_length_k",
        "acceptance_rate_mean",
        "acceptance_rate_std",
        "avg_accepted_prefix_length_mean",
        "avg_accepted_prefix_length_std",
        "tokens_per_s_mean",
        "tokens_per_s_std",
        "total_generation_time_mean",
        "total_generation_time_std",
        "draft_time_mean",
        "draft_time_std",
        "verify_time_mean",
        "verify_time_std",
        "rebuild_time_mean",
        "rebuild_time_std",
        "draft_time_ratio_mean",
        "draft_time_ratio_std",
        "verify_time_ratio_mean",
        "verify_time_ratio_std",
        "rebuild_time_ratio_mean",
        "rebuild_time_ratio_std",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prettify_strategy(strategy: str) -> str:
    mapping = {
        "greedy": "Greedy",
        "top_k": "Top-k",
        "top_p": "Top-p",
    }
    return mapping.get(strategy, strategy)


def prettify_model_pair(draft_model: str, target_model: str) -> str:
    def short_name(model_name: str) -> str:
        if not isinstance(model_name, str):
            return str(model_name)
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        return model_name.lower()

    return f"{short_name(draft_model)} -> {short_name(target_model)}"


def candidate_axis_positions(values) -> tuple[list, list]:
    ticks = sorted({int(value) for value in values if pd.notna(value)})
    positions = list(range(len(ticks)))
    mapping = {tick: pos for pos, tick in zip(positions, ticks)}
    return ticks, mapping


def filter_scope(df: pd.DataFrame, args) -> pd.DataFrame:
    scope = df[df["prompt_id"] == args.prompt_id].copy()
    if args.target_model:
        scope = scope[scope["target_model"] == args.target_model]
    if args.draft_model:
        scope = scope[(scope["mode"] == "baseline") | (scope["draft_model"] == args.draft_model)]
    return scope


def save_acceptance_rate_plot(spec_df: pd.DataFrame, out_dir: str, prompt_id: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ticks, mapping = candidate_axis_positions(spec_df["candidate_length_k"].tolist())
    for strategy in ["greedy", "top_k", "top_p"]:
        subset = spec_df[spec_df["strategy"] == strategy].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("candidate_length_k")
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]
        ax.errorbar(
            x,
            subset["acceptance_rate_mean"] * 100.0,
            yerr=subset["acceptance_rate_std"] * 100.0,
            marker="o",
            linewidth=2,
            capsize=4,
            label=strategy,
        )

    ax.set_title(f"Acceptance Rate vs Candidate Length (prompt_id={prompt_id})")
    ax.set_xlabel("Candidate length k")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_xticks(list(range(len(ticks))))
    ax.set_xticklabels(ticks)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"acceptance_rate_vs_k_{prompt_id}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_throughput_plot(spec_df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: str, prompt_id: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ticks, mapping = candidate_axis_positions(spec_df["candidate_length_k"].tolist())
    for strategy in ["greedy", "top_k", "top_p"]:
        subset = spec_df[spec_df["strategy"] == strategy].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("candidate_length_k")
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]
        ax.errorbar(
            x,
            subset["tokens_per_s_mean"],
            yerr=subset["tokens_per_s_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"{strategy} speculative",
        )

        baseline_subset = baseline_df[baseline_df["strategy"] == strategy]
        if not baseline_subset.empty:
            baseline_value = float(baseline_subset["tokens_per_s_mean"].iloc[0])
            ax.axhline(
                baseline_value,
                linestyle=":",
                linewidth=1.8,
                alpha=0.8,
                label=f"{strategy} baseline",
            )

    ax.set_title(f"Throughput vs Candidate Length (prompt_id={prompt_id})")
    ax.set_xlabel("Candidate length k")
    ax.set_ylabel("Tokens/s")
    ax.set_xticks(list(range(len(ticks))))
    ax.set_xticklabels(ticks)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"throughput_vs_k_{prompt_id}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_baseline_throughput_plot(baseline_df: pd.DataFrame, out_dir: str, prompt_id: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["greedy", "top_k", "top_p"]
    subset = baseline_df[baseline_df["strategy"].isin(order)].copy()
    subset["strategy"] = pd.Categorical(subset["strategy"], categories=order, ordered=True)
    subset = subset.sort_values("strategy")

    ax.bar(
        subset["strategy"].astype(str),
        subset["tokens_per_s_mean"],
        yerr=subset["tokens_per_s_std"],
        capsize=5,
        alpha=0.85,
    )
    ax.set_title(f"Baseline Throughput by Strategy (prompt_id={prompt_id})")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Tokens/s")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    path = os.path.join(out_dir, f"baseline_throughput_{prompt_id}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def collect_synced_strategy_best_rows(synced_root: str, prompt_id: str) -> pd.DataFrame:
    rows = []
    for entry in sorted(os.listdir(synced_root)):
        summary_path = os.path.join(synced_root, entry, "bench_summary.csv")
        if not entry.endswith("_synced") or not os.path.isfile(summary_path):
            continue

        df = load_summary(summary_path)
        subset = df[
            (df["prompt_id"] == prompt_id)
            & (df["mode"] == "speculative")
            & (df["baseline_impl"] == "manual")
        ].copy()
        if subset.empty:
            continue

        subset = subset.dropna(subset=["candidate_length_k", "tokens_per_s_mean"])
        if subset.empty:
            continue

        subset["pair_label"] = subset.apply(
            lambda row: prettify_model_pair(row["draft_model"], row["target_model"]),
            axis=1,
        )

        for (pair_label, strategy), group in subset.groupby(["pair_label", "strategy"], sort=False):
            best_row = group.sort_values(
                by=["tokens_per_s_mean", "candidate_length_k"],
                ascending=[False, True],
            ).iloc[0]
            row = best_row.to_dict()
            row["pair_label"] = pair_label
            row["strategy_label"] = prettify_strategy(strategy)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    best_df = pd.DataFrame(rows)
    pair_order = list(dict.fromkeys(best_df["pair_label"].tolist()))
    strategy_order = ["greedy", "top_k", "top_p"]
    best_df["pair_label"] = pd.Categorical(best_df["pair_label"], categories=pair_order, ordered=True)
    best_df["strategy"] = pd.Categorical(best_df["strategy"], categories=strategy_order, ordered=True)
    best_df = best_df.sort_values(["pair_label", "strategy"]).reset_index(drop=True)
    return best_df


def collect_synced_rows_for_strategy(synced_root: str, prompt_id: str, strategy: str) -> pd.DataFrame:
    rows = []
    for entry in sorted(os.listdir(synced_root)):
        summary_path = os.path.join(synced_root, entry, "bench_summary.csv")
        if not entry.endswith("_synced") or not os.path.isfile(summary_path):
            continue

        df = load_summary(summary_path)
        subset = df[
            (df["prompt_id"] == prompt_id)
            & (df["mode"] == "speculative")
            & (df["baseline_impl"] == "manual")
            & (df["strategy"] == strategy)
        ].copy()
        if subset.empty:
            continue

        subset = subset.dropna(subset=["candidate_length_k", "tokens_per_s_mean"])
        if subset.empty:
            continue

        subset["pair_label"] = subset.apply(
            lambda row: prettify_model_pair(row["draft_model"], row["target_model"]),
            axis=1,
        )
        rows.append(subset)

    if not rows:
        return pd.DataFrame()

    merged = pd.concat(rows, ignore_index=True)
    pair_order = list(dict.fromkeys(merged["pair_label"].tolist()))
    merged["pair_label"] = pd.Categorical(merged["pair_label"], categories=pair_order, ordered=True)
    merged = merged.sort_values(["pair_label", "candidate_length_k"]).reset_index(drop=True)
    return merged


def collect_manual_baseline_rows_for_strategy(synced_root: str, prompt_id: str, strategy: str) -> pd.DataFrame:
    rows = []
    for entry in sorted(os.listdir(synced_root)):
        summary_path = os.path.join(synced_root, entry, "bench_summary.csv")
        if not entry.endswith("_synced") or not os.path.isfile(summary_path):
            continue

        df = load_summary(summary_path)
        pair_subset = df[
            (df["prompt_id"] == prompt_id)
            & (df["mode"] == "speculative")
            & (df["baseline_impl"] == "manual")
            & (df["strategy"] == strategy)
        ].copy()
        if pair_subset.empty:
            continue
        pair_label = prettify_model_pair(
            str(pair_subset["draft_model"].dropna().iloc[0]),
            str(pair_subset["target_model"].dropna().iloc[0]),
        )

        subset = df[
            (df["prompt_id"] == prompt_id)
            & (df["mode"] == "baseline")
            & (df["baseline_impl"] == "manual")
            & (df["strategy"] == strategy)
        ].copy()
        if subset.empty:
            continue

        subset = subset.dropna(subset=["tokens_per_s_mean"])
        if subset.empty:
            continue

        subset["pair_label"] = pair_label
        rows.append(subset)

    if not rows:
        return pd.DataFrame()

    merged = pd.concat(rows, ignore_index=True)
    pair_order = list(dict.fromkeys(merged["pair_label"].tolist()))
    merged["pair_label"] = pd.Categorical(merged["pair_label"], categories=pair_order, ordered=True)
    merged = merged.sort_values(["pair_label"]).reset_index(drop=True)
    return merged


def save_synced_strategy_comparison_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    best_df = collect_synced_strategy_best_rows(synced_root, prompt_id)
    if best_df.empty:
        raise ValueError("No synced speculative/manual summary rows found for the selected prompt_id.")

    pair_order = list(best_df["pair_label"].cat.categories)
    strategy_order = ["greedy", "top_k", "top_p"]
    colors = {
        "greedy": "#4C956C",
        "top_k": "#D17B0F",
        "top_p": "#2F6690",
    }

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = list(range(len(pair_order)))
    width = 0.24

    for offset_idx, strategy in enumerate(strategy_order):
        subset = best_df[best_df["strategy"] == strategy].copy()
        subset = subset.set_index("pair_label").reindex(pair_order).reset_index()
        xpos = [value + (offset_idx - 1) * width for value in x]
        bars = ax.bar(
            xpos,
            subset["tokens_per_s_mean"],
            width=width,
            color=colors[strategy],
            alpha=0.9,
            label=prettify_strategy(strategy),
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            if pd.isna(row["tokens_per_s_mean"]):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"k={int(row['candidate_length_k'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(f"Best Throughput by Decoding Strategy Across Model Pairs (prompt_id={prompt_id})")
    ax.set_xlabel("Model pair")
    ax.set_ylabel("Best tokens/s")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_order, rotation=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"strategy_throughput_comparison_{prompt_id}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_baseline_vs_best_greedy_speculative_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    spec_df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    baseline_df = collect_manual_baseline_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if spec_df.empty or baseline_df.empty:
        raise ValueError("Missing greedy speculative or manual baseline rows for the selected prompt_id.")

    best_spec_rows = []
    for pair_label, subset in spec_df.groupby("pair_label", sort=False, observed=False):
        subset = subset.dropna(subset=["tokens_per_s_mean"])
        if subset.empty:
            continue
        best_row = subset.sort_values(
            by=["tokens_per_s_mean", "candidate_length_k"],
            ascending=[False, True],
        ).iloc[0]
        best_spec_rows.append(
            {
                "pair_label": pair_label,
                "spec_tokens_per_s_mean": best_row["tokens_per_s_mean"],
                "best_k": int(best_row["candidate_length_k"]),
            }
        )

    if not best_spec_rows:
        raise ValueError("No valid greedy speculative rows found for the selected prompt_id.")

    best_spec_df = pd.DataFrame(best_spec_rows)
    baseline_plot_df = baseline_df[["pair_label", "tokens_per_s_mean"]].copy()
    baseline_plot_df = baseline_plot_df.rename(columns={"tokens_per_s_mean": "baseline_tokens_per_s_mean"})

    merged = baseline_plot_df.merge(best_spec_df, on="pair_label", how="inner")
    if merged.empty:
        raise ValueError("No overlapping model pairs between manual baseline and greedy speculative results.")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = list(range(len(merged)))
    width = 0.34

    baseline_bars = ax.bar(
        [value - width / 2 for value in x],
        merged["baseline_tokens_per_s_mean"],
        width=width,
        color="#7A7A7A",
        alpha=0.9,
        label="Manual baseline",
    )
    spec_bars = ax.bar(
        [value + width / 2 for value in x],
        merged["spec_tokens_per_s_mean"],
        width=width,
        color="#4C956C",
        alpha=0.9,
        label="Best greedy speculative",
    )

    for bar, (_, row) in zip(baseline_bars, merged.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{row['baseline_tokens_per_s_mean']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, (_, row) in zip(spec_bars, merged.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{row['spec_tokens_per_s_mean']:.2f}\n(k={row['best_k']})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"Manual Baseline vs Best Greedy Speculative Across Model Pairs (prompt_id={prompt_id})")
    ax.set_xlabel("Model pair")
    ax.set_ylabel("Tokens/s")
    ax.set_ylim(0, 80)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["pair_label"])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"baseline_vs_best_greedy_speculative_{prompt_id}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_greedy_cost_breakdown_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if df.empty:
        raise ValueError("No greedy speculative/manual summary rows found for the selected prompt_id.")

    df = df.dropna(
        subset=[
            "draft_time_ratio_mean",
            "verify_time_ratio_mean",
            "rebuild_time_ratio_mean",
        ]
    )
    if df.empty:
        raise ValueError("No cost breakdown ratio values found for the selected prompt_id.")

    pair_groups = list(df.groupby("pair_label", sort=False, observed=False))
    fig, axes = plt.subplots(1, len(pair_groups), figsize=(5.3 * len(pair_groups), 5.2), sharey=True)
    if len(pair_groups) == 1:
        axes = [axes]

    colors = {
        "draft": "#4C956C",
        "verify": "#D17B0F",
        "rebuild": "#2F6690",
    }

    for ax, (pair_label, subset) in zip(axes, pair_groups):
        subset = subset.sort_values("candidate_length_k")
        ticks, mapping = candidate_axis_positions(subset["candidate_length_k"].tolist())
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]
        draft = (subset["draft_time_ratio_mean"] * 100.0).tolist()
        verify = (subset["verify_time_ratio_mean"] * 100.0).tolist()
        rebuild = (subset["rebuild_time_ratio_mean"] * 100.0).tolist()

        ax.bar(x, draft, color=colors["draft"], alpha=0.9, label="Draft")
        ax.bar(x, verify, bottom=draft, color=colors["verify"], alpha=0.9, label="Verify")
        stacked_bottom = [draft_i + verify_i for draft_i, verify_i in zip(draft, verify)]
        ax.bar(x, rebuild, bottom=stacked_bottom, color=colors["rebuild"], alpha=0.9, label="Rebuild")

        ax.set_title(pair_label)
        ax.set_xlabel("Candidate length k")
        ax.set_xticks(list(range(len(ticks))))
        ax.set_xticklabels(ticks)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Time ratio (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"Cost Breakdown Across Candidate Lengths (Greedy, prompt_id={prompt_id})", y=0.98)
    fig.tight_layout()

    path = os.path.join(out_dir, f"greedy_cost_breakdown_{prompt_id}.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_greedy_cost_breakdown_lines_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if df.empty:
        raise ValueError("No greedy speculative/manual summary rows found for the selected prompt_id.")

    df = df.dropna(
        subset=[
            "draft_time_ratio_mean",
            "verify_time_ratio_mean",
            "rebuild_time_ratio_mean",
        ]
    )
    if df.empty:
        raise ValueError("No cost breakdown ratio values found for the selected prompt_id.")

    pair_groups = list(df.groupby("pair_label", sort=False, observed=False))
    fig, axes = plt.subplots(1, len(pair_groups), figsize=(5.3 * len(pair_groups), 5.2), sharey=True)
    if len(pair_groups) == 1:
        axes = [axes]

    colors = {
        "draft": "#4C956C",
        "verify": "#D17B0F",
        "rebuild": "#2F6690",
    }

    for ax, (pair_label, subset) in zip(axes, pair_groups):
        subset = subset.sort_values("candidate_length_k")
        ticks, mapping = candidate_axis_positions(subset["candidate_length_k"].tolist())
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]

        ax.plot(x, subset["draft_time_ratio_mean"] * 100.0, marker="o", linewidth=2.2, color=colors["draft"], label="Draft")
        ax.plot(x, subset["verify_time_ratio_mean"] * 100.0, marker="o", linewidth=2.2, color=colors["verify"], label="Verify")
        ax.plot(x, subset["rebuild_time_ratio_mean"] * 100.0, marker="o", linewidth=2.2, color=colors["rebuild"], label="Rebuild")

        ax.set_title(pair_label)
        ax.set_xlabel("Candidate length k")
        ax.set_xticks(list(range(len(ticks))))
        ax.set_xticklabels(ticks)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Time ratio (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"Cost Breakdown Across Candidate Lengths (Lines, Greedy, prompt_id={prompt_id})", y=0.98)
    fig.tight_layout()

    path = os.path.join(out_dir, f"greedy_cost_breakdown_lines_{prompt_id}.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_greedy_absolute_cost_breakdown_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if df.empty:
        raise ValueError("No greedy speculative/manual summary rows found for the selected prompt_id.")

    df = df.dropna(
        subset=[
            "draft_time_mean",
            "verify_time_mean",
            "rebuild_time_mean",
        ]
    )
    if df.empty:
        raise ValueError("No absolute cost breakdown values found for the selected prompt_id.")

    pair_groups = list(df.groupby("pair_label", sort=False, observed=False))
    fig, axes = plt.subplots(1, len(pair_groups), figsize=(5.3 * len(pair_groups), 5.2), sharey=True)
    if len(pair_groups) == 1:
        axes = [axes]

    colors = {
        "draft": "#4C956C",
        "verify": "#D17B0F",
        "rebuild": "#2F6690",
    }

    for ax, (pair_label, subset) in zip(axes, pair_groups):
        subset = subset.sort_values("candidate_length_k")
        ticks, mapping = candidate_axis_positions(subset["candidate_length_k"].tolist())
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]
        draft = subset["draft_time_mean"].tolist()
        verify = subset["verify_time_mean"].tolist()
        rebuild = subset["rebuild_time_mean"].tolist()

        ax.bar(x, draft, color=colors["draft"], alpha=0.9, label="Draft")
        ax.bar(x, verify, bottom=draft, color=colors["verify"], alpha=0.9, label="Verify")
        stacked_bottom = [draft_i + verify_i for draft_i, verify_i in zip(draft, verify)]
        ax.bar(x, rebuild, bottom=stacked_bottom, color=colors["rebuild"], alpha=0.9, label="Rebuild")

        ax.set_title(pair_label)
        ax.set_xlabel("Candidate length k")
        ax.set_xticks(list(range(len(ticks))))
        ax.set_xticklabels(ticks)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"Absolute Cost Breakdown Across Candidate Lengths (Greedy, prompt_id={prompt_id})", y=0.98)
    fig.tight_layout()

    path = os.path.join(out_dir, f"greedy_absolute_cost_breakdown_{prompt_id}.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_greedy_throughput_vs_k_across_pairs_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if df.empty:
        raise ValueError("No greedy speculative/manual summary rows found for the selected prompt_id.")

    colors = ["#4C956C", "#D17B0F", "#2F6690"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for color, (pair_label, subset) in zip(colors, df.groupby("pair_label", sort=False, observed=False)):
        subset = subset.sort_values("candidate_length_k")
        ticks, mapping = candidate_axis_positions(subset["candidate_length_k"].tolist())
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]
        ax.plot(
            x,
            subset["tokens_per_s_mean"],
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=color,
            label=pair_label,
        )

    ax.set_title(f"Throughput vs Candidate Length Across Model Pairs (Greedy, prompt_id={prompt_id})")
    ax.set_xlabel("Candidate length k")
    ax.set_ylabel("Tokens/s")
    ax.set_ylim(0, 70)
    ticks, _ = candidate_axis_positions(df["candidate_length_k"].tolist())
    ax.set_xticks(list(range(len(ticks))))
    ax.set_xticklabels(ticks)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"greedy_throughput_vs_k_across_pairs_{prompt_id}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_greedy_prefix_vs_k_across_pairs_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if df.empty:
        raise ValueError("No greedy speculative/manual summary rows found for the selected prompt_id.")

    df = df.dropna(subset=["avg_accepted_prefix_length_mean"])
    if df.empty:
        raise ValueError("No avg_accepted_prefix_length_mean values found for the selected prompt_id.")

    colors = ["#4C956C", "#D17B0F", "#2F6690"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for color, (pair_label, subset) in zip(colors, df.groupby("pair_label", sort=False, observed=False)):
        subset = subset.sort_values("candidate_length_k")
        ticks, mapping = candidate_axis_positions(subset["candidate_length_k"].tolist())
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]
        ax.plot(
            x,
            subset["avg_accepted_prefix_length_mean"],
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=color,
            label=pair_label,
        )

    ax.set_title(f"Average Accepted Prefix Length vs Candidate Length Across Model Pairs (Greedy, prompt_id={prompt_id})")
    ax.set_xlabel("Candidate length k")
    ax.set_ylabel("Average accepted prefix length")
    ticks, _ = candidate_axis_positions(df["candidate_length_k"].tolist())
    ax.set_xticks(list(range(len(ticks))))
    ax.set_xticklabels(ticks)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"greedy_prefix_vs_k_across_pairs_{prompt_id}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_greedy_acceptance_vs_k_across_pairs_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if df.empty:
        raise ValueError("No greedy speculative/manual summary rows found for the selected prompt_id.")

    df = df.dropna(subset=["acceptance_rate_mean"])
    if df.empty:
        raise ValueError("No acceptance_rate_mean values found for the selected prompt_id.")

    colors = ["#4C956C", "#D17B0F", "#2F6690"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for color, (pair_label, subset) in zip(colors, df.groupby("pair_label", sort=False, observed=False)):
        subset = subset.sort_values("candidate_length_k")
        ticks, mapping = candidate_axis_positions(subset["candidate_length_k"].tolist())
        x = [mapping[int(value)] for value in subset["candidate_length_k"].tolist()]
        ax.plot(
            x,
            subset["acceptance_rate_mean"] * 100.0,
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=color,
            label=pair_label,
        )

    ax.set_title(f"Acceptance Rate vs Candidate Length Across Model Pairs (Greedy, prompt_id={prompt_id})")
    ax.set_xlabel("Candidate length k")
    ax.set_ylabel("Acceptance rate (%)")
    ticks, _ = candidate_axis_positions(df["candidate_length_k"].tolist())
    ax.set_xticks(list(range(len(ticks))))
    ax.set_xticklabels(ticks)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"greedy_acceptance_vs_k_across_pairs_{prompt_id}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_best_greedy_throughput_comparison_plot(synced_root: str, out_dir: str, prompt_id: str) -> str:
    df = collect_synced_rows_for_strategy(synced_root, prompt_id, strategy="greedy")
    if df.empty:
        raise ValueError("No greedy speculative/manual summary rows found for the selected prompt_id.")

    rows = []
    for pair_label, subset in df.groupby("pair_label", sort=False, observed=False):
        subset = subset.dropna(subset=["tokens_per_s_mean"])
        if subset.empty:
            continue
        best_row = subset.sort_values(
            by=["tokens_per_s_mean", "candidate_length_k"],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            {
                "pair_label": pair_label,
                "tokens_per_s_mean": best_row["tokens_per_s_mean"],
                "candidate_length_k": int(best_row["candidate_length_k"]),
            }
        )

    if not rows:
        raise ValueError("No valid greedy throughput rows found for the selected prompt_id.")

    best_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = ["#4C956C", "#D17B0F", "#2F6690"]

    bars = ax.bar(
        best_df["pair_label"],
        best_df["tokens_per_s_mean"],
        color=colors[: len(best_df)],
        alpha=0.9,
    )

    for bar, (_, row) in zip(bars, best_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{row['tokens_per_s_mean']:.2f}\n(k={row['candidate_length_k']})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"Best Greedy Throughput Across Model Pairs (prompt_id={prompt_id})")
    ax.set_xlabel("Model pair")
    ax.set_ylabel("Best tokens/s")
    ax.set_ylim(0, 70)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    path = os.path.join(out_dir, f"best_greedy_throughput_comparison_{prompt_id}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.plot_greedy_absolute_cost_breakdown:
        absolute_cost_breakdown_plot_path = save_greedy_absolute_cost_breakdown_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"absolute_cost_breakdown_plot={absolute_cost_breakdown_plot_path}")
        return

    if args.plot_greedy_cost_breakdown_lines:
        cost_breakdown_lines_plot_path = save_greedy_cost_breakdown_lines_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"cost_breakdown_lines_plot={cost_breakdown_lines_plot_path}")
        return

    if args.plot_greedy_cost_breakdown:
        cost_breakdown_plot_path = save_greedy_cost_breakdown_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"cost_breakdown_plot={cost_breakdown_plot_path}")
        return

    if args.plot_baseline_vs_best_greedy_speculative:
        baseline_compare_plot_path = save_baseline_vs_best_greedy_speculative_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"baseline_compare_plot={baseline_compare_plot_path}")
        return

    if args.plot_best_greedy_throughput_comparison:
        best_greedy_plot_path = save_best_greedy_throughput_comparison_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"best_greedy_plot={best_greedy_plot_path}")
        return

    if args.plot_greedy_acceptance_vs_k_across_pairs:
        acceptance_plot_path = save_greedy_acceptance_vs_k_across_pairs_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"greedy_acceptance_plot={acceptance_plot_path}")
        return

    if args.plot_greedy_prefix_vs_k_across_pairs:
        prefix_plot_path = save_greedy_prefix_vs_k_across_pairs_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"greedy_prefix_plot={prefix_plot_path}")
        return

    if args.plot_greedy_throughput_vs_k_across_pairs:
        greedy_plot_path = save_greedy_throughput_vs_k_across_pairs_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"greedy_throughput_plot={greedy_plot_path}")
        return

    if args.plot_synced_strategy_comparison:
        synced_plot_path = save_synced_strategy_comparison_plot(
            synced_root=args.synced_root,
            out_dir=args.out_dir,
            prompt_id=args.prompt_id,
        )
        print(f"synced_strategy_plot={synced_plot_path}")
        return

    df = load_summary(args.summary)
    scope = filter_scope(df, args)

    spec_df = scope[scope["mode"] == "speculative"].copy()
    baseline_df = scope[scope["mode"] == "baseline"].copy()

    if spec_df.empty:
        raise ValueError("No speculative rows found for the selected filters.")

    acceptance_path = save_acceptance_rate_plot(spec_df, args.out_dir, args.prompt_id)
    throughput_path = save_throughput_plot(spec_df, baseline_df, args.out_dir, args.prompt_id)
    baseline_throughput_path = ""
    if not baseline_df.empty:
        baseline_throughput_path = save_baseline_throughput_plot(baseline_df, args.out_dir, args.prompt_id)

    print(f"acceptance_plot={acceptance_path}")
    print(f"throughput_plot={throughput_path}")
    if baseline_throughput_path:
        print(f"baseline_throughput_plot={baseline_throughput_path}")


if __name__ == "__main__":
    main()
