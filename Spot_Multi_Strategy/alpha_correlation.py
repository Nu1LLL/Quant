"""Alpha信号之间的相关性分析：找出冗余信号，估计有效独立信号数量。

只对normalized_signal做相关性分析，不涉及任何前瞻收益标签。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import alphas
from alpha_research import build_alpha_sets, load_symbol_frames
from data import INTERVAL_TO_TIMEDELTA, to_utc_timestamp

REDUNDANCY_THRESHOLD = 0.75

try:
    import scipy.cluster.hierarchy as _scipy_hierarchy  # noqa: F401
    import scipy.spatial.distance as _scipy_distance  # noqa: F401
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def build_normalized_signal_matrix(alpha_signals):
    """把{name: AlphaSignal}拼成一张宽表，列是alpha名字，行是时间。"""
    series_map = {
        name: signal.normalized_signal
        for name, signal in alpha_signals.items()
    }
    return pd.DataFrame(series_map)


def pearson_correlation_matrix(signal_matrix):
    return signal_matrix.corr(method="pearson", min_periods=200)


def spearman_correlation_matrix(signal_matrix):
    # 手动用秩相关，避免依赖scipy（pandas method="spearman"内部需要scipy）
    ranked = signal_matrix.rank()
    return ranked.corr(method="pearson", min_periods=200)


def find_redundant_pairs(correlation_matrix, threshold=REDUNDANCY_THRESHOLD):
    pairs = []
    columns = correlation_matrix.columns

    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = correlation_matrix.loc[left, right]
            if pd.isna(value):
                continue
            if abs(value) >= threshold:
                pairs.append({
                    "alpha_a": left,
                    "alpha_b": right,
                    "correlation": float(value)
                })

    return pd.DataFrame(pairs).sort_values(
        "correlation", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True) if pairs else pd.DataFrame(
        columns=["alpha_a", "alpha_b", "correlation"]
    )


def effective_independent_signals(correlation_matrix):
    """用相关矩阵特征值近似有效独立信号数量（参考PCA中"有效自由度"的做法）：

    N_eff = (sum(eigenvalues))^2 / sum(eigenvalues^2)

    当所有信号完全独立时N_eff等于信号个数；当所有信号完全相关时N_eff趋近1。
    这只是一个近似估计，不是严格的统计检验。
    """
    clean_matrix = correlation_matrix.fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(clean_matrix, 1.0)

    eigenvalues = np.linalg.eigvalsh(clean_matrix)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

    denominator = np.sum(eigenvalues ** 2)
    if denominator == 0:
        return 0.0

    return float(np.sum(eigenvalues) ** 2 / denominator)


def clustered_correlation_table(correlation_matrix):
    """只有scipy可用时才生成层次聚类顺序的相关表，否则原样返回并注明跳过。"""
    if not SCIPY_AVAILABLE:
        return None

    distance_matrix = 1 - correlation_matrix.fillna(0.0).abs()
    condensed = _scipy_distance.squareform(
        distance_matrix.to_numpy(), checks=False
    )
    linkage = _scipy_hierarchy.linkage(condensed, method="average")
    order = _scipy_hierarchy.leaves_list(linkage)

    ordered_columns = correlation_matrix.columns[order]
    return correlation_matrix.loc[ordered_columns, ordered_columns]


def analyze_symbol(symbol, alpha_signals):
    signal_matrix = build_normalized_signal_matrix(alpha_signals)
    pearson_matrix = pearson_correlation_matrix(signal_matrix)
    spearman_matrix = spearman_correlation_matrix(signal_matrix)
    redundant_pairs = find_redundant_pairs(pearson_matrix)
    n_eff = effective_independent_signals(pearson_matrix)
    clustered = clustered_correlation_table(pearson_matrix)

    return {
        "symbol": symbol,
        "alpha_count": len(alpha_signals),
        "pearson_matrix": pearson_matrix,
        "spearman_matrix": spearman_matrix,
        "redundant_pairs": redundant_pairs,
        "effective_independent_signals": n_eff,
        "clustered_matrix": clustered
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Alpha信号相关性与冗余分析"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--threshold", type=float, default=REDUNDANCY_THRESHOLD)
    parser.add_argument("--output-folder", default="alpha_reports")
    return parser.parse_args()


def _resolve_end_time(interval, end):
    if end is not None:
        return to_utc_timestamp(end)

    now = pd.Timestamp.now(tz="UTC")
    interval_seconds = int(INTERVAL_TO_TIMEDELTA[interval].total_seconds())
    now_seconds = int(now.timestamp())
    closed_boundary_seconds = now_seconds // interval_seconds * interval_seconds
    return pd.Timestamp(closed_boundary_seconds, unit="s", tz="UTC")


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    end_time = _resolve_end_time(args.interval, args.end)

    frames = load_symbol_frames(
        symbols=args.symbols,
        interval=args.interval,
        start=args.start,
        end=end_time,
        cache_folder=project_folder / "data_cache"
    )
    alpha_sets = build_alpha_sets(frames)

    output_folder = project_folder / args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"scipy可用：{SCIPY_AVAILABLE}（聚类表只有scipy可用时才生成）")

    for symbol, alpha_signals in alpha_sets.items():
        result = analyze_symbol(symbol, alpha_signals)

        result["pearson_matrix"].to_csv(
            output_folder / f"{symbol}_pearson_correlation.csv"
        )
        result["spearman_matrix"].to_csv(
            output_folder / f"{symbol}_spearman_correlation.csv"
        )
        result["redundant_pairs"].to_csv(
            output_folder / f"{symbol}_redundant_pairs.csv", index=False
        )

        if result["clustered_matrix"] is not None:
            result["clustered_matrix"].to_csv(
                output_folder / f"{symbol}_clustered_correlation.csv"
            )

        print(f"\n{symbol}：{result['alpha_count']}个alpha")
        print(
            "有效独立信号数量（特征值近似）："
            f"{result['effective_independent_signals']:.2f}"
        )
        print(
            f"高度冗余信号对数量（|相关性|>={args.threshold}）："
            f"{len(result['redundant_pairs'])}"
        )
        if not result["redundant_pairs"].empty:
            print(result["redundant_pairs"].head(10).to_string(index=False))

    print(f"\n报告已保存到：{output_folder}")


if __name__ == "__main__":
    main()
