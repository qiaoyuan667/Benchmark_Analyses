"""
Cluster extracted skill labels into a taxonomy using embeddings.

Takes the raw LLM-extracted skill strings, embeds them with
sentence-transformers, reduces dimensionality with UMAP, clusters
with constrained K-Means, and produces a binary Q-matrix
for CDM fitting.

Constraint:
    Exact K clusters, with at least 2 skill strings in each cluster.

Model selection uses a joint criterion:
    joint_score = alpha * normalized_silhouette
                + (1 - alpha) * normalized_qcv

Additionally, for each candidate K in sweep, this script prints:
    - proportion of clusters with QCV > 0.4
    - proportion of clusters with QCV > 0.2
    - cluster-level QCV distribution summary

Outputs also include:
    - cluster_model_selection.png
    - cluster_qcv_boxplot.png
    - cluster_model_selection.csv
    - cluster_qcv_distribution_summary.csv
    - cluster_qcv_details_best.csv

Usage:
    python cluster_skills.py ^
      --input ../data/cdm_ready/skills_extracted.json ^
      --response-matrix ../data/cdm_ready/response_matrix.csv ^
      --cluster-range 50,100,150,200 ^
      --alpha 0.4
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "data" / "cdm_ready")
FIG_DIR = (Path(__file__).resolve().parent.parent / "figures")


def load_skills(input_path):
    """Load extracted skills JSON. Expected format: list of dicts with 'skills' key."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_skills = []
    for item in data:
        skills = item.get("skills", [])
        if isinstance(skills, str):
            skills = eval(skills)
        all_skills.append([s.lower().strip() for s in skills])

    unique = sorted(set(s for skills in all_skills for s in skills))
    counts = Counter(s for skills in all_skills for s in skills)

    print(
        f"Loaded {len(data)} items, "
        f"{sum(len(s) for s in all_skills)} mentions, "
        f"{len(unique)} unique skills"
    )
    print(
        f"Skills/item: mean={np.mean([len(s) for s in all_skills]):.1f}, "
        f"median={np.median([len(s) for s in all_skills]):.0f}"
    )
    return data, all_skills, unique, counts


def load_response_matrix(path):
    """
    Load response matrix of shape (n_models, n_items).
    Assumes first column may be an index column.
    """
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded response matrix: {df.shape[0]} models x {df.shape[1]} items")
    return df


def embed_skills(unique_skills, model_name="all-mpnet-base-v2"):
    """Embed skill strings with a sentence-transformer model."""
    print(f"Embedding {len(unique_skills)} skills with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        unique_skills,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


def pairwise_sq_dists(X, centers):
    """
    Squared Euclidean distances between points and centers.
    X: (N, D)
    centers: (K, D)
    returns: (N, K)
    """
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    xc = X @ centers.T
    return np.maximum(x2 + c2 - 2 * xc, 0.0)


def constrained_kmeans_min2(X, n_clusters, random_state=42, max_iter=50, tol=1e-4):
    """
    Constrained K-Means with exact K clusters and minimum cluster size = 2.
    """
    N, D = X.shape
    if N < 2 * n_clusters:
        raise ValueError(
            f"Cannot enforce min cluster size 2 with N={N} skills and K={n_clusters} clusters. "
            f"Need N >= 2K."
        )

    init_km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = init_km.fit_predict(X)
    centers = init_km.cluster_centers_.copy()

    for _ in range(max_iter):
        old_centers = centers.copy()
        dmat = pairwise_sq_dists(X, centers)  # (N, K)

        mandatory_cost = np.repeat(dmat, repeats=2, axis=1)  # (N, 2K)
        row_ind, col_ind = linear_sum_assignment(mandatory_cost)

        mandatory_points = row_ind
        mandatory_slot_cols = col_ind
        mandatory_clusters = mandatory_slot_cols // 2

        new_labels = np.full(N, -1, dtype=int)
        for i, c in zip(mandatory_points, mandatory_clusters):
            new_labels[i] = c

        remaining = np.where(new_labels == -1)[0]
        if len(remaining) > 0:
            nearest = np.argmin(dmat[remaining], axis=1)
            new_labels[remaining] = nearest

        centers = np.zeros((n_clusters, D), dtype=float)
        for c in range(n_clusters):
            idx = np.where(new_labels == c)[0]
            centers[c] = X[idx].mean(axis=0)

        shift = np.linalg.norm(centers - old_centers)
        labels = new_labels
        if shift < tol:
            break

    final_counts = Counter(labels)
    if len(final_counts) != n_clusters:
        raise RuntimeError("Constrained K-Means failed to return exactly K clusters.")
    if min(final_counts.values()) < 2:
        raise RuntimeError("Constrained K-Means failed to enforce min cluster size >= 2.")

    return labels, centers


def label_clusters(unique_skills, labels, counts):
    """Generate a representative label for each cluster based on most frequent members."""
    cluster_ids = sorted(set(labels))
    cluster_labels = {}

    for cid in cluster_ids:
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        member_counts = [(s, counts.get(s, 0)) for s in members]
        member_counts.sort(key=lambda x: -x[1])

        top = member_counts[0][0]
        label = top.replace("_", " ").title()
        if len(label) > 50:
            label = label[:47] + "..."
        cluster_labels[cid] = label

    return cluster_labels


def build_q_matrix(data, all_skills, unique_skills, labels, cluster_labels):
    """
    Map each item's skills to cluster IDs, produce binary Q-matrix.
    Returns DataFrame of shape (n_items, n_clusters), possibly with metadata columns.
    """
    skill_to_cluster = {unique_skills[i]: labels[i] for i in range(len(unique_skills))}
    cluster_ids = sorted(set(labels))
    cluster_names = [cluster_labels[c] for c in cluster_ids]
    cluster_id_to_colidx = {cid: idx for idx, cid in enumerate(cluster_ids)}

    rows = []
    for item_skills in all_skills:
        row = np.zeros(len(cluster_ids), dtype=int)
        for skill in item_skills:
            cid = skill_to_cluster.get(skill)
            if cid is not None:
                row[cluster_id_to_colidx[cid]] = 1
        rows.append(row)

    q_matrix = pd.DataFrame(rows, columns=cluster_names)

    if data and "item_idx" in data[0]:
        q_matrix.insert(0, "item_idx", [d["item_idx"] for d in data])
    if data and "source" in data[0]:
        insert_pos = 1 if "item_idx" in q_matrix.columns else 0
        q_matrix.insert(insert_pos, "source", [d["source"] for d in data])

    return q_matrix


def compute_qcv_from_qmatrix(response_df, q_matrix, return_details=False):
    """
    Compute overall QCV from a response matrix and a Q-matrix.
    """
    skill_cols = [c for c in q_matrix.columns if c not in ("item_idx", "source")]
    Q_df = q_matrix[skill_cols].copy()

    R = response_df.to_numpy(dtype=float)   # (N, I)
    Q = Q_df.to_numpy(dtype=float)          # (I, K)

    if R.shape[1] != Q.shape[0]:
        raise ValueError(
            f"Response items ({R.shape[1]}) != Q-matrix items ({Q.shape[0]})"
        )

    X = R.T  # (I, N)

    stds = X.std(axis=1)
    non_const = stds > 0

    S = np.full((X.shape[0], X.shape[0]), np.nan)
    if np.any(non_const):
        X_nc = X[non_const]
        corr_nc = np.corrcoef(X_nc)
        idx = np.where(non_const)[0]
        S[np.ix_(idx, idx)] = corr_nc

    np.fill_diagonal(S, 1.0)

    rows = []

    for k, skill_name in enumerate(skill_cols):
        skill_items = np.where(Q[:, k] > 0.5)[0]
        non_skill_items = np.where(Q[:, k] <= 0.5)[0]

        r_within = np.nan
        r_between = np.nan
        qcv = np.nan

        if len(skill_items) >= 2 and len(non_skill_items) > 0:
            within_block = S[np.ix_(skill_items, skill_items)]
            mask = ~np.eye(len(skill_items), dtype=bool)
            within_vals = within_block[mask]
            r_within = np.nanmean(within_vals) if within_vals.size > 0 else np.nan

            between_block = S[np.ix_(skill_items, non_skill_items)]
            r_between = np.nanmean(between_block) if between_block.size > 0 else np.nan

            if not np.isnan(r_within) and not np.isnan(r_between):
                qcv = r_within - r_between

        rows.append({
            "cluster": skill_name,
            "n_items": len(skill_items),
            "r_within": r_within,
            "r_between": r_between,
            "qcv": qcv,
        })

    details_df = pd.DataFrame(rows)
    valid_qcv = details_df["qcv"].dropna()
    overall_qcv = float(valid_qcv.mean()) if len(valid_qcv) > 0 else np.nan

    if return_details:
        return overall_qcv, details_df
    return overall_qcv


def summarize_qcv_thresholds(qcv_details_df):
    """
    Compute cluster-level QCV threshold summaries.
    """
    valid_cluster_qcv = qcv_details_df["qcv"].dropna()
    n_valid = len(valid_cluster_qcv)
    n_total = len(qcv_details_df)

    n_gt_04_valid = int((valid_cluster_qcv > 0.4).sum()) if n_valid > 0 else 0
    n_gt_02_valid = int((valid_cluster_qcv > 0.2).sum()) if n_valid > 0 else 0

    prop_gt_04_valid = float((valid_cluster_qcv > 0.4).mean()) if n_valid > 0 else np.nan
    prop_gt_02_valid = float((valid_cluster_qcv > 0.2).mean()) if n_valid > 0 else np.nan

    gt_04_total_mask = (qcv_details_df["qcv"] > 0.4).fillna(False)
    gt_02_total_mask = (qcv_details_df["qcv"] > 0.2).fillna(False)

    n_gt_04_total = int(gt_04_total_mask.sum())
    n_gt_02_total = int(gt_02_total_mask.sum())

    prop_gt_04_total = float(gt_04_total_mask.mean()) if n_total > 0 else np.nan
    prop_gt_02_total = float(gt_02_total_mask.mean()) if n_total > 0 else np.nan

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_gt_04_valid": n_gt_04_valid,
        "prop_gt_04_valid": prop_gt_04_valid,
        "n_gt_02_valid": n_gt_02_valid,
        "prop_gt_02_valid": prop_gt_02_valid,
        "n_gt_04_total": n_gt_04_total,
        "prop_gt_04_total": prop_gt_04_total,
        "n_gt_02_total": n_gt_02_total,
        "prop_gt_02_total": prop_gt_02_total,
    }


def summarize_qcv_distribution(qcv_details_df, k):
    """
    Summarize cluster-level QCV distribution for one K.
    Returns a dict suitable for DataFrame row storage.
    """
    valid = qcv_details_df["qcv"].dropna()

    if len(valid) == 0:
        return {
            "k": k,
            "n_valid_clusters": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }

    desc = valid.describe(percentiles=[0.25, 0.5, 0.75])

    return {
        "k": k,
        "n_valid_clusters": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if not np.isnan(desc["std"]) else np.nan,
        "min": float(desc["min"]),
        "q25": float(desc["25%"]),
        "median": float(desc["50%"]),
        "q75": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def minmax_scale(series):
    """Min-max scale a pandas Series; if constant, return all ones."""
    s = series.astype(float)
    s_valid = s.replace([np.inf, -np.inf], np.nan)

    if s_valid.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)

    s_min = s_valid.min()
    s_max = s_valid.max()

    if s_max == s_min:
        return pd.Series(np.ones(len(s)), index=s.index)

    out = (s - s_min) / (s_max - s_min)
    return out.fillna(0.0)


def cluster_skills(
    embeddings,
    unique_skills,
    data,
    all_skills,
    counts,
    response_df,
    cluster_range=(50, 100, 150, 200),
    alpha=0.4,
):
    """
    UMAP + constrained K-Means clustering with joint silhouette + QCV selection.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Require 0 <= alpha <= 1")

    reducer_cluster = umap.UMAP(
        n_components=5,
        metric="cosine",
        random_state=42
    )
    reduced = reducer_cluster.fit_transform(embeddings)

    reducer_viz = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=42
    )
    coords_2d = reducer_viz.fit_transform(embeddings)

    print("\nConstrained K-Means sweep:")
    candidates = []
    qcv_boxplot_data = {}
    qcv_distribution_rows = []
    qcv_details_by_k = {}

    for k in cluster_range:
        if len(unique_skills) < 2 * k:
            print(
                f"  K={k}: infeasible because unique_skills={len(unique_skills)} < 2*K={2*k} (skipped)"
            )
            continue

        labels, centers = constrained_kmeans_min2(
            reduced,
            n_clusters=k,
            random_state=42,
            max_iter=50,
            tol=1e-4
        )

        final_counts = Counter(labels)
        n_clusters = len(set(labels))
        min_cluster_size = min(final_counts.values()) if len(final_counts) > 0 else 0
        n_noise = 0

        if n_clusters != k:
            print(f"  K={k}: got {n_clusters} clusters instead of {k} (skipped)")
            continue
        if min_cluster_size < 2:
            print(f"  K={k}: min cluster size {min_cluster_size} < 2 (skipped)")
            continue

        sil = silhouette_score(reduced, labels)

        cluster_labels_tmp = label_clusters(unique_skills, labels, counts)
        q_matrix_tmp = build_q_matrix(
            data,
            all_skills,
            unique_skills,
            labels,
            cluster_labels_tmp
        )

        qcv, qcv_details_df = compute_qcv_from_qmatrix(
            response_df,
            q_matrix_tmp,
            return_details=True
        )
        qcv_summary = summarize_qcv_thresholds(qcv_details_df)
        qcv_dist_summary = summarize_qcv_distribution(qcv_details_df, k)

        qcv_distribution_rows.append(qcv_dist_summary)
        qcv_details_by_k[k] = qcv_details_df.copy()

        valid_qcv = qcv_details_df["qcv"].dropna().to_numpy()
        if len(valid_qcv) > 0:
            qcv_boxplot_data[k] = valid_qcv

        print(
            f"  K={k}: final={n_clusters} clusters, "
            f"min_skill_per_cluster={min_cluster_size}, silhouette={sil:.3f}, qcv={qcv:.3f}"
        )
        print(
            f"       QCV > 0.4: {qcv_summary['n_gt_04_total']}/{qcv_summary['n_total']} "
            f"({qcv_summary['prop_gt_04_total']:.3f} of total clusters)"
        )
        print(
            f"       QCV > 0.2: {qcv_summary['n_gt_02_total']}/{qcv_summary['n_total']} "
            f"({qcv_summary['prop_gt_02_total']:.3f} of total clusters)"
        )
        print(
            f"       QCV distribution: "
            f"mean={qcv_dist_summary['mean']:.3f}, std={qcv_dist_summary['std']:.3f}, "
            f"q25={qcv_dist_summary['q25']:.3f}, median={qcv_dist_summary['median']:.3f}, "
            f"q75={qcv_dist_summary['q75']:.3f}"
        )

        candidates.append({
            "k": k,
            "labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "min_cluster_skill_size": min_cluster_size,
            "silhouette": sil,
            "qcv": qcv,
            "qcv_gt_04_total_count": qcv_summary["n_gt_04_total"],
            "qcv_gt_04_total_prop": qcv_summary["prop_gt_04_total"],
            "qcv_gt_02_total_count": qcv_summary["n_gt_02_total"],
            "qcv_gt_02_total_prop": qcv_summary["prop_gt_02_total"],
        })

    if len(candidates) == 0:
        raise ValueError("No valid clustering candidate found.")

    score_df = pd.DataFrame([
        {
            "k": c["k"],
            "n_clusters": c["n_clusters"],
            "n_noise": c["n_noise"],
            "min_cluster_skill_size": c["min_cluster_skill_size"],
            "silhouette": c["silhouette"],
            "qcv": c["qcv"],
            "qcv_gt_04_total_count": c["qcv_gt_04_total_count"],
            "qcv_gt_04_total_prop": c["qcv_gt_04_total_prop"],
            "qcv_gt_02_total_count": c["qcv_gt_02_total_count"],
            "qcv_gt_02_total_prop": c["qcv_gt_02_total_prop"],
        }
        for c in candidates
    ])

    qcv_distribution_df = pd.DataFrame(qcv_distribution_rows).sort_values("k").reset_index(drop=True)

    score_df["sil_norm"] = minmax_scale(score_df["silhouette"])
    score_df["qcv_norm"] = minmax_scale(score_df["qcv"])
    score_df["joint_score"] = (
        alpha * score_df["sil_norm"] +
        (1 - alpha) * score_df["qcv_norm"]
    )

    best_idx = score_df["joint_score"].idxmax()
    best_row = score_df.loc[best_idx]
    best_k = int(best_row["k"])
    best = next(c for c in candidates if c["k"] == best_k)

    print("\nSelection summary:")
    print(score_df.sort_values("joint_score", ascending=False).to_string(index=False))
    print(
        f"\nSelected K={best_k} "
        f"(final_clusters={best['n_clusters']}, "
        f"min_cluster_skill_size={best['min_cluster_skill_size']}, "
        f"silhouette={best['silhouette']:.3f}, "
        f"qcv={best['qcv']:.3f}, "
        f"joint={best_row['joint_score']:.3f})"
    )

    return best["labels"], coords_2d, score_df, qcv_boxplot_data, qcv_distribution_df, qcv_details_by_k, best_k


def plot_skill_space(coords_2d, labels, cluster_labels, unique_skills, output_path):
    """UMAP 2D scatter plot of the skill embedding space, colored by cluster."""
    fig, ax = plt.subplots(figsize=(12, 8))

    cluster_ids = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(cluster_ids))

    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[cmap(i)],
            label=cluster_labels[cid],
            s=30,
            alpha=0.7,
        )

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Skill Embedding Space (UMAP 2D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved skill space plot: {output_path}")


def plot_cluster_sizes(q_matrix, output_path):
    """Bar chart of how many items involve each skill cluster."""
    skill_cols = [c for c in q_matrix.columns if c not in ("item_idx", "source")]
    counts = q_matrix[skill_cols].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.3)))
    ax.barh(
        counts.index,
        counts.values,
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(counts)))
    )
    ax.set_xlabel("Number of Items")
    ax.set_title("Items per Skill Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cluster size plot: {output_path}")


def plot_model_selection(score_df, output_path):
    """
    Plot silhouette, QCV, QCV-threshold proportions, and joint score versus K.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    x = score_df["k"]

    ax.plot(x, score_df["silhouette"], marker="o", label="Silhouette")
    ax.plot(x, score_df["qcv"], marker="o", label="QCV")
    ax.plot(x, score_df["joint_score"], marker="o", label="Joint Score")
    ax.plot(x, score_df["qcv_gt_04_total_prop"], marker="o", label="Prop(QCV > 0.4)")
    ax.plot(x, score_df["qcv_gt_02_total_prop"], marker="o", label="Prop(QCV > 0.2)")

    ax.set_xlabel("Requested / actual number of clusters (K)")
    ax.set_ylabel("Score / Proportion")
    ax.set_title("Model Selection: Silhouette + QCV + QCV-threshold Proportions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model selection plot: {output_path}")


def plot_qcv_boxplots(qcv_boxplot_data, output_path):
    """
    Plot boxplots of cluster-level QCV distributions across K.
    """
    if len(qcv_boxplot_data) == 0:
        print("No QCV boxplot data available; skipping boxplot.")
        return

    ks = sorted(qcv_boxplot_data.keys())
    data = [qcv_boxplot_data[k] for k in ks]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=[str(k) for k in ks], showfliers=False)
    ax.set_xlabel("K")
    ax.set_ylabel("Cluster-level QCV")
    ax.set_title("Distribution of Cluster-level QCV across K")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved QCV boxplot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster extracted skills into taxonomy")
    parser.add_argument("--input", required=True, help="Path to extracted skills JSON")
    parser.add_argument(
        "--response-matrix",
        required=True,
        help="Path to response_matrix.csv"
    )
    parser.add_argument(
        "--model",
        default="all-mpnet-base-v2",
        help="Sentence-transformer model"
    )
    parser.add_argument(
        "--cluster-range",
        default="50,100,150,200",
        help="Comma-separated n_clusters values to sweep for constrained K-Means"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Weight for silhouette in joint score"
    )
    args = parser.parse_args()

    if args.alpha < 0 or args.alpha > 1:
        raise ValueError("--alpha must be between 0 and 1")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, all_skills, unique_skills, counts = load_skills(args.input)
    response_df = load_response_matrix(args.response_matrix)

    if response_df.shape[1] != len(data):
        raise ValueError(
            f"Response matrix has {response_df.shape[1]} items, "
            f"but skills JSON has {len(data)} items. They must match."
        )

    embeddings = embed_skills(unique_skills, args.model)
    cluster_range = [int(x) for x in args.cluster_range.split(",")]

    (
        labels,
        coords_2d,
        score_df,
        qcv_boxplot_data,
        qcv_distribution_df,
        qcv_details_by_k,
        best_k,
    ) = cluster_skills(
        embeddings=embeddings,
        unique_skills=unique_skills,
        data=data,
        all_skills=all_skills,
        counts=counts,
        response_df=response_df,
        cluster_range=cluster_range,
        alpha=args.alpha,
    )

    cluster_labels = label_clusters(unique_skills, labels, counts)
    n_clusters = len(set(labels))

    final_skill_counts = Counter(labels)
    min_cluster_skill_size = min(final_skill_counts.values()) if len(final_skill_counts) > 0 else 0

    print(f"\nFinal taxonomy ({n_clusters} clusters):")
    print(f"Minimum number of skills per cluster: {min_cluster_skill_size}")
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        print(f"  [{cid}] {cluster_labels[cid]} ({len(members)} skills)")

    q_matrix = build_q_matrix(data, all_skills, unique_skills, labels, cluster_labels)
    q_matrix.to_csv(OUTPUT_DIR / "q_matrix.csv", index=False)
    print(f"\nQ-matrix: {q_matrix.shape[0]} items x {n_clusters} skills")

    skill_cols = [c for c in q_matrix.columns if c not in ("item_idx", "source")]
    skills_per_item = q_matrix[skill_cols].sum(axis=1)
    print(
        f"Skills/item: mean={skills_per_item.mean():.1f}, "
        f"min={skills_per_item.min()}, max={skills_per_item.max()}"
    )

    final_qcv, qcv_details_df = compute_qcv_from_qmatrix(
        response_df,
        q_matrix,
        return_details=True
    )

    print(f"Final QCV: {final_qcv:.3f}")

    qcv_summary = summarize_qcv_thresholds(qcv_details_df)

    print(
        f"Valid clusters with QCV > 0.4: {qcv_summary['n_gt_04_valid']}/{qcv_summary['n_valid']} "
        f"({qcv_summary['prop_gt_04_valid']:.3f} among valid clusters)"
    )
    print(
        f"Valid clusters with QCV > 0.2: {qcv_summary['n_gt_02_valid']}/{qcv_summary['n_valid']} "
        f"({qcv_summary['prop_gt_02_valid']:.3f} among valid clusters)"
    )
    print(
        f"Clusters with QCV > 0.4: {qcv_summary['n_gt_04_total']}/{qcv_summary['n_total']} "
        f"({qcv_summary['prop_gt_04_total']:.3f} of total clusters)"
    )
    print(
        f"Clusters with QCV > 0.2: {qcv_summary['n_gt_02_total']}/{qcv_summary['n_total']} "
        f"({qcv_summary['prop_gt_02_total']:.3f} of total clusters)"
    )

    qcv_details_df.to_csv(OUTPUT_DIR / "cluster_qcv_details.csv", index=False)
    print(f"Saved cluster-level QCV details to {OUTPUT_DIR / 'cluster_qcv_details.csv'}")

    # Save best-K QCV details separately for convenience
    if best_k in qcv_details_by_k:
        qcv_details_by_k[best_k].to_csv(OUTPUT_DIR / "cluster_qcv_details_best.csv", index=False)
        print(f"Saved best-K cluster-level QCV details to {OUTPUT_DIR / 'cluster_qcv_details_best.csv'}")

    qcv_distribution_df.to_csv(OUTPUT_DIR / "cluster_qcv_distribution_summary.csv", index=False)
    print(f"Saved QCV distribution summary to {OUTPUT_DIR / 'cluster_qcv_distribution_summary.csv'}")

    taxonomy = {}
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        taxonomy[cluster_labels[cid]] = members

    with open(OUTPUT_DIR / "skill_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    skill_map = {unique_skills[i]: cluster_labels[labels[i]] for i in range(len(unique_skills))}
    with open(OUTPUT_DIR / "skill_to_cluster.json", "w", encoding="utf-8") as f:
        json.dump(skill_map, f, indent=2, ensure_ascii=False)

    np.savez(
        OUTPUT_DIR / "skill_embeddings.npz",
        embeddings=embeddings,
        skills=np.array(unique_skills, dtype=object),
        labels=labels,
        coords_2d=coords_2d,
    )

    score_df.to_csv(OUTPUT_DIR / "cluster_model_selection.csv", index=False)
    print(f"Saved model selection summary to {OUTPUT_DIR / 'cluster_model_selection.csv'}")

    # plot_skill_space(coords_2d, labels, cluster_labels, unique_skills, FIG_DIR / "skill_space_umap.png")
    # plot_cluster_sizes(q_matrix, FIG_DIR / "cluster_sizes.png")
    plot_model_selection(score_df, FIG_DIR / "cluster_model_selection.png")
    plot_qcv_boxplots(qcv_boxplot_data, FIG_DIR / "cluster_qcv_boxplot.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()