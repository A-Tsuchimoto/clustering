import io
import platform
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import patheffects
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


st.set_page_config(page_title="Patent Clustering App", layout="wide")
st.title("特許文献クラスタリングアプリ")
st.caption("CSVを読み込み、選択列をもとにクラスタリングして2次元可視化・CSV/PNG出力します。")


def read_csv_flexible(uploaded_file: io.BytesIO) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:  # pragma: no cover - best effort fallback
            last_error = e
    raise ValueError(f"CSVの読み込みに失敗しました: {last_error}")


def build_text_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()


def extract_cluster_keywords(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    labels: np.ndarray,
    top_k: int = 5,
) -> dict[int, str]:
    terms = np.array(vectorizer.get_feature_names_out())
    result: dict[int, str] = {}
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() == 0:
            result[int(c)] = ""
            continue
        mean_scores = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_scores)[::-1][:top_k]
        words = [w for w in terms[top_idx] if w.strip()]
        result[int(c)] = " / ".join(words)
    return result


def make_scatter_figure(
    plot_df: pd.DataFrame,
    cluster_keywords: dict[int, str],
    fig_inches: float,
    dpi: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)
    cmap = plt.cm.get_cmap("tab10", max(10, plot_df["cluster"].nunique()))

    for i, cluster_id in enumerate(sorted(plot_df["cluster"].unique())):
        group = plot_df[plot_df["cluster"] == cluster_id]
        ax.scatter(
            group["x"],
            group["y"],
            s=24,
            alpha=0.78,
            label=f"Cluster {cluster_id}",
            color=cmap(i),
        )

        cx = group["x"].mean()
        cy = group["y"].mean()
        label = f"C{cluster_id}: {cluster_keywords.get(cluster_id, '')}"
        text = ax.text(
            cx,
            cy,
            label,
            fontsize=9,
            weight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.72),
        )
        text.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])

    ax.set_title("2次元クラスターマップ")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    x_min, x_max = plot_df["x"].min(), plot_df["x"].max()
    y_min, y_max = plot_df["y"].min(), plot_df["y"].max()
    span = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    pad = span * 0.12 + 1e-9
    ax.set_xlim(x_center - span / 2 - pad, x_center + span / 2 + pad)
    ax.set_ylim(y_center - span / 2 - pad, y_center + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return fig


uploaded = st.file_uploader("CSVファイルをアップロード", type=["csv"])

with st.sidebar:
    st.header("クラスタリング設定")
    n_clusters = st.slider("クラスタ数 (KMeans)", min_value=2, max_value=15, value=6)
    max_features = st.slider("TF-IDF最大語彙数", min_value=1000, max_value=50000, value=10000, step=1000)
    random_state = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)

    st.header("画像出力設定")
    fig_inches = st.slider("画像サイズ (inch, 正方形)", min_value=4.0, max_value=18.0, value=10.0, step=0.5)
    dpi = st.slider("DPI", min_value=100, max_value=600, value=320, step=20)

if uploaded is None:
    st.info("まずCSVをアップロードしてください。")
    st.stop()

try:
    df = read_csv_flexible(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

if df.empty:
    st.error("CSVが空です。")
    st.stop()

st.subheader("1) 読み込む列を選択")
text_candidate_cols = [c for c in df.columns if df[c].dtype == object]
if not text_candidate_cols:
    text_candidate_cols = list(df.columns)

selected_cols = st.multiselect(
    "クラスタリング対象列（複数可）",
    options=text_candidate_cols,
    default=text_candidate_cols[: min(2, len(text_candidate_cols))],
)

if not selected_cols:
    st.warning("少なくとも1列を選択してください。")
    st.stop()

text_series = build_text_series(df, selected_cols)
valid_mask = text_series.str.len() > 0
work_df = df.loc[valid_mask].copy()
work_text = text_series.loc[valid_mask]

if len(work_df) < n_clusters:
    st.error(f"有効データ数({len(work_df)})がクラスタ数({n_clusters})未満です。")
    st.stop()

with st.spinner("特徴量生成・クラスタリング・2次元化を実行中..."):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=max_features,
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(work_text)

    kmeans = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init="auto")
    labels = kmeans.fit_predict(tfidf)

    # 次元圧縮: 疎行列対応のSVDで50次元へ、その後t-SNEで2次元
    svd_dim = max(2, min(50, tfidf.shape[1] - 1))
    reduced = TruncatedSVD(n_components=svd_dim, random_state=int(random_state)).fit_transform(tfidf)
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, max(5, len(work_df) // 10)),
        random_state=int(random_state),
    ).fit_transform(reduced)

cluster_keywords = extract_cluster_keywords(vectorizer, tfidf, labels, top_k=4)

plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels}, index=work_df.index)

fig = make_scatter_figure(plot_df, cluster_keywords, fig_inches=fig_inches, dpi=int(dpi))

left, right = st.columns([2, 1])
with left:
    st.subheader("2) クラスタリング結果の2次元可視化")
    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("3) 再現性情報（手順・ライブラリ・モデル）")
    sil = silhouette_score(tfidf, labels) if len(np.unique(labels)) > 1 else np.nan
    st.markdown(
        f"""
- **手順**: CSV読込 → 列選択 → テキスト結合 → TF-IDF(文字n-gram) → KMeans → SVD → t-SNE(2D)
- **モデル/アルゴリズム**: `TfidfVectorizer`, `KMeans`, `TruncatedSVD`, `TSNE`
- **主なハイパーパラメータ**
  - n_clusters = `{n_clusters}`
  - max_features = `{max_features}`
  - random_state = `{int(random_state)}`
  - fig_inches = `{fig_inches}` (正方形)
  - dpi = `{int(dpi)}`
- **データ件数**: 入力 `{len(df)}` 行 / 有効 `{len(work_df)}` 行
- **評価指標 (Silhouette)**: `{sil:.4f}`
- **実行UTC**: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`
- **実行環境**: Python `{platform.python_version()}`
  - pandas `{pd.__version__}`
  - scikit-learn `{KMeans.__module__.split('.')[0]} ({__import__('sklearn').__version__})`
  - matplotlib `{plt.matplotlib.__version__}`
"""
    )

out_df = work_df.copy()
out_df["_cluster"] = labels
out_df["_x"] = coords[:, 0]
out_df["_y"] = coords[:, 1]

csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "クラスタリング結果CSVをダウンロード",
    data=csv_bytes,
    file_name="clustering_result.csv",
    mime="text/csv",
)

png_buffer = io.BytesIO()
fig.savefig(png_buffer, format="png", dpi=int(dpi), bbox_inches="tight")
png_buffer.seek(0)
st.download_button(
    "2次元マップPNGをダウンロード",
    data=png_buffer,
    file_name="cluster_map.png",
    mime="image/png",
)
