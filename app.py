import io
import platform
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


st.set_page_config(page_title="Patent Clustering App", layout="wide")
st.title("特許文献クラスタリングアプリ")
st.caption("CSVを読み込み、選択列をもとにクラスタリングして2次元可視化・CSV/HTML出力します。")


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


JA_EN_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[ぁ-んァ-ン一-龥ー]{2,}")
EN_STOP_WORDS = set(sklearn_text.ENGLISH_STOP_WORDS)
JA_STOP_WORDS = {
    "これ",
    "それ",
    "ため",
    "および",
    "また",
    "及び",
    "又は",
    "ならびに",
    "もの",
    "こと",
    "よう",
    "できる",
    "する",
    "した",
    "して",
    "いる",
    "ある",
    "れる",
    "られる",
    "なる",
    "より",
    "おいて",
    "おける",
    "について",
    "において",
    "とともに",
    "ならび",
    "及",
}


def multilingual_tokenizer(text: str) -> list[str]:
    return [t.lower() for t in JA_EN_TOKEN_PATTERN.findall(text)]


def is_informative_token(token: str) -> bool:
    if not token or token.isdigit():
        return False
    if token in EN_STOP_WORDS or token in JA_STOP_WORDS:
        return False
    if re.fullmatch(r"[a-z]", token):
        return False
    if len(token) < 2:
        return False
    return True


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
        sorted_idx = np.argsort(mean_scores)[::-1]
        words: list[str] = []
        for idx in sorted_idx:
            w = terms[idx]
            if not is_informative_token(w):
                continue
            words.append(w)
            if len(words) >= top_k:
                break
        result[int(c)] = " / ".join(words)
    return result


def make_plotly_figure(plot_df: pd.DataFrame, cluster_keywords: dict[int, str]):
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster_str",
        hover_data={"cluster": True, "cluster_str": False, "x": ':.3f', "y": ':.3f'},
        title="2次元クラスターマップ (Plotly)",
        labels={"x": "Dim-1", "y": "Dim-2", "cluster_str": "Cluster"},
        opacity=0.8,
    )

    for cluster_id in sorted(plot_df["cluster"].unique()):
        group = plot_df[plot_df["cluster"] == cluster_id]
        label = f"C{cluster_id}: {cluster_keywords.get(cluster_id, '')}"
        fig.add_annotation(
            x=float(group["x"].mean()),
            y=float(group["y"].mean()),
            text=label,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.25)",
            font={"size": 11},
        )

    fig.update_layout(legend_title_text="クラスタ", height=760)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


uploaded = st.file_uploader("CSVファイルをアップロード", type=["csv"])

with st.sidebar:
    st.header("クラスタリング設定")
    method = st.selectbox(
        "アルゴリズム",
        options=["KMeans (クラスタ数を指定)", "DBSCAN (クラスタ数を自動推定)"],
    )

    if method.startswith("KMeans"):
        n_clusters = st.slider("クラスタ数 (KMeans)", min_value=2, max_value=15, value=6)
    else:
        dbscan_eps = st.slider("DBSCAN eps", min_value=0.05, max_value=2.00, value=0.45, step=0.05)
        dbscan_min_samples = st.slider("DBSCAN min_samples", min_value=2, max_value=30, value=5)

    max_features = st.slider("TF-IDF最大語彙数", min_value=1000, max_value=50000, value=10000, step=1000)
    random_state = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)

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

if len(work_df) < 4:
    st.error(f"有効データ数({len(work_df)})が少なすぎます。最低4行以上で試してください。")
    st.stop()

if method.startswith("KMeans") and len(work_df) < n_clusters:
    st.error(f"有効データ数({len(work_df)})がクラスタ数({n_clusters})未満です。")
    st.stop()

with st.spinner("特徴量生成・クラスタリング・2次元化を実行中..."):
    cluster_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=max_features,
        min_df=2,
    )
    tfidf = cluster_vectorizer.fit_transform(work_text)

    label_vectorizer = TfidfVectorizer(
        tokenizer=multilingual_tokenizer,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        min_df=2,
    )
    try:
        label_tfidf = label_vectorizer.fit_transform(work_text)
    except ValueError:
        label_vectorizer = cluster_vectorizer
        label_tfidf = tfidf

    svd_dim = max(2, min(50, tfidf.shape[1] - 1))
    reduced = TruncatedSVD(n_components=svd_dim, random_state=int(random_state)).fit_transform(tfidf)

    if method.startswith("KMeans"):
        cluster_model = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init="auto")
        labels = cluster_model.fit_predict(reduced)
    else:
        cluster_model = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples), metric="euclidean")
        labels = cluster_model.fit_predict(reduced)

    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, max(5, len(work_df) // 10)),
        random_state=int(random_state),
    ).fit_transform(reduced)
cluster_keywords = extract_cluster_keywords(label_vectorizer, label_tfidf, labels, top_k=4)

plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels}, index=work_df.index)
plot_df["cluster_str"] = plot_df["cluster"].map(lambda x: "Noise (-1)" if x == -1 else f"Cluster {x}")

fig = make_plotly_figure(plot_df, cluster_keywords)

left, right = st.columns([2, 1])
with left:
    st.subheader("2) クラスタリング結果の2次元可視化")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("3) 再現性情報（手順・ライブラリ・モデル）")
    valid_for_score = labels != -1
    unique_valid = np.unique(labels[valid_for_score])
    if valid_for_score.sum() >= 3 and len(unique_valid) > 1:
        sil = silhouette_score(reduced[valid_for_score], labels[valid_for_score])
    else:
        sil = np.nan

    if method.startswith("KMeans"):
        method_info = f"KMeans (n_clusters={n_clusters})"
    else:
        n_found = len(set(labels) - {-1})
        noise_n = int((labels == -1).sum())
        method_info = (
            "DBSCAN (クラスタ数自動推定) "
            f"eps={float(dbscan_eps):.2f}, min_samples={int(dbscan_min_samples)}, "
            f"検出クラスタ数={n_found}, ノイズ={noise_n}"
        )

    st.markdown(
        f"""
- **手順**: CSV読込 → 列選択 → テキスト結合 → TF-IDF(文字n-gram) → {method_info} → SVD → t-SNE(2D)
- **モデル/アルゴリズム**: `TfidfVectorizer`, `{method.split(' ')[0]}`, `TruncatedSVD`, `TSNE`
- **主なハイパーパラメータ**
  - max_features = `{max_features}`
  - random_state = `{int(random_state)}`
- **データ件数**: 入力 `{len(df)}` 行 / 有効 `{len(work_df)}` 行
- **評価指標 (Silhouette)**: `{sil:.4f}`
- **実行UTC**: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`
- **実行環境**: Python `{platform.python_version()}`
  - pandas `{pd.__version__}`
  - scikit-learn `{KMeans.__module__.split('.')[0]} ({__import__('sklearn').__version__})`
  - plotly `{__import__('plotly').__version__}`
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

html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
st.download_button(
    "2次元マップHTMLをダウンロード",
    data=html_bytes,
    file_name="cluster_map.html",
    mime="text/html",
)
