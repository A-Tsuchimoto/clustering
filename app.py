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


def summarize_top_frequency_by_cluster(
    df_with_cluster: pd.DataFrame,
    target_cols: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(df_with_cluster["_cluster"].unique()):
        cluster_part = df_with_cluster[df_with_cluster["_cluster"] == cluster_id]
        cluster_label = "Noise (-1)" if cluster_id == -1 else f"Cluster {int(cluster_id)}"
        for col in target_cols:
            values = (
                cluster_part[col]
                .fillna("(欠損)")
                .astype(str)
                .str.strip()
                .replace("", "(空文字)")
            )
            top = values.value_counts().head(top_n)
            for rank, (value, count) in enumerate(top.items(), start=1):
                rows.append(
                    {
                        "cluster": cluster_label,
                        "列名": col,
                        "順位": rank,
                        "値": value,
                        "件数": int(count),
                    }
                )
    return pd.DataFrame(rows)


def build_yearly_trend(
    df_with_cluster: pd.DataFrame,
    date_col: str,
) -> tuple[pd.DataFrame, int]:
    date_series = df_with_cluster[date_col]
    parsed_dt = pd.to_datetime(date_series, errors="coerce")

    if parsed_dt.notna().sum() == 0:
        year_from_digits = pd.to_numeric(
            date_series.astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        )
        years = year_from_digits
    else:
        years = parsed_dt.dt.year

    valid = years.notna()
    trend_df = df_with_cluster.loc[valid, ["_cluster"]].copy()
    trend_df["year"] = years.loc[valid].astype(int)
    trend_df["cluster_str"] = trend_df["_cluster"].map(lambda x: "Noise (-1)" if x == -1 else f"Cluster {x}")
    agg_df = trend_df.groupby(["year", "cluster_str"], as_index=False).size().rename(columns={"size": "件数"})
    return agg_df.sort_values(["year", "cluster_str"]).reset_index(drop=True), int(valid.sum())


def evaluate_dbscan_labels(features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    noise_rate = float((labels == -1).mean())
    valid_for_score = labels != -1
    unique_valid = np.unique(labels[valid_for_score])
    if valid_for_score.sum() >= 3 and len(unique_valid) > 1:
        sil = float(silhouette_score(features[valid_for_score], labels[valid_for_score]))
    else:
        sil = float("nan")
    return sil, noise_rate


def pick_dbscan_by_eps_candidates(
    features: np.ndarray,
    eps_candidates: np.ndarray,
    min_samples: int,
    max_noise_rate: float,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    evaluations: list[dict[str, float]] = []
    best_labels: np.ndarray | None = None
    best_eps: float | None = None
    best_score = (-1.0, -np.inf)

    for eps in eps_candidates:
        labels = DBSCAN(eps=float(eps), min_samples=min_samples, metric="euclidean").fit_predict(features)
        sil, noise_rate = evaluate_dbscan_labels(features, labels)
        n_clusters = len(set(labels) - {-1})
        candidate = {
            "eps": float(eps),
            "silhouette": sil,
            "noise_rate": noise_rate,
            "n_clusters": float(n_clusters),
            "within_noise_limit": noise_rate <= max_noise_rate,
        }
        evaluations.append(candidate)

        valid_sil = np.isfinite(sil)
        if not valid_sil:
            continue
        noise_ok = noise_rate <= max_noise_rate
        score_key = (1.0 if noise_ok else 0.0, sil)
        if score_key > best_score:
            best_score = score_key
            best_labels = labels
            best_eps = float(eps)

    if best_labels is None:
        fallback = max(
            evaluations,
            key=lambda x: (-np.inf if not np.isfinite(x["silhouette"]) else x["silhouette"]),
        )
        best_eps = float(fallback["eps"])
        best_labels = DBSCAN(eps=best_eps, min_samples=min_samples, metric="euclidean").fit_predict(features)

    eval_df = pd.DataFrame(evaluations).sort_values("eps").reset_index(drop=True)
    return best_labels, float(best_eps), eval_df


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
        dbscan_eps_mode = st.radio(
            "DBSCAN epsの決め方",
            options=["手動", "自動探索"],
            horizontal=True,
        )
        if dbscan_eps_mode == "手動":
            dbscan_eps = st.slider("DBSCAN eps", min_value=0.05, max_value=2.00, value=0.45, step=0.05)
            dbscan_eps_min = dbscan_eps_max = dbscan_eps
            dbscan_eps_step = 0.05
            dbscan_max_noise_rate = 0.30
        else:
            dbscan_eps_min = st.slider("eps最小値", min_value=0.05, max_value=2.00, value=0.20, step=0.05)
            dbscan_eps_max = st.slider("eps最大値", min_value=0.05, max_value=2.00, value=1.20, step=0.05)
            dbscan_eps_step = st.slider("eps刻み", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
            dbscan_max_noise_rate = st.slider(
                "許容ノイズ率 (自動探索)",
                min_value=0.0,
                max_value=0.9,
                value=0.30,
                step=0.05,
            )
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
        dbscan_eval_df = None
        selected_eps = None
    else:
        if dbscan_eps_mode == "手動":
            selected_eps = float(dbscan_eps_min)
            cluster_model = DBSCAN(eps=selected_eps, min_samples=int(dbscan_min_samples), metric="euclidean")
            labels = cluster_model.fit_predict(reduced)
            dbscan_eval_df = None
        else:
            eps_lo = min(float(dbscan_eps_min), float(dbscan_eps_max))
            eps_hi = max(float(dbscan_eps_min), float(dbscan_eps_max))
            eps_candidates = np.arange(eps_lo, eps_hi + (dbscan_eps_step * 0.5), float(dbscan_eps_step))
            labels, selected_eps, dbscan_eval_df = pick_dbscan_by_eps_candidates(
                reduced,
                eps_candidates=eps_candidates,
                min_samples=int(dbscan_min_samples),
                max_noise_rate=float(dbscan_max_noise_rate),
            )

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
    sil, noise_rate = evaluate_dbscan_labels(reduced, labels)

    if method.startswith("KMeans"):
        method_info = f"KMeans (n_clusters={n_clusters})"
    else:
        n_found = len(set(labels) - {-1})
        noise_n = int((labels == -1).sum())
        if dbscan_eps_mode == "手動":
            dbscan_mode_info = f"手動eps={selected_eps:.2f}"
        else:
            dbscan_mode_info = (
                f"自動eps探索(選択eps={selected_eps:.2f}, 許容ノイズ率<={float(dbscan_max_noise_rate):.2f})"
            )
        method_info = (
            "DBSCAN (クラスタ数自動推定) "
            f"{dbscan_mode_info}, min_samples={int(dbscan_min_samples)}, "
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
- **ノイズ率**: `{noise_rate:.2%}`
- **実行UTC**: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`
- **実行環境**: Python `{platform.python_version()}`
  - pandas `{pd.__version__}`
  - scikit-learn `{KMeans.__module__.split('.')[0]} ({__import__('sklearn').__version__})`
  - plotly `{__import__('plotly').__version__}`
"""
    )

    if method.startswith("DBSCAN") and dbscan_eval_df is not None:
        st.markdown("**eps探索結果（上位）**")
        ranked = (
            dbscan_eval_df.sort_values(
                by=["within_noise_limit", "silhouette"],
                ascending=[False, False],
                na_position="last",
            )
            .head(10)
            .copy()
        )
        ranked["silhouette"] = ranked["silhouette"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "nan")
        ranked["noise_rate"] = ranked["noise_rate"].map(lambda x: f"{x:.2%}")
        ranked["n_clusters"] = ranked["n_clusters"].astype(int)
        ranked["within_noise_limit"] = ranked["within_noise_limit"].map(lambda x: "OK" if x else "超過")
        st.dataframe(
            ranked.rename(
                columns={
                    "eps": "eps",
                    "silhouette": "silhouette",
                    "noise_rate": "noise率",
                    "n_clusters": "クラスタ数",
                    "within_noise_limit": "ノイズ許容内",
                }
            ),
            use_container_width=True,
            hide_index=True,
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

st.divider()
st.subheader("4) 出願人分析（クラスター別 頻度ベスト5）")
applicant_candidate_cols = list(work_df.columns)
default_applicant_cols = [c for c in applicant_candidate_cols if any(k in c.lower() for k in ["applicant", "assignee"])]
if not default_applicant_cols and applicant_candidate_cols:
    default_applicant_cols = applicant_candidate_cols[:1]

applicant_cols = st.multiselect(
    "出願人情報に紐づく列（複数可）",
    options=applicant_candidate_cols,
    default=default_applicant_cols,
)

if applicant_cols:
    cluster_ready_df = out_df.copy()
    freq_df = summarize_top_frequency_by_cluster(cluster_ready_df, applicant_cols, top_n=5)
    if freq_df.empty:
        st.info("集計可能なデータがありません。")
    else:
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
else:
    st.info("出願人分析を行うには、対象列を1つ以上選択してください。")

st.divider()
st.subheader("5) 出願トレンド分析（年次 × クラスター積層棒グラフ）")
date_candidate_cols = list(work_df.columns)
date_like_default = [
    c
    for c in date_candidate_cols
    if any(k in c.lower() for k in ["date", "year", "filing", "出願", "公開", "日付"])
]
default_date_col = date_like_default[0] if date_like_default else date_candidate_cols[0]
selected_date_col = st.selectbox(
    "出願日の情報に紐づく列",
    options=date_candidate_cols,
    index=date_candidate_cols.index(default_date_col),
)

trend_df, valid_date_count = build_yearly_trend(out_df, selected_date_col)
if trend_df.empty:
    st.warning("年を抽出できるデータがありませんでした。日付形式や対象列を確認してください。")
else:
    trend_fig = px.bar(
        trend_df,
        x="year",
        y="件数",
        color="cluster_str",
        barmode="stack",
        title=f"出願年ごとの件数推移（有効日付: {valid_date_count}件）",
        labels={"year": "年", "cluster_str": "クラスタ"},
    )
    trend_fig.update_layout(legend_title_text="クラスタ")
    st.plotly_chart(trend_fig, use_container_width=True)
