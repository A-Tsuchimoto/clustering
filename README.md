# 特許文献クラスタリングアプリ

## 概要
CSVから特許文献データを読み込み、GUIで列を選択してクラスタリングし、Plotlyで2次元可視化できるStreamlitアプリです。

## 主な機能
- CSVファイルの読み込み（`utf-8-sig` / `utf-8` / `cp932` / `shift_jis` を順に試行）
- GUIで対象列を選択してクラスタリング
- クラスタリング手法を選択可能
  - `KMeans`（クラスタ数を手動指定）
  - `DBSCAN`（クラスタ数を事前指定せず自動推定）
- Plotlyによるインタラクティブ2次元マップ
- 特徴語ラベルをクラスタ中心付近に表示
- クラスタリング結果入りCSVをダウンロード
- 可視化HTMLをダウンロード
- 使用手順・モデル・ライブラリ・主要パラメータをGUIに明示

## 使用ライブラリ / モデル
- `pandas`, `numpy`
- `scikit-learn`
  - `TfidfVectorizer`（文字n-gram）
  - `KMeans`
  - `DBSCAN`
  - `TruncatedSVD`
  - `TSNE`
- `plotly`
- `streamlit`

## 起動方法
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 再現性について
GUI右側に、処理手順・ハイパーパラメータ・実行時刻（UTC）・ライブラリバージョンを表示します。
同一データ・同一seedで再実行すると、同等のクラスタリング結果が得られることを確認しやすくしています。
