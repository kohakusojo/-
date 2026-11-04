# 筋トレフォームチェック（Streamlit）

ベンチプレス・アームカール・スクワットの3種目をMediaPipe Poseで解析する試作サイトです。動画アップロード、骨格描画、角度グラフ、推定回数、CSV/MP4ダウンロードを実装しています。

## 実行

Python 3.11 推奨。

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud 公開

1. このフォルダの内容をGitHubリポジトリにアップロード。
2. https://share.streamlit.io/ でGitHubリポジトリを選択。
3. Main file path に `streamlit_app.py` を指定してデプロイ。

## 機械学習モデルの追加

`models/bench_press.joblib`, `models/bicep_curl.joblib`, `models/squat.joblib` を置くと、対応モデルの `predict()` を呼び出します。**現在このZIPには学習済みモデルは含まれません。** モデルは信頼できるファイルのみ読み込んでください（joblibは任意コード実行の危険があります）。

入力は以下の7特徴量を**この順序で**学習したscikit-learn分類器である必要があります。

`left_elbow, right_elbow, left_knee, right_knee, left_hip, right_hip, torso_tilt`

特徴量は2次元ランドマークの角度（度）です。既存のExercise-Correctionモデルの入力仕様とは互換性が確認されていません。既存モデルを直接コピーしても動作するとは限りません。元モデルの特徴量抽出・スケーラー・クラス定義を確認してアダプターを実装してください。

学習済みモデルの出力ラベルは学習データ依存であり、正誤判定として扱うにはラベル定義と検証が必要です。ベンチプレスの学習済みモデルは新規作成が必要です。

## 注意

- カウントは簡易的な角度閾値ベースで、精度保証はありません。
- 100点評価・筋肉図・個別の改善助言は、学習モデルと評価基準が未整備のため未実装です。
- 解析は最大1800フレーム。音声は出力に含まれません。
- 一部のスマホMOV/HEVC動画はOpenCVで読み取れないため、H.264 MP4に変換してください。
- MITライセンスの元プロジェクトを参考にしていますが、元の学習済みモデルやコードを同梱していません。
