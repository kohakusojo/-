
import streamlit as st

st.set_page_config(
    page_title="筋トレフォームチェック",
    page_icon="🏋️",
    layout="wide"
)

st.title("🏋️ 筋トレフォームチェック")
st.success("サイトの起動に成功しました！")

exercise = st.selectbox(
    "筋トレ種目を選択",
    ["ベンチプレス", "アームカール", "スクワット"]
)

video = st.file_uploader(
    "筋トレ動画をアップロード",
    type=["mp4", "mov", "avi"]
)

if video:
    st.video(video)
    st.info(f"{exercise}の動画を読み込みました。")
