import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from model_loader import load_classifier, predict, FEATURE_NAMES

st.set_page_config(page_title="筋トレフォームチェック", page_icon="🏋️", layout="wide")
st.title("🏋️ 筋トレフォームチェック")
st.caption("MediaPipeによる関節角度解析 + 任意の学習済みモデル推論。フォーム評価は参考値であり安全性を保証しません。")
EXERCISES = {"ベンチプレス":"bench_press", "アームカール":"bicep_curl", "スクワット":"squat"}
name = st.selectbox("種目を選択", list(EXERCISES))
exercise = EXERCISES[name]
video = st.file_uploader("動画をアップロード（MP4 / MOV / AVI）", type=["mp4","mov","avi"])
st.info("撮影の目安：全身が映る横方向の動画。左右の関節が見える明るい場所で撮影してください。")
model, model_status = load_classifier(exercise)
st.write("**機械学習:** " + model_status)
if model is None:
    st.warning("学習済みモデルがないため、現在は角度解析のみ実行します。モデルの判定結果は表示しません。")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def angle(a, b, c):
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    u, v = a-b, c-b
    den = np.linalg.norm(u)*np.linalg.norm(v)
    if den < 1e-8:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/den, -1, 1))))

def features_of(lms):
    p = lambda i: (lms[i].x, lms[i].y)
    P = mp_pose.PoseLandmark
    def a(x,y,z): return angle(p(x.value),p(y.value),p(z.value))
    f = {
        "left_elbow":a(P.LEFT_SHOULDER,P.LEFT_ELBOW,P.LEFT_WRIST),
        "right_elbow":a(P.RIGHT_SHOULDER,P.RIGHT_ELBOW,P.RIGHT_WRIST),
        "left_knee":a(P.LEFT_HIP,P.LEFT_KNEE,P.LEFT_ANKLE),
        "right_knee":a(P.RIGHT_HIP,P.RIGHT_KNEE,P.RIGHT_ANKLE),
        "left_hip":a(P.LEFT_SHOULDER,P.LEFT_HIP,P.LEFT_KNEE),
        "right_hip":a(P.RIGHT_SHOULDER,P.RIGHT_HIP,P.RIGHT_KNEE),
    }
    shoulder = np.mean([p(P.LEFT_SHOULDER.value),p(P.RIGHT_SHOULDER.value)],axis=0)
    hip = np.mean([p(P.LEFT_HIP.value),p(P.RIGHT_HIP.value)],axis=0)
    vec = shoulder-hip
    f["torso_tilt"] = float(np.degrees(np.arctan2(abs(vec[0]),abs(vec[1])+1e-8)))
    return f

def measure(f, ex):
    key = "left_elbow" if ex in ("bench_press","bicep_curl") else "left_knee"
    alt = "right_elbow" if ex in ("bench_press","bicep_curl") else "right_knee"
    return float(np.nanmean([f[key],f[alt]]))

if video is not None and st.button("解析を開始", type="primary"):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "input" + os.path.splitext(video.name)[1].lower())
        with open(src,"wb") as out: out.write(video.getbuffer())
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps < 1 or fps > 120: fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened():
            st.error("動画を読み込めません。H.264形式のMP4に変換して再試行してください。")
            st.stop()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, 720 / max(width,1))
        size = (max(2,int(width*scale)//2*2),max(2,int(height*scale)//2*2))
        dst = os.path.join(td,"annotated.mp4")
        writer = cv2.VideoWriter(dst,cv2.VideoWriter_fourcc(*"mp4v"),min(fps,30),size)
        if not writer.isOpened():
            st.error("出力動画を作成できませんでした。")
            st.stop()
        rows=[]; reps=0; state="ready"; good_frames=0; valid_frames=0
        thresholds = {"bench_press":(95,150),"bicep_curl":(65,145),"squat":(95,155)}
        low, high = thresholds[exercise]
        bar=st.progress(0)
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            idx=0
            while idx < 1800:
                ok,frame=cap.read()
                if not ok: break
                idx+=1
                frame=cv2.resize(frame,size)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                result=pose.process(rgb)
                if result.pose_landmarks:
                    lms=result.pose_landmarks.landmark
                    mp_draw.draw_landmarks(frame,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                    f=features_of(lms)
                    value=measure(f,exercise)
                    if np.isfinite(value):
                        valid_frames+=1
                        # Count only complete cycles; hysteresis reduces jitter.
                        if exercise == "bicep_curl":
                            if value > high: state="extended"
                            elif value < low and state == "extended": reps+=1; state="flexed"
                        else:
                            if value < low: state="bottom"
                            elif value > high and state == "bottom": reps+=1; state="top"
                        good_frames += int(low <= value <= high)
                    label=predict(model,f)
                    rows.append({"frame":idx,"time_s":round(idx/fps,2),"angle":value,"ml_prediction":label,**f})
                    cv2.putText(frame,f"REPS: {reps}",(15,35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),4)
                    cv2.putText(frame,f"REPS: {reps}",(15,35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
                writer.write(frame)
                if idx % 20 == 0: bar.progress(min(1.0,idx/max(1,min(total,1800))))
        cap.release(); writer.release(); bar.progress(1.0)
        if idx == 0:
            st.error("動画フレームを読み取れませんでした。")
            st.stop()
        if valid_frames == 0:
            st.warning("関節を検出できませんでした。全身が映る動画で再試行してください。")
        else:
            st.subheader("解析結果")
            a,b,c=st.columns(3)
            a.metric("推定回数",reps)
            b.metric("姿勢検出率",f"{valid_frames/idx*100:.0f}%")
            c.metric("解析フレーム",idx)
            st.caption("回数は関節角度の閾値による推定です。ベンチプレスは横方向からの映像で精度が変動します。")
            if model is not None:
                labels=[r["ml_prediction"] for r in rows if r["ml_prediction"] is not None]
                if labels:
                    st.write("**機械学習による最多判定ラベル:**",pd.Series(labels).mode().iloc[0])
                    st.caption("ラベルの意味は学習データの定義に従います。")
            st.write("**角度解析の参考:**",f"設定した動作角度帯に入ったフレームの割合 {good_frames/valid_frames*100:.0f}%")
            st.caption("この割合はフォーム点数ではありません。学習済み正誤分類器の検証が済むまで100点評価は行いません。")
            st.line_chart(pd.DataFrame(rows).set_index("time_s")["angle"])
            st.download_button("解析データ（CSV）",pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),file_name="pose_analysis.csv",mime="text/csv")
        with open(dst,"rb") as file:
            st.download_button("骨格表示動画（MP4）",file.read(),file_name="annotated.mp4",mime="video/mp4")
        st.caption("最大1800フレームを解析します。動画形式によっては出力MP4をブラウザで再生できない場合があります。")
