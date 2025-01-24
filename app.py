import streamlit as st
import numpy as np
import joblib

# モデルの読み込み
model = joblib.load("linear_model.pkl")

# Streamlit アプリのタイトル
st.title("機械学習モデルのデプロイ with Streamlit")

# 説明文
st.write("このアプリでは、学習済みの線形回帰モデルを使って予測を行います。")

# 入力フォーム
st.header("入力値を指定してください")
input_value = st.number_input("値を入力してください (例: 5.0)", min_value=0.0, max_value=10.0, step=0.1)

# ボタンで予測を実行
if st.button("予測実行"):
    # モデルで予測
    prediction = model.predict(np.array([[input_value]]))
    st.write(f"入力値: {input_value}")
    st.write(f"予測結果: {prediction[0][0]}")

# デモ用の説明
st.write("このアプリは Streamlit を使って作成されています。")
