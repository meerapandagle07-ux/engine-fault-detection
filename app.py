import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Engine AI Dashboard", layout="wide")

st.title("🚗 AI-Based Engine Monitoring Dashboard")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙ Control Panel")

mode = st.sidebar.radio("Select Mode", ["Manual Input", "Live Simulation"])

# -----------------------------
# INPUTS
# -----------------------------
if mode == "Manual Input":
    temp = st.sidebar.slider("Temperature (°C)", 0, 150, 50)
    vib = st.sidebar.slider("Vibration (mm/s)", 0, 100, 20)
    sound = st.sidebar.slider("Sound (dB)", 0, 120, 30)
else:
    temp = np.random.randint(30, 130)
    vib = np.random.randint(10, 100)
    sound = np.random.randint(20, 110)
    st.sidebar.write("Live values updating...")

# -----------------------------
# DISPLAY INPUTS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🌡 Temperature", f"{temp} °C")
col2.metric("📳 Vibration", f"{vib} mm/s")
col3.metric("🔊 Sound", f"{sound} dB")

# -----------------------------
# ENGINE LOAD
# -----------------------------
st.subheader("⚙ Engine Load")

load = (temp + vib + sound) / 3
load = min(max(int(load), 0), 100)

st.progress(load)
st.write(f"Load: {load}%")

# -----------------------------
# DATASET (REALISTIC)
# -----------------------------
data = pd.DataFrame({
    "Temperature": np.random.randint(20, 130, 300),
    "Vibration": np.random.randint(5, 100, 300),
    "Sound": np.random.randint(10, 110, 300),
})

def label_engine(row):
    if row["Temperature"] > 100 or row["Vibration"] > 80 or row["Sound"] > 90:
        return 2
    elif row["Temperature"] > 70 or row["Vibration"] > 50:
        return 1
    else:
        return 0

data["Status"] = data.apply(label_engine, axis=1)

# -----------------------------
# MODEL
# -----------------------------
X = data[["Temperature", "Vibration", "Sound"]]
y = data["Status"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

labels = {0: "Normal", 1: "Maintenance", 2: "Fault"}

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# PREDICT
# -----------------------------
if st.button("🔍 Analyze Engine"):

    pred = model.predict([[temp, vib, sound]])[0]
    conf = max(model.predict_proba([[temp, vib, sound]])[0]) * 100

    label = labels[pred]
    time_now = datetime.now().strftime("%d-%b %H:%M:%S")

    st.session_state.history.append(
        [time_now, temp, vib, sound, label, round(conf, 2)]
    )

    st.subheader("🧠 AI Result")

    if pred == 0:
        st.success(f"{label} ({conf:.2f}%)")
    elif pred == 1:
        st.warning(f"{label} ({conf:.2f}%)")
    else:
        st.error(f"{label} ({conf:.2f}%)")

    # Diagnosis
    st.subheader("🔍 Diagnosis")

    if temp > 100:
        st.write("🔥 Overheating detected")
    if vib > 80:
        st.write("⚙ Mechanical imbalance")
    if sound > 90:
        st.write("🔊 Noise abnormality")

# -----------------------------
# HISTORY
# -----------------------------
st.subheader("📜 History")

if st.session_state.history:
    df = pd.DataFrame(
        st.session_state.history,
        columns=["Time", "Temp", "Vibration", "Sound", "Prediction", "Confidence"]
    )

    st.dataframe(df)

    # -----------------------------
    # CHARTS
    # -----------------------------
    st.subheader("📈 Trends")
    st.line_chart(df[["Temp", "Vibration", "Sound"]])

    # -----------------------------
    # FAULT DISTRIBUTION
    # -----------------------------
    st.subheader("📊 Fault Distribution")

    chart_data = df["Prediction"].value_counts()
    st.bar_chart(chart_data)

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Report", csv, "report.csv")

else:
    st.write("No data yet")

# -----------------------------
# AUTO REFRESH (LIVE MODE)
# -----------------------------
if mode == "Live Simulation":
    time.sleep(2)
    st.rerun()

# -----------------------------
# RESET
# -----------------------------
if st.button("🔄 Reset Data"):
    st.session_state.history = []

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.success("✅ AI Powered Monitoring Active")
st.write("👩‍💻 Developed by Meera Pandagale")
