import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.graph_objects as go

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Engine Dashboard", layout="wide")

st.title("🚗 Intelligent Engine Monitoring System")
st.markdown("---")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("⚙ Input Panel")

temp = st.sidebar.slider("🌡 Temperature (°C)", 0, 150, 50)
vib = st.sidebar.slider("🔧 Vibration (mm/s)", 0, 100, 20)
sound = st.sidebar.slider("🔊 Sound (dB)", 0, 120, 30)

# -----------------------------
# MODEL (REALISTIC DATA)
# -----------------------------
data = pd.DataFrame({
    "Temperature": np.random.randint(20, 130, 200),
    "Vibration": np.random.randint(5, 100, 200),
    "Sound": np.random.randint(10, 110, 200),
})

def label_engine(row):
    if row["Temperature"] > 100 or row["Vibration"] > 80 or row["Sound"] > 90:
        return 2
    elif row["Temperature"] > 70 or row["Vibration"] > 50:
        return 1
    else:
        return 0

data["Status"] = data.apply(label_engine, axis=1)

X = data[["Temperature", "Vibration", "Sound"]]
y = data["Status"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

labels = {0: "Normal", 1: "Maintenance Required", 2: "Fault Detected"}

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "count" not in st.session_state:
    st.session_state.count = 0

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "🧠 Diagnosis", "📊 Analytics"])

# -----------------------------
# TAB 1: PREDICTION
# -----------------------------
with tab1:

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{temp} °C")
    col2.metric("Vibration", f"{vib} mm/s")
    col3.metric("Sound", f"{sound} dB")

    # Engine Load
    st.subheader("⚙ Engine Load")
    load = (temp + vib + sound) / 3
    load = min(max(int(load), 0), 100)
    st.progress(load)
    st.write(f"Load: {load}%")

    # Health Score
    health_score = 100 - load
    st.subheader("💚 Engine Health Score")
    st.metric("Health", f"{health_score}%")

    # Risk Level
    if load < 40:
        risk = "Low Risk"
        st.success("🟢 Overall Status: Healthy")
    elif load < 70:
        risk = "Medium Risk"
        st.warning("🟡 Overall Status: Moderate")
    else:
        risk = "High Risk"
        st.error("🔴 Overall Status: Critical")

    st.write(f"⚠ Risk Level: {risk}")

    # Parameter Check
    st.subheader("📊 Parameter Check")
    st.write(f"Temperature: {temp} / 85 (Safe)")
    st.write(f"Vibration: {vib} / 15 (Safe)")
    st.write(f"Sound: {sound} / 60 (Safe)")

    # Predict Button
    if st.button("🔍 Analyze Engine"):

        pred = model.predict([[temp, vib, sound]])[0]
        conf = max(model.predict_proba([[temp, vib, sound]])[0]) * 100
        label = labels[pred]

        st.session_state.last_pred = pred
        st.session_state.last_label = label
        st.session_state.last_conf = conf
        st.session_state.count += 1

        time_now = datetime.now().strftime("%d-%b %H:%M:%S")

        st.session_state.history.append(
            [time_now, temp, vib, sound, label, round(conf, 2)]
        )

        # Result
        st.subheader("🧠 Prediction Result")

        if pred == 0:
            st.success(f"✔ {label} ({conf:.2f}%)")
        elif pred == 1:
            st.warning(f"⚠ {label} ({conf:.2f}%)")
        else:
            st.error(f"❌ {label} ({conf:.2f}%)")
            st.error("🚨 ALERT: Immediate attention required!")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={'text': "Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig)

    # Last Prediction
    if "last_label" in st.session_state:
        st.info(f"Last Prediction: {st.session_state.last_label}")

    # Prediction Count
    st.write(f"Total Predictions: {st.session_state.count}")

# -----------------------------
# TAB 2: DIAGNOSIS
# -----------------------------
with tab2:

    st.subheader("🔍 Diagnosis")

    if "last_pred" in st.session_state:

        if temp > 100:
            st.error("🔥 Overheating Issue")
        if vib > 80:
            st.error("⚙ Mechanical Imbalance")
        if sound > 90:
            st.error("🔊 Noise Problem")

        st.subheader("🛠 Recommended Action")

        if st.session_state.last_pred == 2:
            st.write("• Check spark plug")
            st.write("• Inspect fuel system")
            st.write("• Visit service center")
        elif st.session_state.last_pred == 1:
            st.write("• Schedule maintenance")
        else:
            st.write("• System is working normally")

    else:
        st.info("Run prediction first")

# -----------------------------
# TAB 3: ANALYTICS
# -----------------------------
with tab3:

    st.subheader("📜 Prediction History")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history,
            columns=["Time", "Temp", "Vibration", "Sound", "Prediction", "Confidence"]
        )

        st.dataframe(df, use_container_width=True)

        # Simple Graph
        st.subheader("📈 Temperature Trend")
        st.line_chart(df.set_index("Time")["Temp"])

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Report", csv, "engine_report.csv")

    else:
        st.info("No predictions yet. Start analyzing engine.")

# -----------------------------
# RESET
# -----------------------------
if st.button("🔄 Reset Data"):
    st.session_state.history = []
    st.session_state.count = 0

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.success("✅ System Active")
st.write("👩‍💻 Developed by Meera Pandagale")
