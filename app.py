import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Engine Fault Detection", layout="centered")

# ---------------- HEADER ---------------- #
st.image("https://cdn-icons-png.flaticon.com/512/743/743922.png", width=120)
st.title("🚗 Intelligent Engine Health Monitoring & Fault Diagnosis System")

st.write("Predict engine condition using Temperature (°C), Vibration (mm/s), and Sound (dB).")

st.markdown("---")

# ---------------- LOAD DATA ---------------- #
data = pd.read_excel("engine_fault_dataset_10000.xlsx")

X = data[['Temperature', 'Vibration', 'Sound']]
y = data['Condition']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# ---------------- SIDEBAR INPUT ---------------- #
st.sidebar.title("⚙️ Input Panel")

temperature = st.sidebar.slider("🌡 Temperature (°C)", 60, 120)
vibration = st.sidebar.slider("🔧 Vibration (mm/s)", 0, 100)
sound = st.sidebar.slider("🔊 Sound (dB)", 0, 100)

# ---------------- SYSTEM STATUS ---------------- #
if temperature <= 85 and vibration <= 15 and sound <= 60:
    st.markdown("### 🟢 System Status: HEALTHY")
elif temperature <= 100 and vibration <= 30 and sound <= 80:
    st.markdown("### 🟡 System Status: WARNING")
else:
    st.markdown("### 🔴 System Status: CRITICAL")

# ---------------- INPUT SUMMARY ---------------- #
st.subheader("📋 Input Summary")
st.write(f"Temperature: {temperature} °C")
st.write(f"Vibration: {vibration} mm/s")
st.write(f"Sound: {sound} dB")

# ---------------- ENGINE LOAD ---------------- #
st.subheader("⚙️ Engine Load Indicator")
load = int((temperature + vibration + sound) / 3)
st.progress(load)

st.markdown("---")

# ---------------- LIVE HEALTH STATUS ---------------- #
st.subheader("🩺 Engine Health Status (Live)")

if temperature <= 85 and vibration <= 15 and sound <= 60:
    st.success("🟢 Engine Status: Healthy")
elif temperature <= 100 and vibration <= 30 and sound <= 80:
    st.warning("🟡 Engine Status: Needs Attention")
else:
    st.error("🔴 Engine Status: Critical Condition")

# ---------------- WARNINGS ---------------- #
st.subheader("⚠️ Real-Time Monitoring")

normal = True

if temperature > 100:
    st.error("🔥 High Temperature! Risk of overheating")
    normal = False
elif temperature > 85:
    st.warning("⚠️ Temperature slightly high")
    normal = False

if vibration > 30:
    st.error("🔧 High Vibration! Possible mechanical issue")
    normal = False
elif vibration > 15:
    st.warning("⚠️ Moderate vibration detected")
    normal = False

if sound > 80:
    st.error("🔊 High Noise! Possible engine fault")
    normal = False
elif sound > 60:
    st.warning("⚠️ Noise level increasing")
    normal = False

if normal:
    st.success("✅ All parameters are within normal range")

st.markdown("---")

# ---------------- MODEL ACCURACY ---------------- #
pred_train = model.predict(X)
acc = accuracy_score(y, pred_train)
st.write(f"📈 Model Accuracy: {acc*100:.2f}%")

st.markdown("---")

# ---------------- SESSION ---------------- #
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Predict Engine Condition"):
    prediction = model.predict([[temperature, vibration, sound]])
    probability = model.predict_proba([[temperature, vibration, sound]])

    confidence = max(probability[0]) * 100

    st.subheader("🤖 Prediction Result")

    if prediction[0] == 0:
        st.success(f"✅ Normal Engine ({confidence:.2f}% confident)")
        st.write("✔ No action needed")

    elif prediction[0] == 1:
        st.warning(f"⚠️ Maintenance Required ({confidence:.2f}% confident)")
        st.write("🔧 Recommended Action:")
        st.write("- Schedule maintenance")
        st.write("- Monitor engine regularly")

    else:
        st.error(f"🚨 Fault Detected ({confidence:.2f}% confident)")
        st.write("🔧 Recommended Action:")
        st.write("- Check spark plug")
        st.write("- Inspect fuel system")
        st.write("- Visit service center")

    # ---------------- DIAGNOSIS ---------------- #
    st.subheader("🧠 Diagnosis Explanation")

    if temperature > 100:
        st.write("• Overheating detected")
    if vibration > 30:
        st.write("• Mechanical imbalance")
    if sound > 80:
        st.write("• Abnormal engine noise")

    if temperature <= 85 and vibration <= 15 and sound <= 60:
        st.write("• All parameters are normal")

    # Save history
    st.session_state.history.append([temperature, vibration, sound, prediction[0]])

    # ---------------- GAUGE ---------------- #
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))
    st.plotly_chart(fig)

    # ---------------- DOWNLOAD ---------------- #
    result_df = pd.DataFrame({
        "Temperature": [temperature],
        "Vibration": [vibration],
        "Sound": [sound],
        "Prediction": [prediction[0]]
    })

    st.download_button("📥 Download Report", result_df.to_csv(index=False), "report.csv")

# ---------------- FEATURE IMPORTANCE ---------------- #
st.subheader("📊 Feature Importance")

importance = model.feature_importances_
features = ["Temperature", "Vibration", "Sound"]

df = pd.DataFrame({"Feature": features, "Importance": importance})
st.bar_chart(df.set_index("Feature"))

# ---------------- HISTORY ---------------- #
st.subheader("📜 Prediction History")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history,
                             columns=["Temperature", "Vibration", "Sound", "Prediction"])

    # Convert numbers to labels
    label_map = {0: "Normal", 1: "Maintenance", 2: "Fault"}
    df_history["Prediction"] = df_history["Prediction"].map(label_map)

    st.dataframe(df_history)
else:
    st.info("No predictions yet. Click 'Predict' to see history.")
# ---------------- RESET BUTTON ---------------- #
if st.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.info("💡 Tip: Use dark mode for better experience")
st.write("👩‍💻 Developed by Meera Pandagale")

  
