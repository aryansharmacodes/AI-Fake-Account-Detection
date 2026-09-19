import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Fake Account Detector",
    page_icon="🤖",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fake_influencer_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Model could not be loaded.")
    st.error(f"Model path: {MODEL_PATH}")
    st.error(f"Error: {e}")
    st.stop()

st.title("🤖 Fake Account Detection System")
st.markdown(
    """
    ### AI-Based Fake Account Detection

    This system uses a **Random Forest machine learning model**
    to analyze account-level features and estimate the likelihood
    of an Instagram account being fake.

    **Model:** Random Forest  
    **Features:** 9 account-level features  
    **Output:** Real/Fake prediction, probability and risk level
    """
)
st.write("Enter Instagram account details to estimate whether the account is likely to be Real or Fake.")
st.divider()

st.subheader("📋 Account Details")

followers = st.number_input("👥 Followers", min_value=0, value=100, step=1)
following = st.number_input("➡️ Following", min_value=0, value=100, step=1)
bio_length = st.number_input("📝 Biography Length", min_value=0, value=50, step=1)
media_count = st.number_input("📸 Media / Post Count", min_value=0, value=20, step=1)
profile_pic = st.selectbox("🖼️ Profile Picture?", ["Yes", "No"])
private = st.selectbox("🔒 Private Account?", ["Yes", "No"])
username_digits = st.number_input("🔢 Username Digit Count", min_value=0, value=0, step=1)
username_length = st.number_input("🔤 Username Length", min_value=0, value=10, step=1)

st.divider()

if st.button("🔍 Detect Account", use_container_width=True):
    profile_pic_value = 1 if profile_pic == "Yes" else 0
    private_value = 1 if private == "Yes" else 0
    ratio = followers / (following + 1)

    user_data = pd.DataFrame([{
        "userFollowerCount": followers,
        "userFollowingCount": following,
        "userBiographyLength": bio_length,
        "userMediaCount": media_count,
        "userHasProfilPic": profile_pic_value,
        "userIsPrivate": private_value,
        "usernameDigitCount": username_digits,
        "usernameLength": username_length,
        "follower_following_ratio": ratio
    }])

    try:
        prediction = model.predict(user_data)[0]
        probabilities = model.predict_proba(user_data)[0]
        real_probability = probabilities[0] * 100
        fake_probability = probabilities[1] * 100
    except Exception as e:
        st.error("Prediction failed.")
        st.error(f"Error: {e}")
        st.stop()

    st.divider()
    st.subheader("📊 Result")

    if prediction == 1:
        st.error("🚨 FAKE ACCOUNT DETECTED")
    else:
        st.success("✅ REAL ACCOUNT DETECTED")

    st.subheader("📈 Prediction Probability")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Real Probability", f"{real_probability:.2f}%")

    with col2:
        st.metric("Fake Probability", f"{fake_probability:.2f}%")

    st.subheader("⚠️ Risk Level")

    if fake_probability < 30:
        st.success("🟢 LOW RISK")
    elif fake_probability < 70:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")

    st.subheader("🔍 Why this result?")

    signals = []

    if ratio < 0.5:
        signals.append("⚠️ Very low follower/following ratio")

    if following > followers * 3:
        signals.append("⚠️ Following count is much higher than followers")

    if media_count < 10:
        signals.append("⚠️ Very low media/post count")

    if bio_length < 10:
        signals.append("⚠️ Very short biography")

    if profile_pic_value == 0:
        signals.append("⚠️ Profile picture is missing")

    if username_digits >= 3:
        signals.append("⚠️ Username contains several digits")

    if signals:
        for signal in signals:
            st.warning(signal)
    else:
        st.success("✅ No strong warning signals detected.")

    st.caption("These are account-level indicators used to explain the assessment. They are not proof that an account is fake.")

    st.subheader("📊 Model Feature Importance")

    feature_names = [
        "Followers",
        "Following",
        "Biography Length",
        "Media Count",
        "Profile Picture",
        "Private Account",
        "Username Digits",
        "Username Length",
        "Follower/Following Ratio"
    ]

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.info("Feature importance is not available for this model.")

    st.subheader("📋 Account Summary")

    summary_data = pd.DataFrame({
        "Feature": [
            "Followers",
            "Following",
            "Biography Length",
            "Media Count",
            "Profile Picture",
            "Private Account",
            "Username Digits",
            "Username Length",
            "Follower/Following Ratio"
        ],
        "Value": [
            followers,
            following,
            bio_length,
            media_count,
            profile_pic,
            private,
            username_digits,
            username_length,
            f"{ratio:.2f}"
        ]
    })

    st.dataframe(summary_data, use_container_width=True, hide_index=True)
# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.subheader("📊 Model Performance")

st.caption("Performance measured on the held-out test dataset.")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "99.14%")
    st.metric("Recall", "95.00%")

with col2:
    st.metric("Precision", "100.00%")
    st.metric("F1 Score", "97.44%")

st.info(
    "Final Model: Random Forest"
)
st.divider()
st.caption("⚠️ This system provides an ML-based estimate using the trained dataset. It should not be treated as definitive proof of account authenticity.")
