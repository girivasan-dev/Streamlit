import joblib
import pandas as pd
import streamlit as st

FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)


@st.cache_resource
def load_model(path: str = "model.joblib"):
    return joblib.load(path)


st.title("Would this passenger have survived?")
st.write("An educational model trained on the Titanic passenger list.")

pclass = st.selectbox("Ticket class", options=[1, 2, 3], index=1)
sex = st.radio("Sex", options=["female", "male"], horizontal=True)
age = st.slider("Age", min_value=1, max_value=80, value=30)
fare = st.slider("Fare paid", min_value=0, max_value=500, value=30)

if st.button("Predict"):
    # Friendly input validation before running inference
    if pclass not in (1, 2, 3) or not (1 <= age <= 80) or not (0 <= fare <= 500):
        st.warning("Please choose valid values: Ticket class (1–3), Age (1–80), and Fare (0–500).")
        st.stop()

    try:
        model = load_model("model.joblib")
    except FileNotFoundError:
        st.error("The model file (`model.joblib`) is missing. Train and save it first.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load the model cleanly: {exc}")
        st.stop()

    sex_encoded = 1 if sex == "female" else 0
    row = pd.DataFrame(
        [[pclass, sex_encoded, age, fare]],
        columns=FEATURES,
    )

    try:
        if hasattr(model, "predict_proba"):
            chance = float(model.predict_proba(row)[0][1])
        else:
            chance = float(model.predict(row)[0])
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.stop()

    st.metric("Estimated chance of survival", f"{chance:.0%}")
    st.caption(
        "Model note: Decision tree classifier trained on historical Titanic passenger data "
        "(pclass, sex, age, fare). This is an educational exercise, not real-world advice."
    )