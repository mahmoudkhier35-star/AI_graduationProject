from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_pipeline.joblib"

st.set_page_config(page_title="Customer Churn Prediction")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(155deg, #eaf6f0, #f7faf9 38rem);
    }
    [data-testid="stHeader"] {
        background: #eef7f2;
        color: #24473b;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 62rem;
        padding-top: 4rem;
    }
    [data-testid="stMain"] h1 {
        color: #173c31;
        font-size: clamp(2rem, 5vw, 3rem);
        letter-spacing: -0.04em;
    }
    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #d7e7df;
        border-radius: 1.1rem;
        padding: 1.65rem;
        box-shadow: 0 12px 32px #1a413411;
    }
    [data-testid="stForm"] h3 {
        color: #173c31;
        font-size: 1.2rem;
    }
    [data-testid="stForm"] [data-testid="stWidgetLabel"] {
        color: #24473b;
    }
    [data-testid="stForm"] [data-testid="stExpander"] details {
        border-color: #d7e7df;
    }
    [data-testid="stForm"] [data-testid="stExpander"] summary,
    [data-testid="stForm"] [data-testid="stCaptionContainer"] {
        color: #3d6053;
    }
    [data-testid="stForm"] [data-testid="stCaptionContainer"] {
        opacity: 1;
    }
    [data-testid="stForm"] [data-testid="stExpander"] summary:hover {
        background-color: #f4f9f6;
        color: #173c31;
    }
    [data-testid="stCheckbox"] label > span + div {
        background-color: #ffffff;
        border-color: #79998d;
    }
    [data-testid="stCheckbox"] label:has(input:checked) > span + div {
        background-color: #176650;
        border-color: #176650;
    }
    [data-testid="stForm"] [data-baseweb="select"] > div,
    [data-testid="stForm"] [data-baseweb="input"],
    [data-testid="stSelectbox"] [role="group"],
    [data-testid="stNumberInputContainer"] {
        background-color: #f4f9f6;
        border-color: #d7e7df;
        color: #203d34;
    }
    [data-testid="stForm"] [data-baseweb="base-input"],
    [data-testid="stForm"] input {
        background-color: transparent;
        color: #203d34;
    }
    [data-testid="stForm"] input::placeholder {
        color: #527064;
    }
    [data-testid="stSelectbox"] [role="group"] button {
        background-color: transparent;
        color: #24473b;
    }
    [data-testid="stForm"] [data-testid="stNumberInput"] button {
        background-color: #eaf3ee;
        color: #24473b;
    }
    [data-testid="stForm"] :focus-within > [data-baseweb="input"],
    [data-testid="stForm"] [data-baseweb="select"]:focus-within > div,
    [data-testid="stSelectbox"] [role="group"]:focus-within,
    [data-testid="stNumberInputContainer"]:focus-within {
        border-color: #26765e;
    }
    [role="listbox"] {
        background-color: #ffffff;
        color: #203d34;
        color-scheme: light;
    }
    [role="listbox"] [role="option"] {
        background-color: #ffffff;
        color: #203d34;
    }
    [role="listbox"] [role="option"]:hover,
    [role="listbox"] [role="option"][data-focused],
    [role="listbox"] [role="option"][aria-selected="true"] {
        background-color: #e4f3eb;
    }
    [data-testid="stFormSubmitButton"] button {
        background: #176650;
        border-color: #176650;
        color: #ffffff;
        padding: 0.55rem 1.4rem;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #104c3c;
        border-color: #104c3c;
        color: #ffffff;
    }
    [data-testid="stFormSubmitButton"] button:focus-visible {
        outline: 2px solid #176650;
        outline-offset: 3px;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7e7df;
        border-radius: 0.85rem;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] {
        color: #173c31;
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding-top: 3rem; }
        [data-testid="stForm"] { padding: 1rem; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-direction: column; }
        [data-testid="stForm"] [data-testid="stColumn"] { width: 100%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Customer Churn Prediction")

if not MODEL_PATH.is_file():
    st.error("Model file missing. Save churn_pipeline.joblib in the models folder beside app.py.")
    st.stop()

yes_no = ["No", "Yes"]

with st.form("customer_form"):
    customer_column, services_column = st.columns(2)

    with customer_column:
        st.subheader("Customer and billing")
        customer = {
            "tenure": st.number_input("Tenure (months)", min_value=0, value=12),
            "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
            "MonthlyCharges": st.number_input("Monthly charges", min_value=0.0, value=65.0),
        }

    with services_column:
        st.subheader("Services")
        services = {
            "InternetService": st.selectbox("Internet service", ["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": st.selectbox("Online security", yes_no),
            "TechSupport": st.selectbox("Tech support", yes_no),
        }

    with st.expander("Additional details"):
        billing_details, service_details = st.columns(2)

        with billing_details:
            customer.update({
                "gender": st.selectbox("Gender", ["Female", "Male"]),
                "SeniorCitizen": int(st.checkbox("Senior citizen")),
                "Partner": st.selectbox("Partner", yes_no),
                "Dependents": st.selectbox("Dependents", yes_no),
                "PaperlessBilling": st.selectbox("Paperless billing", yes_no),
                "PaymentMethod": st.selectbox(
                    "Payment method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                ),
                "TotalCharges": st.number_input(
                    "Total charges",
                    min_value=0.0,
                    value=None,
                    placeholder="Leave blank if unknown",
                ),
            })

        with service_details:
            services.update({
                "PhoneService": st.selectbox("Phone service", ["Yes", "No"]),
                "MultipleLines": st.selectbox("Multiple lines", yes_no),
                "OnlineBackup": st.selectbox("Online backup", yes_no),
                "DeviceProtection": st.selectbox("Device protection", yes_no),
                "StreamingTV": st.selectbox("Streaming TV", yes_no),
                "StreamingMovies": st.selectbox("Streaming movies", yes_no),
            })

    st.caption("Additional details use their defaults unless you change them.")
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    if services["PhoneService"] == "No":
        services["MultipleLines"] = "No phone service"

    if services["InternetService"] == "No":
        for service in (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ):
            services[service] = "No internet service"

    customer.update(services)
    customer_data = pd.DataFrame([customer])
    customer_data["TotalCharges"] = pd.to_numeric(customer_data["TotalCharges"], errors="coerce")

    pipeline = joblib.load(MODEL_PATH)
    prediction = pipeline.predict(customer_data)[0]
    churn_probability = pipeline.predict_proba(customer_data)[0, 1]

    if prediction == 1:
        st.warning("Prediction: Customer may leave")
    else:
        st.success("Prediction: Customer likely to stay")

    st.metric("Estimated churn probability", f"{churn_probability:.1%}")
