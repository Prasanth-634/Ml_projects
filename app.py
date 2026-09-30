"""
=============================================================================
Bank Marketing Prediction - Streamlit Web Application
=============================================================================
This interactive web app predicts whether a bank customer is likely to
subscribe to a term deposit based on demographic, financial, and campaign data.
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Paths & Constants
# -----------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "bank_model.pkl")
METRICS_PATH = os.path.join("models", "model_metrics.json")
DATA_PATH = os.path.join("data", "bank.csv")

# Standard feature categories derived directly from the Bank Marketing dataset
JOB_OPTIONS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
]
MARITAL_OPTIONS = ["married", "single", "divorced"]
EDUCATION_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
BINARY_OPTIONS = ["no", "yes"]
CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
MONTH_OPTIONS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]
POUTCOME_OPTIONS = ["unknown", "failure", "other", "success"]


# -----------------------------------------------------------------------------
# Helper Functions: Loading Model & Metrics
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Load the pre-trained Logistic Regression pipeline."""
    if not os.path.exists(MODEL_PATH):
        return None, "Model file not found."
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


@st.cache_data
def load_model_metrics():
    """Load precomputed model evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data
def load_dataset():
    """Load raw dataset for dataset overview section."""
    if not os.path.exists(DATA_PATH):
        return None
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            first_line = f.readline()
            sep = ";" if ";" in first_line else ","
        df = pd.read_csv(DATA_PATH, sep=sep)
        return df
    except Exception:
        return None


# -----------------------------------------------------------------------------
# App Header
# -----------------------------------------------------------------------------
st.title("🏦 Bank Marketing Prediction")
st.subheader("Predict whether a customer is likely to subscribe to a term deposit.")
st.markdown("---")

# -----------------------------------------------------------------------------
# Section: About the Project
# -----------------------------------------------------------------------------
with st.expander("📋 About the Project", expanded=False):
    st.info(
        """
        **Project Goal:**
        This project uses machine learning to assist banks in optimizing direct marketing campaigns.
        By analyzing customer demographics, past financial behavior, and recent campaign interactions,
        the model predicts whether a client will subscribe (`yes`) or decline (`no`) a **term deposit**.

        **Key Points:**
        - **Algorithm:** Logistic Regression with Scikit-Learn `ColumnTransformer` Pipeline.
        - **Preprocessing:** One-Hot Encoding for categorical features & Standard Scaling for numerical features.
        - **Dataset:** UCI Bank Marketing dataset based on Portuguese banking institution campaigns.
        """
    )

# -----------------------------------------------------------------------------
# Sidebar: Customer Information Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Customer Information")
st.sidebar.write("Configure the customer and campaign parameters below:")

# Demographic & Personal Profile
st.sidebar.subheader("👤 Demographic Details")
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=35, step=1, help="Age of customer in years")
job = st.sidebar.selectbox("Job", options=JOB_OPTIONS, index=4, help="Type of occupation")
marital = st.sidebar.selectbox("Marital Status", options=MARITAL_OPTIONS, index=0)
education = st.sidebar.selectbox("Education Level", options=EDUCATION_OPTIONS, index=1)

# Financial Information
st.sidebar.subheader("💳 Financial Profile")
default = st.sidebar.selectbox("Credit in Default?", options=BINARY_OPTIONS, index=0, help="Has credit in default?")
balance = st.sidebar.number_input("Average Yearly Balance (€)", min_value=-5000, max_value=100000, value=1500, step=100)
housing = st.sidebar.selectbox("Housing Loan?", options=BINARY_OPTIONS, index=1, help="Has housing loan?")
loan = st.sidebar.selectbox("Personal Loan?", options=BINARY_OPTIONS, index=0, help="Has personal loan?")

# Current Campaign Contact Details
st.sidebar.subheader("📞 Current Campaign Details")
contact = st.sidebar.selectbox("Contact Communication Type", options=CONTACT_OPTIONS, index=0)
month = st.sidebar.selectbox("Last Contact Month", options=MONTH_OPTIONS, index=4)
day = st.sidebar.slider("Last Contact Day of Month", min_value=1, max_value=31, value=15, step=1)
duration = st.sidebar.number_input(
    "Contact Duration (seconds)",
    min_value=0,
    max_value=5000,
    value=220,
    step=10,
    help="Duration of the last contact in seconds"
)
campaign = st.sidebar.number_input(
    "Campaign Contacts",
    min_value=1,
    max_value=60,
    value=1,
    step=1,
    help="Number of contacts performed during this campaign for this client"
)

# Previous Campaign History
st.sidebar.subheader("⏳ Previous Campaign History")
pdays = st.sidebar.number_input(
    "Days Since Last Contact (pdays)",
    min_value=-1,
    max_value=1000,
    value=-1,
    step=1,
    help="Number of days since client was last contacted (-1 means client was not previously contacted)"
)
previous = st.sidebar.number_input(
    "Previous Contacts Count",
    min_value=0,
    max_value=50,
    value=0,
    step=1,
    help="Number of contacts performed before this campaign for this client"
)
poutcome = st.sidebar.selectbox(
    "Outcome of Previous Campaign",
    options=POUTCOME_OPTIONS,
    index=0,
    help="Outcome of the previous marketing campaign"
)

# -----------------------------------------------------------------------------
# Section: Customer Prediction
# -----------------------------------------------------------------------------
st.header("🔮 Customer Prediction")
st.write("Review the customer's input parameters and click **Predict** to evaluate subscription likelihood.")

# Display a preview of the selected customer data
input_dict = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}
input_df = pd.DataFrame([input_dict])

with st.expander("🔍 View Selected Customer Features Table", expanded=False):
    st.dataframe(input_df, use_container_width=True)

# Prediction Button
predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

if predict_btn:
    model, err = load_trained_model()
    if err:
        st.error(f"⚠️ {err} Please make sure to run `python train_model.py` first to generate the model file.")
    else:
        try:
            # 1. Generate prediction
            prediction = model.predict(input_df)[0]

            # 2. Generate probability if model supports it
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                # Classes in model
                classes = list(model.classes_)
                prob_no = probabilities[classes.index("no")] * 100
                prob_yes = probabilities[classes.index("yes")] * 100

            st.write("---")
            st.subheader("Prediction Result")

            if prediction == "yes":
                st.success("### ✅ Customer is likely to subscribe to the term deposit.")
            else:
                st.error("### ❌ Customer is unlikely to subscribe to the term deposit.")

            # Display probabilities
            if probabilities is not None:
                st.write("**Prediction Probability:**")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric(label="Likelihood to Subscribe ('Yes')", value=f"{prob_yes:.2f}%")
                    st.progress(float(prob_yes / 100))
                with prob_col2:
                    st.metric(label="Likelihood NOT to Subscribe ('No')", value=f"{prob_no:.2f}%")
                    st.progress(float(prob_no / 100))

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Section: Model Performance
# -----------------------------------------------------------------------------
st.header("📊 Model Performance")
st.write("Evaluation metrics obtained on the held-out 20% test dataset:")

metrics = load_model_metrics()

if metrics:
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Accuracy", value=f"{metrics['accuracy'] * 100:.2f}%")
    with m_col2:
        st.metric(label="Precision", value=f"{metrics['precision'] * 100:.2f}%")
    with m_col3:
        st.metric(label="Recall", value=f"{metrics['recall'] * 100:.2f}%")
    with m_col4:
        st.metric(label="F1 Score", value=f"{metrics['f1_score'] * 100:.2f}%")

    st.write("")
    st.subheader("Confusion Matrix")

    # Plot Confusion Matrix with Seaborn / Matplotlib
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
        ax=ax,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_title("Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("⚠️ Model performance metrics not found. Please run `python train_model.py` to evaluate and cache metrics.")

st.markdown("---")

# -----------------------------------------------------------------------------
# Section: Dataset Overview
# -----------------------------------------------------------------------------
st.header("📈 Dataset Overview")

df = load_dataset()

if df is not None:
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        st.metric(label="Total Customers (Rows)", value=f"{df.shape[0]:,}")
    with d_col2:
        st.metric(label="Total Features (Columns)", value=f"{df.shape[1]}")
    with d_col3:
        yes_count = int((df["y"] == "yes").sum())
        total = len(df)
        pct = (yes_count / total) * 100
        st.metric(label="Subscribed Rate ('Yes')", value=f"{pct:.2f}% ({yes_count} clients)")

    st.write("")
    st.subheader("Target Distribution (Subscribed vs Did Not Subscribe)")

    # Bar chart for Target Distribution
    y_counts = df["y"].value_counts().rename({"no": "No (Declined)", "yes": "Yes (Subscribed)"})

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    colors = ["#4A90E2", "#50E3C2"]
    sns.barplot(x=y_counts.index, y=y_counts.values, palette=colors, ax=ax2, hue=y_counts.index, legend=False)
    ax2.set_ylabel("Number of Customers", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Term Deposit Subscription ('y')", fontsize=10, fontweight="bold")
    ax2.set_title("Class Balance in Bank Marketing Dataset", fontsize=12, fontweight="bold", pad=10)

    # Add count labels on top of bars
    for idx, count in enumerate(y_counts.values):
        ax2.text(idx, count + 50, f"{count} ({count/total*100:.1f}%)", ha="center", fontweight="bold", fontsize=10)

    ax2.set_ylim(0, max(y_counts.values) * 1.15)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("👀 View Sample Dataset Records", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

else:
    st.warning("⚠️ Dataset not found at `data/bank.csv`. Please make sure the file is placed in the `data/` folder.")

st.markdown("---")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em; padding: 15px 0;">
        <strong>Bank Marketing Prediction | Machine Learning + Streamlit</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
