import streamlit as st
import pandas as pd
import joblib

# ===============================
# 1. LOAD ASSETS
# ===============================
model = joblib.load('final_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
ohe_gen = joblib.load('ohe_general.pkl')
ohe_pay = joblib.load('ohe_payment.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# ===============================
# SIDEBAR — About & context
# ===============================
with st.sidebar:
    st.markdown("## 📖 About this app")
    st.markdown("---")

    with st.expander("**What this website does**", expanded=True):
        st.markdown("""
        This is a **Customer Churn Predictor** for telecom/service customers.  
        You enter a customer's profile (demographics, services, contract, charges),  
        and the app predicts the **probability they will churn** (leave the company).  
        Results are shown as **Low / Medium / High risk** with a percentage.
        """)

    with st.expander("**Use cases**"):
        st.markdown("""
        - **Retention teams** — Find high-risk customers and target them with offers or support.
        - **Sales & support** — Prioritize calls and interventions by churn risk.
        - **Marketing** — Design campaigns for at-risk segments (e.g. month-to-month, high charges).
        - **Product** — See which services (e.g. no tech support, no contract) correlate with churn.
        - **Executives** — Track and reduce churn rate using data-driven predictions.
        """)

    with st.expander("**Where the features come from**"):
        st.markdown("""
        The inputs mirror a **telecom/cable-style customer dataset**:
        - **Demographics** — Gender, senior citizen, partner, dependents (from customer records).
        - **Tenure** — How long they've been with the company (months).
        - **Services** — Phone, internet, multiple lines, streaming TV/movies, security, backup, device protection, tech support (from product/subscription data).
        - **Billing** — Contract type, paperless billing, payment method, monthly and total charges (from billing systems).
        The model was trained on a customer churn dataset with these same kinds of features.
        """)

    with st.expander("**Why customer churn matters**"):
        st.markdown("""
        - **Revenue** — Losing a customer means lost recurring revenue and often higher cost to win a new one.
        - **Cost** — Acquiring a new customer is typically 5–7× more expensive than keeping an existing one.
        - **Growth** — High churn makes it harder to grow; reducing churn directly improves retention and LTV.
        - **Signals** — Churn risk highlights product, pricing, or service issues before customers leave.
        Predicting churn lets the business act in time (offers, support, pricing) instead of reacting after loss.
        """)

st.title("📞 Customer Churn Predictor")
# ===============================
# 2. USER INPUTS
# ===============================
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    senior = st.radio("Senior Citizen", ["Yes", "No"])
    partner = st.radio("Partner", ["Yes", "No"])
    dependents = st.radio("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", 0, 72, 12)

    phone = st.radio("Phone Service", ["Yes", "No"])

    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col2:
    backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.radio("Paperless Billing", ["Yes", "No"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    m_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=150.0,
        value=70.0,
        step=0.5
    )

# ===============================
# 3. AUTO-CALCULATE TOTAL CHARGES
# ===============================
t_charges = tenure * m_charges

st.info(f"💰 **Total Charges :** {t_charges:.2f}")

# ===============================
# 4. PREPROCESSING + PREDICTION
# ===============================
if st.button("Predict"):
    data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': protection,
        'TechSupport': support,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': m_charges,
        'TotalCharges': t_charges
    }

    df_input = pd.DataFrame([data])

    # A. Binary Mapping
    df_input['gender'] = df_input['gender'].map({'Male': 0, 'Female': 1})

    for c in ['Partner', 'SeniorCitizen', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df_input[c] = df_input[c].map({'Yes': 1, 'No': 0})

    # B. General OHE
    ohe_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract'
    ]

    gen_enc = ohe_gen.transform(df_input[ohe_cols])
    gen_df = pd.DataFrame(
        gen_enc,
        columns=ohe_gen.get_feature_names_out(ohe_cols),
        index=df_input.index
    )

    # C. Payment OHE
    pay_enc = ohe_pay.transform(df_input[['PaymentMethod']])
    pay_df = pd.DataFrame(
        pay_enc,
        columns=ohe_pay.get_feature_names_out(['PaymentMethod']),
        index=df_input.index
    )

    # D. Merge
    df_final = df_input.drop(columns=ohe_cols + ['PaymentMethod'])
    df_final = pd.concat([df_final, gen_df, pay_df], axis=1)

    # E. Scaling
    df_final[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df_final[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # F. Column Order Safety
    df_final = df_final[model.feature_names_in_]

    # ===============================
    # 5. PREDICT
    # ===============================
    prob = model.predict_proba(df_final)[0][1]

    st.divider()
    if prob >= 0.5:
        st.error(f"HIGH RISK: CHURN ❌ : ({prob:.1%})")
    elif prob >= 0.3:
        st.warning(f"MEDIUM RISK :  ⚠️ : ({prob:.1%})")
    else:
        st.success(f"LOW RISK: STAY ✅ : ({prob:.1%})")