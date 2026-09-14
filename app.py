import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import shap


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND PREPROCESSOR
# =========================================================

@st.cache_resource
def load_model():

    preprocessor = joblib.load("preprocessor.pkl")

    model = xgb.XGBClassifier()
    model.load_model("xgboost_model.json")

    return preprocessor, model


preprocessor, model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict the probability of customer churn using "
    "a trained XGBoost machine learning model."
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col3:
    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )


# =========================================================
# USAGE INFORMATION
# =========================================================

st.subheader("📱 Usage Information")

col1, col2, col3 = st.columns(3)

with col1:
    usage_frequency = st.number_input(
        "Usage Frequency",
        min_value=0,
        max_value=100,
        value=15,
        step=1
    )

with col2:
    support_calls = st.number_input(
        "Support Calls",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

with col3:
    payment_delay = st.number_input(
        "Payment Delay",
        min_value=0,
        max_value=100,
        value=10,
        step=1
    )


# =========================================================
# SUBSCRIPTION INFORMATION
# =========================================================

st.subheader("💳 Subscription Information")

col1, col2, col3 = st.columns(3)

with col1:
    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

with col2:
    contract_length = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )

with col3:
    total_spend = st.number_input(
        "Total Spend",
        min_value=0.0,
        value=500.0,
        step=50.0
    )


last_interaction = st.number_input(
    "Last Interaction",
    min_value=0,
    max_value=100,
    value=15,
    step=1
)


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    avg_spend_per_tenure = float(
        total_spend / (tenure + 1)
    )

    support_call_rate = float(
        support_calls / (tenure + 1)
    )

    payment_delay_rate = float(
        payment_delay / (tenure + 1)
    )

    engagement_score = float(
        usage_frequency + last_interaction
    )


    # =====================================================
    # CREATE INPUT DATA
    # =====================================================

    input_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "Tenure": [tenure],

        "Usage Frequency": [usage_frequency],

        "Support Calls": [support_calls],

        "Payment Delay": [payment_delay],

        "Subscription Type": [subscription_type],

        "Contract Length": [contract_length],

        "Total Spend": [total_spend],

        "Last Interaction": [last_interaction],

        "Avg_Spend_Per_Tenure": [
            avg_spend_per_tenure
        ],

        "Support_Call_Rate": [
            support_call_rate
        ],

        "Payment_Delay_Rate": [
            payment_delay_rate
        ],

        "Engagement_Score": [
            engagement_score
        ]
    })


    # =====================================================
    # PREPROCESS INPUT
    # =====================================================

    input_processed = preprocessor.transform(
        input_data
    )


    # =====================================================
    # GET FEATURE NAMES
    # =====================================================

    try:

        feature_names = (
            preprocessor.get_feature_names_out()
        )

    except:

        feature_names = [
            f"Feature_{i}"
            for i in range(
                input_processed.shape[1]
            )
        ]


    # =====================================================
    # CONVERT PROCESSED DATA
    # =====================================================

    if hasattr(input_processed, "toarray"):

        input_processed_array = (
            input_processed.toarray()
        )

    else:

        input_processed_array = input_processed


    input_processed_df = pd.DataFrame(
        input_processed_array,
        columns=feature_names
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = int(
        model.predict(input_processed)[0]
    )

    probability = float(
        model.predict_proba(
            input_processed
        )[0][1]
    )

    churn_probability = float(
        probability * 100
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.subheader("📈 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to CHURN"
            )

        else:

            st.success(
                "✅ Customer is unlikely to CHURN"
            )


    with col2:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.4f}%"
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    progress_value = float(
        min(
            max(
                churn_probability / 100,
                0.0
            ),
            1.0
        )
    )

    st.progress(progress_value)


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if churn_probability >= 70:

        risk = "🔴 High Risk"

    elif churn_probability >= 40:

        risk = "🟠 Medium Risk"

    else:

        risk = "🟢 Low Risk"


    st.metric(
        "Customer Risk Level",
        risk
    )


    # =====================================================
    # BUSINESS RECOMMENDATION
    # =====================================================

    st.subheader("💡 Recommended Action")

    if churn_probability >= 70:

        st.warning(
            "High-risk customer: consider a retention offer, "
            "priority support, personalized discount, or "
            "direct customer outreach."
        )

    elif churn_probability >= 40:

        st.info(
            "Medium-risk customer: monitor customer engagement, "
            "provide proactive support, and consider a targeted "
            "retention campaign."
        )

    else:

        st.success(
            "Low-risk customer: continue regular engagement, "
            "loyalty activities, and personalized customer service."
        )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.subheader("📋 Customer Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Age",
            int(age)
        )

    with col2:

        st.metric(
            "Tenure",
            int(tenure)
        )

    with col3:

        st.metric(
            "Usage Frequency",
            int(usage_frequency)
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Support Calls",
            int(support_calls)
        )

    with col2:

        st.metric(
            "Payment Delay",
            int(payment_delay)
        )

    with col3:

        st.metric(
            "Total Spend",
            f"₹{float(total_spend):,.0f}"
        )


    # =====================================================
    # DERIVED METRICS
    # =====================================================

    st.subheader("📊 Derived Customer Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Avg Spend / Tenure",
            f"{avg_spend_per_tenure:.2f}"
        )

    with col2:

        st.metric(
            "Support Call Rate",
            f"{support_call_rate:.2f}"
        )

    with col3:

        st.metric(
            "Payment Delay Rate",
            f"{payment_delay_rate:.2f}"
        )

    with col4:

        st.metric(
            "Engagement Score",
            f"{engagement_score:.2f}"
        )


    # =====================================================
    # SHAP EXPLAINABILITY
    # =====================================================

    st.divider()

    st.subheader(
        "🔎 Why did the model make this prediction?"
    )

    st.write(
        "SHAP explains which features had the strongest "
        "influence on this customer's churn prediction."
    )


    try:

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(
            input_processed_df
        )


        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            shap_values = shap_values[1]


        if len(shap_values.shape) == 2:

            shap_values = shap_values[0]


        # =================================================
        # CLEAN FEATURE NAMES
        # =================================================

        clean_feature_names = []

        for name in feature_names:

            name = str(name)

            # Remove preprocessing prefixes
            name = name.replace(
                "num__",
                ""
            )

            name = name.replace(
                "cat__",
                ""
            )

            # Improve categorical feature names
            name = name.replace(
                "_",
                " — "
            )

            clean_feature_names.append(
                name
            )


        # =================================================
        # CREATE SHAP DATAFRAME
        # =================================================

        shap_importance = pd.DataFrame({

            "Feature": clean_feature_names,

            "SHAP Value": shap_values

        })


        shap_importance[
            "Absolute Impact"
        ] = (
            shap_importance[
                "SHAP Value"
            ].abs()
        )


        # Top 10 features
        shap_importance = (
            shap_importance
            .sort_values(
                "Absolute Impact",
                ascending=False
            )
            .head(10)
        )


        # =================================================
        # SHAP TABLE
        # =================================================

        st.write(
            "Top features influencing this prediction:"
        )

        display_shap = shap_importance[
            [
                "Feature",
                "SHAP Value"
            ]
        ].copy()


        display_shap[
            "SHAP Value"
        ] = display_shap[
            "SHAP Value"
        ].round(4)


        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # FEATURE IMPACT CHART
        # =================================================

        st.write(
            "Feature impact:"
        )

        chart_data = (
            shap_importance
            .set_index(
                "Feature"
            )[
                "SHAP Value"
            ]
            .sort_values()
        )


        st.bar_chart(
            chart_data
        )


        # =================================================
        # TOP FEATURE EXPLANATION
        # =================================================

        top_feature = str(
            shap_importance.iloc[0]["Feature"]
        )

        top_value = float(
            shap_importance.iloc[0]["SHAP Value"]
        )


        if top_value > 0:

            st.warning(
                f"📌 **{top_feature}** had the strongest "
                f"influence toward higher churn risk "
                f"for this customer."
            )

        elif top_value < 0:

            st.success(
                f"📌 **{top_feature}** had the strongest "
                f"influence toward lower churn risk "
                f"for this customer."
            )

        else:

            st.info(
                f"📌 **{top_feature}** had very little "
                f"influence on this prediction."
            )


    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated."
        )

        st.code(
            str(e)
        )


    # =====================================================
    # MODEL INPUT DATA
    # =====================================================

    with st.expander(
        "🔍 View Model Input Data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )