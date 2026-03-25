import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# Try to import groq, handle gracefully if missing (fallback to template)
try:
    from groq import Groq
except ImportError:
    Groq = None

# Set up the page layout
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide"  
)

st.title("🔮 Telecom Customer Churn Prediction & Analytics")

# Load the model directly
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "gb_churn_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_model()

if pipeline is None:
    st.error(f"Model not found! Make sure the model exists at `models/gb_churn_model.joblib`")
    st.stop()

# Extract preprocessor and model from the pipeline
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]
feature_names = preprocessor.get_feature_names_out()

# Generate the tabs for different features
tab1, tab2 = st.tabs(["🔮 Prediction & AI Retention", "📊 Analytics Dashboard"])

with tab1:
    st.header("Assess Individual Risk")
    st.write("Enter the customer details below to predict their likelihood of churning and see AI-driven retention strategies.")

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
        senior_citizen = st.selectbox("Senior Citizen", options=[0, 1])
        partner = st.selectbox("Partner", options=["Yes", "No"])

    with col2:
        dependents = st.selectbox("Dependents", options=["Yes", "No"])
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method", 
            options=[
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ]
        )
        internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
        paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])

    # Submission button to trigger all the logic
    if st.button("Predict Risk", type="primary"):
        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "InternetService": internet_service,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PaperlessBilling": paperless_billing
        }
        
        df = pd.DataFrame([input_data])
        
        with st.spinner("Analyzing Customer Risk & Generating Insights..."):
            try:
                # 1. Pipeline Prediction
                churn_proba = pipeline.predict_proba(df)[:, 1][0]
                threshold = 0.3 # Business-determined threshold
                churn_pred = int(churn_proba >= threshold)
                
                st.divider()
                res_col1, res_col2 = st.columns([1, 1.5])
                
                with res_col1:
                    st.subheader("Prediction Results")
                    if churn_pred == 1:
                        st.error(f"**🔴 High Risk** (Probability: {churn_proba:.1%})")
                        st.write("This customer is likely to cancel their service.")
                    else:
                        st.success(f"**🟢 Low Risk** (Probability: {churn_proba:.1%})")
                        st.write("This customer is likely to stay.")
                        
                with res_col2:
                    st.subheader("Why? (Risk Drivers)")
                    
                    # Data preprocessing
                    X_preprocessed = preprocessor.transform(df)
                    
                    # Initialize TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    
                    # Compute SHAP values for the specific input
                    shap_values = explainer(X_preprocessed)
                    
                    # Explicitly set feature names for the waterfall plot
                    shap_values.feature_names = list(feature_names)
                    
                    # Waterfall plot
                    fig, ax = plt.subplots(figsize=(6, 4))
                    shape_plot = shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()

                # 2. AI Retention Strategy
                if churn_pred == 1:
                    st.divider()
                    st.subheader("🤖 AI Retention Strategy")
                    st.info("The model indicates high risk. Generating a personalized retention gameplan...")
                    
                    groq_api_key = os.environ.get("GROQ_API_KEY")
                    
                    if Groq and groq_api_key:
                        client = Groq(api_key=groq_api_key)
                        
                        prompt = f"""
                        You are a customer success manager for a telecom company. 
                        A customer is predicted to churn. 
                        Here is their profile:
                        - Tenure: {tenure} months
                        - Monthly Charges: ${monthly_charges}
                        - Contract Type: {contract}
                        - Internet Service: {internet_service}
                        - Payment Method: {payment_method}
                        
                        Write a short, engaging retention email (max 4 sentences) to save this customer, offering a specific, relevant discount or incentive based on their profile. Do not include subject line placeholders, just the raw email body starting with 'Hi there'.
                        """
                        
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are a helpful telecom customer retention expert."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=150,
                            temperature=0.7
                        )
                        
                        email_copy = response.choices[0].message.content
                        st.success("Drafted tailored email using Groq:")
                        st.write(email_copy)
                        
                    else:
                        st.warning("⚠️ GROQ_API_KEY environment variable not set. Falling back to rule-based template.")
                        if contract == "Month-to-month":
                            st.write("**Suggested Action:** Offer $10/mo off their bill if they sign a 1-year contract today.")
                            st.text_area("Template Email:", "Hi there,\n\nWe value your loyalty! To show our appreciation, we'd love to offer you $10 off your monthly bill for the next year if you upgrade to a 1-year contract. Reply 'YES' to claim this offer.\n\nBest,\nTelecom Team")
                        else:
                            st.write("**Suggested Action:** Provide a free 3-month premium channel upgrade to increase engagement.")
                            st.text_area("Template Email:", "Hi there,\n\nWe're gifting you 3 free months of premium sports & movie channels! Click below to activate your free upgrade immediately.\n\nBest,\nTelecom Team")
                            
            except Exception as e:
                st.error(f"An error occurred during prediction/explanation: {e}")

with tab2:
    st.header("📊 Global Feature Importance")
    st.write("This dashboard shows the universal drivers of customer churn across the entire database, helping business teams understand which features matter most over the entirety of the model's history.")
    
    # Extract overall importances from the gradient boosting model inside the pipeline
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True).tail(15)  # Get top 15 most important
    
    # Plot horizontal bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    ax2.set_xlabel("Relative Importance (Gini Importance / Gain)")
    ax2.set_title("Top 15 Most Important Features driving the Churn Model")
    st.pyplot(fig2)
    plt.clf()
