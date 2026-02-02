import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score

# --- CONFIG & STYLING ---
st.set_page_config(page_title="ChurnGuard: AutoML Pipeline", layout="wide")
st.download_button(label="You can download dataset from here",data="csv",file_name="churn dataset.csv")
# --- CORE PIPELINE FUNCTIONS ---

def clean_and_prepare(df):
    """Missing value gate and data type correction"""
    df_clean = df.copy()
    if 'customerID' in df_clean.columns:
        df_clean.drop("customerID", axis=1, inplace=True)
    
    # Data type correction for 'TotalCharges'
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce").fillna(0)
    
    # Missing value gate: Simple imputer
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

def auto_encode(df, threshold=10):
    """Uniqueness threshold for Encoding logic"""
    # Map obvious binary strings
    mapdict = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}
    for col in df.select_dtypes(include=['object']).columns:
        if len(df[col].unique()) == 2:
            df[col] = df[col].map(mapdict)
            
    # One-Hot Encoding for remaining categories under threshold
    return pd.get_dummies(df, drop_first=True, dtype=int)

def visualize_modular(df_encoded, df_original):
    """Identify top 10 correlated features and plot"""
    st.subheader("📊 Automated Exploratory Analysis")
    
    # Heatmap of top 10 correlations
    corr = df_encoded.corr()['Churn'].sort_values(ascending=False).head(11) # Churn + top 10
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded[corr.index].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_heat)
    st.pyplot(fig_heat)
    
    # Distribution of Churn
    col1, col2 = st.columns(2)
    with col1:
        fig_count, ax_count = plt.subplots()
        sns.countplot(x='Contract', hue='Churn', data=df_original, palette='magma', ax=ax_count)
        st.pyplot(fig_count)
    with col2:
        fig_kde, ax_kde = plt.subplots()
        sns.kdeplot(df_original[df_original['Churn'] == 'Yes']['tenure'], label='Churn: Yes', fill=True, ax=ax_kde)
        sns.kdeplot(df_original[df_original['Churn'] == 'No']['tenure'], label='Churn: No', fill=True, ax=ax_kde)
        plt.title('Tenure Density')
        st.pyplot(fig_kde)

# --- APP LAYOUT ---

st.title("📡 ChurnGuard AutoML Pipeline")
st.sidebar.header("User Control Panel")

# Load Model
try:
    model = joblib.load("gradient_boosting_model.joblib")
    scaler = joblib.load("scaler.joblib")
    model_columns = joblib.load('model_columns.joblib')
except:
    st.error("Model file not found. Please ensure 'gradient_boosting_model.joblib' is in the directory.")



btn = st.button(label="START PIPELINE")
if btn:
    df = pd.read_csv("churn dataset.csv")
    st.write("### Data Preview", df.head())
    
    df_clean = clean_and_prepare(df)
    df_encoded = auto_encode(df_clean)
    
    # Visualization
    visualize_modular(df_encoded, df)
    
    # Split -> Scale -> Resample logic
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scaler = StandardScaler()
    X_test_ready = X_test.reindex(columns=model_columns, fill_value=0)
    
    huge_cols = ["tenure", "TotalCharges", "MonthlyCharges"]
    X_test_ready[huge_cols] = scaler.transform(X_test_ready[huge_cols])
    
    y_pred = model.predict(X_test_ready)
    probs = model.predict_proba(X_test_ready)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1= f1_score(y_test,y_pred)
    
    st.write("### 📏 Model Performance Metrics")
    m1, m2, m3,m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall", f"{rec:.2%}")
    m4.metric("F1 Score", f"{f1:.2%}")
    
    # Probability Report
    probs = model.predict_proba(X_test_ready)[:, 1]
    results = pd.DataFrame({
            'Actual_Churn': y_test.values,
            'Churn_Probability': probs
        }, index=y_test.index) # Keeping the original ID/Index
    
    results['Risk_Status'] = results['Churn_Probability'].apply(
            lambda x: 'High Risk ⚠️' if x >= 0.5 else 'Stable ✅'
        )
    
    st.write("### 📈 Probability Report")
    st.dataframe(results.sort_values(by='Churn_Probability', ascending=False), use_container_width=True)
    st.write("### ⚖️ Prediction vs. Reality Comparison")
    
    # Creating a summary table for the plot
    comparison_df = results.groupby(['Risk_Status', 'Actual_Churn']).size().reset_index(name='Count')
    
    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    sns.barplot(data=comparison_df, x='Risk_Status', y='Count', hue='Actual_Churn', palette='viridis', ax=ax_comp)
    plt.title("Volume of Predicted Risk Levels vs. Actual Churn Labels")
    st.pyplot(fig_comp)
        

# 2. MANUAL INPUT (SIDEBAR)
st.sidebar.markdown("---")
st.sidebar.subheader("Single Customer Prediction")

with st.sidebar.form("manual_entry"):
    # Placeholder fields based on your dataset columns
    gender = st.selectbox("Gender", ["Female", "Male"])
    sc = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.slider("Tenure", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    m_charges = st.number_input("Monthly Charges", value=50.0)
    t_charges = st.number_input("Total Charges", value=500.0)
    
    submit_manual = st.form_submit_button("Predict Churn")

if submit_manual:
    st.sidebar.metric("Churn Risk", "High" if m_charges > 50 else "Stable")