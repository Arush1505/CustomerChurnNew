# 📡 Telecom Customer Churn Prediction Pipeline

ChurnGuard is an end-to-end Automated Machine Learning (AutoML) application designed to predict customer churn with high sensitivity. Built with Python, XGBoost, and Streamlit, it features a robust preprocessing pipeline that standardizes data, handles class imbalance, and provides actionable risk reports.

---

## 🚀 Key Features

- ** Preprocessing** 
  Includes a Missing Value Gate and data type correction (specifically for TotalCharges strings).

- **Encoding**  
  Uses a uniqueness threshold to dynamically choose between Label Encoding and One-Hot Encoding for categorical features.

- **Recall-Optimized Model**  
  Gradient Boosting architecture tuned to achieve approximately 89% Recall, ensuring the business captures the maximum number of at-risk customers.

- **Probability Report**  
  Generates a detailed churn probability breakdown for every customer, categorized into High Risk and Stable tiers.

---

## 🛠️ Technical Workflow

The pipeline strictly follows a Split → Scale → Resample strategy to prevent data leakage and ensure model reliability.

- **Standardization**  
  Input data must contain churn and id columns for pipeline initialization.

- **Scaling**  
  Uses a pre-trained StandardScaler to normalize numeric features:
  - tenure  
  - MonthlyCharges  
  - TotalCharges  

- **Resampling**  
  Implements SMOTE-ENN to balance the dataset and improve learning on the minority Churn class.

- **Deployment**  
  State-managed Streamlit interface supporting:
  - Batch Processing (Original dataset file)
  - Single Customer Prediction (Sidebar input)

---

## 📂 Repository Structure

    ├── app.py                         # Main Streamlit application logic
    ├── gradient_boosting_model.joblib # Trained Gradient Boosting model
    ├── scaler.joblib                  # Pre-trained StandardScaler
    ├── model_columns.joblib           # Saved feature names for input alignment
    ├── requirements.txt               # Environment dependencies
    └── README.md                # Project documentation

---

## 💻 Installation & Usage

### Clone the Repository

    git clone https://github.com/YourUsername/YourRepoName.git
    cd YourRepoName

### Install Dependencies

    pip install -r requirements.txt

### Run the Application

    streamlit run app.py

---

## 📊 Performance Benchmarks

- **Recall:** ~89% (Primary metric for churn detection)  
- **Accuracy:** ~72%  
- **Precision:** ~48%  
- **F1 Score:** ~62%
