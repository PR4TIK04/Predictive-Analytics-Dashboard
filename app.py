import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="ML Dashboard", layout="wide")

# ===========================
# STEP 1: UPLOAD DATA
# ===========================
st.title("📊 Predictive Analytics Dashboard")
st.markdown("Upload your dataset and explore insights, build ML models, and explain predictions!")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Dataset Summary")
    st.write(df.describe())

    # ===========================
    # STEP 2: EDA
    # ===========================
    st.markdown("---")
    st.header("🔍 Exploratory Data Analysis (EDA)")

    option = st.selectbox("Choose Analysis Type", ["Dataset Overview", "Profiling Report"])

    if option == "Dataset Overview":
        col = st.selectbox("Select a Column for Visualization", df.columns)
        st.bar_chart(df[col].value_counts())

    elif option == "Profiling Report":
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)

    # ===========================
    # STEP 3: DATA VISUALIZATION
    # ===========================
    st.markdown("---")
    st.header("📈 Data Visualization")

    x_axis = st.selectbox("Select X-axis", df.columns)
    y_axis = st.selectbox("Select Y-axis", df.columns)
    graph_type = st.selectbox("Select Graph Type", ["Scatter", "Line", "Bar", "Box", "Histogram"])

    if graph_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis)
    elif graph_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis)
    elif graph_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis)
    elif graph_type == "Box":
        fig = px.box(df, x=x_axis, y=y_axis)
    elif graph_type == "Histogram":
        fig = px.histogram(df, x=x_axis)
    st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # STEP 4: AUTOMATED ML
    # ===========================
    st.markdown("---")
    st.header("🤖 Automated ML & Predictions")

    target_col = st.selectbox("📌 Select Target Column", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X.select_dtypes(include=[np.number])),
                         columns=X.select_dtypes(include=[np.number]).columns)
        X = pd.get_dummies(X, drop_first=True)

        # Encode target if categorical
        if y.dtypes == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Detect problem type
        problem_type = "Regression" if df[target_col].dtype in ["int64", "float64"] and len(df[target_col].unique()) > 10 else "Classification"
        st.write(f"**Detected Problem Type:** 🧠 {problem_type}")

        models = {}
        results = {}

        if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Classifier": SVC(probability=True)
            }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[model_name] = {"MSE": mse, "MAE": mae, "R²": r2}
            else:
                acc = accuracy_score(y_test, y_pred)
                roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(set(y_test)) == 2 else "N/A"
                results[model_name] = {"Accuracy": acc, "ROC-AUC": roc}

        results_df = pd.DataFrame(results).T
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(results_df)

        # Confusion matrix visualization (classification only)
        if problem_type == "Classification":
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, models["Random Forest Classifier"].predict(X_test))
            fig = ff.create_annotated_heatmap(z=cm, x=['Pred 0', 'Pred 1'], y=['Actual 0', 'Actual 1'],
                                             colorscale='Blues', showscale=True)
            st.plotly_chart(fig, use_container_width=True)

        # Choose model for predictions
        best_model = st.selectbox("🎯 Select Model for Prediction", list(models.keys()))

        if st.button("🔮 Generate Predictions"):
            final_model = models[best_model]
            y_pred_final = final_model.predict(X)

            df_predictions = df.copy()
            df_predictions["Predictions"] = y_pred_final

            st.subheader("🔹 Predictions Preview")
            st.dataframe(df_predictions.head())

            csv = df_predictions.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions (CSV)", data=csv,
                               file_name="Predictions.csv", mime="text/csv")

    

