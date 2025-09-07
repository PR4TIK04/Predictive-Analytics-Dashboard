import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("📊 Predictive Analytics Dashboard")
st.markdown("Upload your dataset and explore insights, build ML models, and get predictions!")

# -------------------------
# Step 1: Upload Dataset
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Use sample for demo to reduce memory load
    df = df.sample(min(500, len(df)))
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.markdown("### Dataset Summary")
    st.write(df.describe())

    # -------------------------
    # Step 2: EDA
    # -------------------------
    st.markdown("---")
    st.header("🔍 Exploratory Data Analysis (EDA)")

    option = st.selectbox("Choose Analysis Type", ["Dataset Overview", "Profiling Report"])
    
    if option == "Dataset Overview":
        col = st.selectbox("Select a Column for Visualization", df.columns)
        st.bar_chart(df[col].value_counts())
        
    elif option == "Profiling Report":
        st.info("⚡ Generating profiling report on-demand...")
        import ydata_profiling
        from streamlit_pandas_profiling import st_profile_report
        profile = ydata_profiling.ProfileReport(df, explorative=True)
        st_profile_report(profile)

    # -------------------------
    # Step 3: Visualization
    # -------------------------
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

    # -------------------------
    # Step 4: Automated ML
    # -------------------------
    st.markdown("---")
    st.header("🤖 Automated ML & Predictions")

    target_col = st.selectbox("📌 Select Target Column", df.columns)
    
    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle missing values & categorical data
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X.select_dtypes(include=[np.number])),
                         columns=X.select_dtypes(include=[np.number]).columns)
        X = pd.get_dummies(X, drop_first=True)

        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        problem_type = "Regression" if df[target_col].dtype in ["int64", "float64"] and len(df[target_col].unique()) > 10 else "Classification"
        st.write(f"**Detected Problem Type:** 🧠 {problem_type}")

        models = {}
        if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor()
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Classifier": SVC(probability=True)
            }

        selected_model_name = st.selectbox("🎯 Select Model for Training", list(models.keys()))

        if st.button("Train Model"):
            model = models[selected_model_name]
            model.fit(X_train, y_train)
            st.success(f"✅ {selected_model_name} trained successfully!")

            # Evaluate
            if problem_type == "Regression":
                y_pred = model.predict(X_test)
                st.subheader("📊 Model Performance")
                st.write({
                    "MSE": mean_squared_error(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R²": r2_score(y_test, y_pred)
                })
            else:
                y_pred = model.predict(X_test)
                st.subheader("📊 Model Performance")
                st.write({
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "ROC-AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(set(y_test)) == 2 else "N/A"
                })

                # Confusion Matrix on-demand
                if st.checkbox("Show Confusion Matrix"):
                    from plotly.figure_factory import create_annotated_heatmap
                    cm = confusion_matrix(y_test, y_pred)
                    fig = create_annotated_heatmap(z=cm,
                                                   x=[f'Pred {i}' for i in range(cm.shape[0])],
                                                   y=[f'Actual {i}' for i in range(cm.shape[0])],
                                                   colorscale='Blues', showscale=True)
                    st.plotly_chart(fig, use_container_width=True)

            # Predictions on full dataset
            if st.checkbox("Generate Predictions for Full Dataset"):
                y_pred_full = model.predict(X)
                df_predictions = df.copy()
                df_predictions["Predictions"] = y_pred_full
                st.subheader("🔹 Predictions Preview")
                st.dataframe(df_predictions.head())
                csv = df_predictions.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions (CSV)", data=csv,
                                   file_name="Predictions.csv", mime="text/csv")
