
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Decision Tree Analyzer", layout="wide")
st.title("📊 Decision Tree Analyzer — Journal-Standard Output")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select your target variable (the one you want to predict)", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        data = pd.concat([X, y], axis=1).dropna()
        X = data.drop(columns=[target])
        y = data[target]

        problem_type = "classification" if y.nunique() < 10 else "regression"
        st.markdown(f"🧠 Detected as a **{problem_type.upper()}** problem based on the target variable.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        n_train, n_test = len(X_train), len(X_test)

        if problem_type == "classification":
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=3, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model metrics
        st.subheader("📈 Model Fit Summary")
        if problem_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            st.markdown(f"**Accuracy:** `{acc:.2f}`")
            model_fit_text = f"""
            The decision tree classifier was trained on 70% of the data (N = {n_train}), 
            and tested on the remaining 30% (N = {n_test}). The model achieved an accuracy of **{acc:.2f}**, 
            meaning it correctly predicted the outcome in {acc*100:.1f}% of test cases. 
            This indicates reasonably strong predictive power.
            """
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.markdown(f"**R²:** `{r2:.2f}`")
            st.markdown(f"**Mean Squared Error:** `{mse:.2f}`")
            model_fit_text = f"""
            The decision tree regression model was trained on 70% of the data (N = {n_train}), 
            and evaluated on the remaining 30% (N = {n_test}). It achieved an R² of **{r2:.2f}**, 
            explaining approximately {r2*100:.1f}% of the variance in the outcome. 
            The average prediction error (MSE) was **{mse:.2f}**.
            """

        st.markdown("🗣️ **Interpretation:**")
        st.markdown(model_fit_text)

        # Feature importance
        st.subheader("📌 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig_imp, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        st.pyplot(fig_imp)

        top_feat = importance_df.iloc[0]
        imp_text = f"""
        The most influential predictor was **{top_feat.Feature}**, contributing the most to splitting decisions in the tree. 
        This suggests that this variable plays a central role in determining the predicted outcome. 
        Other notable predictors include: {', '.join(importance_df['Feature'][1:4])}.
        """
        st.markdown("🗣️ **Interpretation:**")
        st.markdown(imp_text)

        # Tree plot
        st.subheader("🌳 Decision Tree Visualization (Depth = 3)")
        fig_tree, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10, ax=ax)
        st.pyplot(fig_tree)

        tree_text = f"""
        The decision tree begins by splitting on **{top_feat.Feature}**, indicating it is the primary factor in distinguishing outcomes. 
        Subsequent splits reveal how combinations of other predictors influence the target variable. 
        This hierarchical structure captures interactions and non-linear effects typical in educational and psychological phenomena.
        """
        st.markdown("🗣️ **Interpretation:**")
        st.markdown(tree_text)

        # Report export
        report = f"""
        === DECISION TREE ANALYSIS REPORT ===

        Target variable: {target}
        Problem type: {problem_type.upper()}
        Train/Test Split: 70% / 30% (Train N = {n_train}, Test N = {n_test})

        --- Model Fit ---
        {model_fit_text.strip()}

        --- Feature Importance ---
        Top features: {', '.join(importance_df['Feature'][:5])}
        {imp_text.strip()}

        --- Tree Summary ---
        {tree_text.strip()}
        """

        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name="decision_tree_report.txt",
            mime="text/plain"
        )
