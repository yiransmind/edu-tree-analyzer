
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Decision Tree Analyzer", layout="wide")
st.title("🎓 Decision Tree Analyzer for Educational Research")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select your target variable (the one you want to predict)", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Drop rows with missing values
        data = pd.concat([X, y], axis=1).dropna()
        X = data.drop(columns=[target])
        y = data[target]

        # Determine problem type
        problem_type = "classification" if y.nunique() < 10 else "regression"
        st.markdown(f"🧠 Detected as a **{problem_type.upper()}** problem based on target variable.")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit decision tree with depth = 3
        if problem_type == "classification":
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=3, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model fit
        st.subheader("📈 Model Fit Summary")
        if problem_type == "classification":
            score = accuracy_score(y_test, y_pred)
            st.markdown(f"**Accuracy:** `{score:.2f}`")
            st.markdown(f"🗣️ Interpretation: The model correctly predicted the outcome **{score*100:.1f}%** of the time. A higher score indicates better predictive performance.")
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.markdown(f"**Mean Squared Error:** `{mse:.2f}`")
            st.markdown("🗣️ Interpretation: A lower MSE means better prediction. This tells us the average squared difference between predicted and actual values.")

        # Variable importance
        st.subheader("📌 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig_imp, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        st.pyplot(fig_imp)

        top_feat = importance_df.iloc[0]
        st.markdown(f"🗣️ Interpretation: The most influential variable in predicting the outcome was **`{top_feat.Feature}`**. This means the model often used this variable at the top of the decision path.")

        # Tree visualization
        st.subheader("🌳 Decision Tree Structure (Depth = 3)")
        fig_tree, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10, ax=ax)
        st.pyplot(fig_tree)

        st.markdown("🗣️ Interpretation: This tree shows how the model splits the data to make predictions. At each node, it picks the variable and condition that best separates the outcomes.")
