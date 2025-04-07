
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎓 Easy Decision Tree Analyzer for Education Research")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of your data:")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select your target variable", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        data = pd.concat([X, y], axis=1).dropna()
        X = data.drop(columns=[target])
        y = data[target]

        problem_type = "classification" if y.nunique() < 10 else "regression"
        st.write(f"Detected as a **{problem_type}** problem.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if problem_type == "classification":
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=4, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            score = accuracy_score(y_test, y_pred)
            st.write(f"✅ Accuracy: **{score:.2f}**")
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"✅ Mean Squared Error: **{mse:.2f}**")

        st.subheader("🧠 Decision Tree Logic")
        tree_rules = export_text(model, feature_names=list(X.columns))
        st.code(tree_rules)

        st.subheader("📌 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        st.pyplot(fig)

        st.subheader("📝 Basic Interpretation")
        top_feat = importance_df.iloc[0]
        st.markdown(f"""
        - **Most important variable**: `{top_feat.Feature}`
        - This variable contributed the most to splitting decisions in the tree.
        - Consider exploring its interaction with other features or using it in further modeling.
        """)
