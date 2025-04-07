import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
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

    all_columns = df.columns.tolist()

    target = st.selectbox("🎯 Select your target variable", all_columns)
    target_type = st.radio("🔎 Target variable type", ["Categorical", "Numerical"])

    predictors = st.multiselect("🧩 Select predictor variables", [col for col in all_columns if col != target])

    predictor_types = {}
    for col in predictors:
        predictor_types[col] = st.selectbox(f"📌 Type for predictor `{col}`", ["Categorical", "Numerical"], key=col)

    if target and predictors:
        X = df[predictors].copy()
        y = df[target].copy()

        # Convert types
        for col, vtype in predictor_types.items():
            if vtype == "Categorical":
                X[col] = X[col].astype("category")

        if target_type == "Categorical":
            y = y.astype("category")
            problem_type = "classification"
        else:
            y = pd.to_numeric(y, errors="coerce")
            problem_type = "regression"

        # Drop missing values
        data = pd.concat([X, y], axis=1).dropna()
        X = data[predictors]
        y = data[target]

        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        st.write(f"Detected as a **{problem_type}** problem.")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model fitting
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

        # Display rules
        st.subheader("🧠 Decision Tree Logic")
        tree_rules = export_text(model, feature_names=X.columns.tolist())
        st.code(tree_rules)

        # Plot tree
        st.subheader("🌳 Tree Diagram")
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10, ax=ax)
        st.pyplot(fig)

        # Feature importance
        st.subheader("📌 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        st.pyplot(fig)

        # Interpretation
        st.subheader("📝 Basic Interpretation")
        top_feat = importance_df.iloc[0]
        st.markdown(f"""
        - **Most important variable**: `{top_feat.Feature}`
        - This variable contributed the most to splitting decisions in the tree.
        - Consider exploring its interaction with other features or using it in further modeling.
        """)
