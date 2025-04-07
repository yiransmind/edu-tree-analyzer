import streamlit as st
import pandas as pd
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    r2_score
)
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Streamlit Title
# ------------------------------------------------
st.title("🎓 Easy Decision Tree Analyzer for Education Research")

# ------------------------------------------------
# 2. File Upload
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV data
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Preview of Your Data")
    st.write(
        "Below is a quick look at the first few rows of the dataset you uploaded."
    )
    st.dataframe(df.head())

    # ------------------------------------------------
    # 3. Variable Selection
    # ------------------------------------------------
    st.subheader("2. Select Target and Predictor Variables")
    all_columns = df.columns.tolist()

    # Target selection
    target = st.selectbox("🎯 Choose the target (outcome) variable", all_columns)

    # Target type selection
    target_type = st.radio(
        "What is the target variable’s type?",
        ["Categorical (e.g., pass/fail)", "Numerical (continuous)"]
    )

    # Predictor selection
    predictors = st.multiselect(
        "🧩 Choose one or more predictor variables (excluding the target)",
        [col for col in all_columns if col != target]
    )

    # Data types for each predictor
    predictor_types = {}
    for col in predictors:
        predictor_types[col] = st.selectbox(
            f"Data type for predictor `{col}`",
            ["Categorical", "Numerical"],
            key=col
        )

    # ------------------------------------------------
    # 4. Model Setup and Training
    # ------------------------------------------------
    if target and predictors:
        # Separate features (X) and target (y)
        X = df[predictors].copy()
        y = df[target].copy()

        # Convert predictor types
        for col, vtype in predictor_types.items():
            if vtype == "Categorical":
                X[col] = X[col].astype("category")

        # Determine problem type
        if "Categorical" in target_type:
            # Classification
            y = y.astype("category")
            problem_type = "classification"
        else:
            # Regression
            y = pd.to_numeric(y, errors="coerce")
            problem_type = "regression"

        # Combine and drop missing rows
        data = pd.concat([X, y], axis=1).dropna()
        X = data[predictors]
        y = data[target]

        # Encode categorical predictors (dummy variables)
        X = pd.get_dummies(X, drop_first=True)

        st.write(f"Detected **{problem_type.upper()}** problem based on your choices.")

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Build the model (with a modest max_depth=4 for simplicity)
        if problem_type == "classification":
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=4, random_state=42)

        # Train (fit) the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # ------------------------------------------------
        # 5. Multiple Model Fit Indices
        # ------------------------------------------------
        st.subheader("3. Model Evaluation Metrics")

        if problem_type == "classification":
            # Classification metrics: Accuracy, Precision, Recall, F1
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            st.write("**Classification Performance**")
            st.write(f"- Accuracy: {accuracy:.2f}")
            st.write(f"- Precision: {precision:.2f}")
            st.write(f"- Recall: {recall:.2f}")
            st.write(f"- F1 Score: {f1:.2f}")

        else:
            # Regression metrics: MSE, RMSE, R^2
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, y_pred)

            st.write("**Regression Performance**")
            st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"- R-squared (R2): {r2:.2f}")

        # ------------------------------------------------
        # 6. Simplified Decision Tree Rules
        # ------------------------------------------------
        st.subheader("4. Simplified Decision Tree Rules")
        st.write(
            "Below is a text-based outline of how the decision tree is splitting the data. "
            "This can be somewhat technical, but try reading it top-down: each rule shows a condition, "
            "and `class:` (or `value:` for regression) at the end shows the outcome when that path is taken."
        )

        rules_text = export_text(model, feature_names=X.columns.tolist(), max_depth=3)
        st.text(rules_text)

        st.caption(
            "Tip: The tree is limited to a depth of 3 for readability here. "
            "You can increase `max_depth` to see deeper splits."
        )

        # ------------------------------------------------
        # 7. Minimal Decision Tree Diagram
        # ------------------------------------------------
        st.subheader("5. Minimal Decision Tree Diagram")
        st.write(
            "Below is a simplified diagram of the decision tree. We’ve removed extra details "
            "like impurities and filled colors to keep it clean for beginners."
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_tree(
            model,
            feature_names=X.columns,
            max_depth=3,           # Show only first 3 levels
            impurity=False,        # Hide impurity (e.g., Gini)
            filled=False,          # No colors
            proportion=False,      # Avoid proportions
            label='none'           # Hide node labels like 'Node #'
        )
        st.pyplot(fig)

        st.caption(
            "Note: The actual trained tree can be up to depth=4, but we're only displaying 3 levels here for clarity."
        )

        # ------------------------------------------------
        # 8. Report Feature Importance (No Plot)
        # ------------------------------------------------
        st.subheader("6. Feature Importance Scores")
        st.write(
            "Below are the importance scores for each predictor, showing how much they contribute to the decisions. "
            "A larger score indicates a bigger influence on the final splits."
        )

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.table(importance_df.reset_index(drop=True))

        # ------------------------------------------------
        # 9. Interpretation and Next Steps
        # ------------------------------------------------
        st.subheader("7. Interpretation and Next Steps")
        st.write(
            "Here’s how you might interpret these results:\n\n"
            "- **Model Fit Indices**: Help judge how well your model performs. "
            "For classification, pay attention to accuracy, precision, recall, and F1. "
            "For regression, consider MSE, RMSE, and R-squared.\n"
            "- **Decision Rules**: Read the text-based decision rules or follow the diagram from top to bottom. "
            "Each split is driven by a predictor with a threshold (for numeric) or category check.\n"
            "- **Feature Importance**: The highest-scoring features are the most influential in splitting. "
            "These are often where you learn which predictors best explain your outcome.\n"
            "- **Limit Tree Depth**: Keeping max_depth around 3–4 can make the tree easier to understand. "
            "Deeper trees may overfit or become too complicated.\n"
            "- **Consider Other Models**: This is a single decision tree. "
            "You could explore ensembles (Random Forest, Gradient Boosting) or other methods for potentially better performance.\n"
            "- **Data Quality**: Make sure your data is clean. Missing values, outliers, or poorly-encoded categorical data can harm model performance."
        )

