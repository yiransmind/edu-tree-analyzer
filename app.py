import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Guided Decision Tree Builder for Classification & Regression")

    st.markdown("""
    ### 👋 Welcome!
    This app helps you build and **understand decision trees** using your own dataset.
    It's designed for **non-coders** and those new to machine learning.

    We'll walk you through:
    1. Uploading your data 📁
    2. Choosing between **Classification** or **Regression** 🌳
    3. Selecting your **target** and **predictors** 🔍
    4. Tuning tree parameters for interpretability ⚙️
    5. Viewing model performance 📈
    6. Visualizing and understanding the **Decision Tree** with explanations 💡
    7. Downloading results for your reports 📤
    
    > 📌 Don't worry if you're not familiar with every term — we'll explain everything as you go.
    """)

    # Upload CSV File
    st.header("Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        with st.expander("🔍 Preview the first few rows of your data"):
            st.dataframe(df.head())

        # Classification or Regression
        st.header("Step 2: Choose Your Prediction Task")
        task_type = st.radio(
            "What kind of outcome are you predicting?",
            ["Classification", "Regression"],
            help="Classification = predicting categories (e.g. Yes/No), Regression = predicting numbers (e.g. score)"
        )

        # Select target variable
        st.header("Step 3: Select Your Target Variable")
        target_col = st.selectbox("Choose the column you want to predict:", df.columns)

        # Select predictor variables
        st.header("Step 4: Select Predictor Variables")
        st.markdown("""
        Select the columns that might help predict your outcome. Avoid including the target column itself.
        """)
        predictors = st.multiselect("Choose predictor columns:", [c for c in df.columns if c != target_col])

        # Identify data type of each predictor
        st.subheader("Step 4.1: Label Each Predictor")
        st.markdown("""
        Tell us whether each predictor is **numeric** (e.g. age, score) or **categorical** (e.g. gender, school).
        We'll handle encoding automatically.
        """)
        scale_dict = {}
        for col in predictors:
            scale_dict[col] = st.selectbox(f"'{col}' is...", ["numeric", "categorical"], key=f"scale_{col}")

        # Model hyperparameters
        st.header("Step 5: Tree Settings (Hyperparameters)")
        st.markdown("""
        You can control how complex or simple the tree becomes.

        - **Max Depth**: How many times can the tree split? Shallower trees are easier to interpret.
        - **Min Samples to Split**: Minimum data points a node must have before it can split.
        - **Criterion**: The rule for deciding the best split at each node.
        """)

        criterion = st.selectbox(
            "Choose splitting criterion:",
            ["gini", "entropy"] if task_type == "Classification" else ["squared_error", "absolute_error"],
            help="For classification: Gini and Entropy measure class purity. For regression: squared_error = MSE."
        )

        max_depth = st.slider("Maximum depth of the tree (0 = unlimited):", 0, 20, 3)
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider("Minimum samples required to split:", 2, 20, 2)

        # Train Model
        st.header("Step 6: Train the Tree 🌳")
        if st.button("Train Model"):
            # Prepare data
            X_parts = []
            col_names = []
            for col in predictors:
                if scale_dict[col] == "categorical":
                    dummies = pd.get_dummies(df[col], prefix=col)
                    X_parts.append(dummies)
                    col_names.extend(dummies.columns)
                else:
                    X_parts.append(df[[col]])
                    col_names.append(col)

            X = pd.concat(X_parts, axis=1)
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if task_type == "Classification":
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            else:
                model = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )

            model.fit(X_train, y_train)

            st.success("✅ Model trained!")

            # Performance
            st.header("Step 7: Model Performance")
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc:.2%}")
                st.write("Accuracy = proportion of test samples predicted correctly.")

                st.subheader("Confusion Matrix")
                st.write(confusion_matrix(y_test, y_pred))

                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("R²", f"{r2:.2f}")
                st.write("Note: The tree figure will show 'squared_error' (which is MSE). Don't let the label confuse you!")

            # Feature importance
            st.header("Step 8: Feature Importance")
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            fi_df = pd.DataFrame({
                "Feature": np.array(col_names)[sorted_idx],
                "Importance": importances[sorted_idx]
            })
            st.dataframe(fi_df)

            fig_imp, ax_imp = plt.subplots()
            ax_imp.barh(fi_df.Feature, fi_df.Importance)
            ax_imp.set_xlabel("Importance")
            ax_imp.set_title("Feature Importance")
            st.pyplot(fig_imp)

            buf_imp = io.BytesIO()
            fig_imp.savefig(buf_imp, format="png", dpi=300, bbox_inches="tight")
            st.download_button("Download Importance Chart", data=buf_imp, file_name="importance.png")

            # Tree figure
            st.header("Step 9: Decision Tree Visualization")
            fig_tree, ax_tree = plt.subplots(figsize=(12, 8), dpi=300)
            plot_tree(
                model,
                feature_names=col_names,
                class_names=[str(c) for c in model.classes_] if task_type == "Classification" else None,
                filled=True,
                rounded=True,
                impurity=True,
                ax=ax_tree
            )
            st.pyplot(fig_tree)

            buf_tree = io.BytesIO()
            fig_tree.savefig(buf_tree, format="png", dpi=300, bbox_inches="tight")
            st.download_button("Download Tree Figure", data=buf_tree, file_name="tree.png")

            st.markdown("""
            #### How to Interpret the Tree Figure
            - Each box is a **node**. The top node is the root.
            - **samples**: number of training examples at that node.
            - **value**:
              - For classification: class distribution.
              - For regression: predicted mean value.
            - **impurity**:
              - For classification: Gini/Entropy.
              - For regression: "squared_error" (a.k.a. MSE).
            - **Color** helps you spot majority classes or high/low values.

            👉 **Tip**: Trace a path from root to a leaf to learn what combinations lead to specific predictions.
            """)

            # Show text-based tree if depth is high
            depth = model.get_depth()
            if depth > 3:
                st.warning(f"The tree is {depth} levels deep — that can be hard to read. Here's a text version:")
                st.code(export_text(model, feature_names=col_names))

                if task_type == "Regression":
                    st.info("Note: scikit-learn does NOT show 'impurity' (e.g. MSE) in this text view for regression trees.")

if __name__ == "__main__":
    main()



