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
    # ----------------------------------------------------------------
    # Title and High-Level Introduction
    # ----------------------------------------------------------------
    st.title("Comprehensive Decision Tree Builder & Interpreter")

    st.write(
        """
        ### Welcome to the All-in-One Decision Tree App

        This tool helps **non-coders** (and coders alike) to:
        
        1. **Upload a CSV** dataset.\n
        2. Decide if their problem is **Classification** or **Regression**.\n
        3. Select a **Target** variable (the outcome to predict) and **Predictor** variables (features).\n
        4. Specify which predictors are **numeric** vs. **categorical** (the app will one-hot encode categorical).\n
        5. Adjust **hyperparameters** (criterion, max depth, min samples split) to control the tree’s complexity.\n
        6. **Train** the model and view:\n
           - **Performance metrics** (Accuracy, Confusion Matrix, Classification Report **or** RMSE, R²)\n
           - **Feature Importance** (which predictors matter most)\n
           - A **High-Resolution Decision Tree Figure** (with an option to download it)\n
           - A **Text-Based Tree** if the tree ends up deeper than 3 levels (since large trees can be hard to read visually)\n

        #### Important Notes:
        - For **Regression Trees**, scikit-learn’s figure displays node impurity as "**squared_error**."  
          **This is effectively the Mean Squared Error (MSE)** for the samples in that node.  
          The name "squared_error" just matches the internal scikit-learn criterion.\n
        - For **Regression Trees**, the text-based tree (`export_text`) does **not** show the node’s impurity.  
          That’s a scikit-learn quirk: classification rules show Gini/Entropy in the text, but regression rules do not show MSE.\n
        """
    )

    # ----------------------------------------------------------------
    # Step 1: Upload CSV
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("1) Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write("**Data Dimensions**:")
        st.write(f"- Rows: {df.shape[0]}")
        st.write(f"- Columns: {df.shape[1]}")

        # ------------------------------------------------------------
        # Step 2: Classification or Regression
        # ------------------------------------------------------------
        st.write("### 2) Classification or Regression?")
        task_type = st.selectbox(
            "Select which type of problem fits your target variable",
            ["Classification", "Regression"],
            help=(
                "Choose Classification if your outcome is a discrete category (e.g., Pass/Fail). "
                "Choose Regression if your outcome is numeric/continuous (e.g., a score or amount)."
            )
        )

        # ------------------------------------------------------------
        # Step 3: Select Target Variable
        # ------------------------------------------------------------
        all_cols = df.columns.tolist()
        target_col = st.selectbox(
            "3) Pick your Target (outcome) variable",
            all_cols,
            help="Which column are you trying to predict?"
        )

        # ------------------------------------------------------------
        # Step 4: Pick Predictors & Indicate Scale
        # ------------------------------------------------------------
        st.write("### 4) Choose Predictor Variables & Their Scales")
        possible_predictors = [c for c in all_cols if c != target_col]
        selected_predictors = st.multiselect(
            "Select the columns to use as features (predictors):",
            possible_predictors,
            default=possible_predictors,
            help="Pick any columns that you believe might help predict the target."
        )

        st.write("#### Indicate Numeric or Categorical for Each Predictor:")
        scale_info = {}
        for pred in selected_predictors:
            user_scale = st.selectbox(
                f"'{pred}' scale:",
                ["numeric", "categorical"],
                key=f"scale_{pred}",
                help=(
                    "If the column has continuous values, choose numeric. "
                    "If it has discrete labels, pick categorical (the app will one-hot encode it)."
                )
            )
            scale_info[pred] = user_scale

        # ------------------------------------------------------------
        # Step 5: Decision Tree Hyperparameters
        # ------------------------------------------------------------
        st.write("### 5) Decision Tree Hyperparameters")

        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion (Classification)",
                ["gini", "entropy", "log_loss"],
                help=(
                    "How to measure node purity for classification. "
                    "'gini' and 'entropy' are common choices; 'log_loss' is another option."
                )
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion (Regression)",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                help=(
                    "How we measure node quality for regression. 'squared_error' effectively means MSE, "
                    "which is typical for standard regression trees."
                )
            )

        max_depth = st.slider(
            "Max Depth of the Tree (0 = no limit)",
            min_value=0,
            max_value=20,
            value=3,
            help=(
                "How many times the tree can split. A smaller depth is more interpretable; "
                "a larger depth may fit complex patterns but could overfit."
            )
        )
        # Convert 0 to None for scikit-learn
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider(
            "Minimum Samples Required to Split",
            min_value=2,
            max_value=50,
            value=2,
            help=(
                "A node must have at least this many samples for the tree to consider splitting it. "
                "Increasing this can reduce overfitting by avoiding overly specific splits."
            )
        )

        # ------------------------------------------------------------
        # Step 6: Train Model Button
        # ------------------------------------------------------------
        if st.button("Train Decision Tree"):
            if not selected_predictors:
                st.error("You must select at least one predictor to proceed.")
                return

            # Build X by encoding categorical variables
            X_parts = []
            col_names = []
            for col in selected_predictors:
                if scale_info[col] == "categorical":
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    X_parts.append(dummies)
                    col_names.extend(dummies.columns.tolist())
                else:
                    X_parts.append(df[[col]])
                    col_names.append(col)

            if not X_parts:
                st.error("Failed to build feature matrix. Check your selections.")
                return

            X = pd.concat(X_parts, axis=1)
            y = df[target_col]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Initialize the tree model
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

            # Fit the model
            model.fit(X_train, y_train)

            # --------------------------------------------------------
            # Model Performance
            # --------------------------------------------------------
            st.subheader("Model Performance")

            y_pred = model.predict(X_test)
            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy**: {acc:.4f}")
                st.write(
                    "Proportion of correct predictions on the test set. 1.0 = perfect, 0.0 = none."
                )

                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)
                st.write(
                    "Rows represent **actual classes**, columns represent **predicted classes**. "
                    "The diagonal shows correct predictions."
                )

                cr = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report**:")
                st.write(pd.DataFrame(cr).transpose())

            else:
                mse_value = mean_squared_error(y_test, y_pred)
                rmse_value = np.sqrt(mse_value)
                r2_val = r2_score(y_test, y_pred)
                st.write(f"**RMSE**: {rmse_value:.4f}")
                st.write(
                    "RMSE (Root Mean Squared Error) means on average how far your predictions are off from the actual values."
                )
                st.write(f"**R²**: {r2_val:.4f}")
                st.write(
                    "R² (Coefficient of Determination) measures how much variance is explained. 1.0 = perfect, 0.0 = none."
                )

                st.write(
                    "**Note**: In the **tree figure** below, if you chose 'squared_error' as your criterion, "
                    "each node will display `'squared_error'` for the impurity measure. "
                    "That is effectively the **mean squared error** at that node."
                )

            # --------------------------------------------------------
            # Feature Importance
            # --------------------------------------------------------
            st.subheader("Feature Importance")

            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_idx]
            sorted_features = np.array(col_names)[sorted_idx]

            fi_df = pd.DataFrame({
                "Feature": sorted_features,
                "Importance": sorted_importances
            })
            st.write(fi_df)

            st.write(
                "A higher importance indicates that predictor was used more often (or more effectively) "
                "in splitting nodes to reduce error (regression) or increase purity (classification)."
            )

            # Bar chart
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            # Download button for feature importance plot
            buf_imp = io.BytesIO()
            fig_imp.savefig(buf_imp, format="png", dpi=300, bbox_inches="tight")
            buf_imp.seek(0)
            st.download_button(
                label="Download Feature Importance Plot (PNG)",
                data=buf_imp,
                file_name="feature_importance.png",
                mime="image/png"
            )

            # --------------------------------------------------------
            # Decision Tree Figure
            # --------------------------------------------------------
            st.subheader("Decision Tree Figure")
            st.write(
                "A high-resolution plot showing how the tree splits. "
                "Scroll or zoom to see details if it's large."
            )

            fig_tree, ax_tree = plt.subplots(figsize=(12, 8), dpi=300)

            if task_type == "Classification":
                class_names = [str(cls) for cls in model.classes_]
            else:
                class_names = None

            plot_tree(
                model,
                feature_names=col_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                impurity=True,  # Ensures impurity is shown in the figure
                ax=ax_tree
            )
            st.pyplot(fig_tree)

            st.write(
                """
                ### How to Read This Tree Figure

                - **samples**: Number of training samples in this node.\n
                - **value**:
                  - **Classification**: distribution of samples across classes.\n
                  - **Regression**: the average target value in this node.\n
                - **impurity**:
                  - **Classification**: 'gini', 'entropy', or 'log_loss' depending on your chosen criterion.\n
                  - **Regression**: e.g., 'squared_error' if you chose that criterion (the numeric value is MSE).\n
                - **Splits**: Each node splits according to a rule like "Feature <= x".\n
                - **Leaf nodes**: No further splits, so that node's 'value' is your final prediction for that subgroup.\n

                #### Note for Regression Trees:
                - The figure might show something like "squared_error = 45.2". That is effectively **MSE** at that node.\n
                - scikit-learn uses 'squared_error' as a label instead of 'MSE'.\n
                """
            )

            # Download the tree plot
            buf_tree = io.BytesIO()
            fig_tree.savefig(buf_tree, format="png", dpi=300, bbox_inches="tight")
            buf_tree.seek(0)
            st.download_button(
                label="Download Decision Tree (PNG)",
                data=buf_tree,
                file_name="decision_tree_plot.png",
                mime="image/png"
            )

            # --------------------------------------------------------
            # If Depth > 3, Provide a Text Tree
            # --------------------------------------------------------
            final_depth = model.get_depth()
            if final_depth and final_depth > 3:
                st.warning(
                    f"Your tree has an actual depth of {final_depth}, which can be hard to read in the diagram. "
                    "Below is a text-based breakdown for clarity."
                )
                tree_txt = export_text(model, feature_names=col_names)
                st.code(tree_txt)

                st.write(
                    """
                    #### Text Tree Limitations for Regression
                    - For Classification, this text typically includes node impurity (like Gini/Entropy).
                    - For Regression, scikit-learn **does not show** the impurity measure (MSE) in this text format.
                      You will see `value = [some_average]`, but not an 'impurity' line.\n
                    """
                )

    else:
        st.info("Please upload a CSV file to start.")

if __name__ == "__main__":
    main()


