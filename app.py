import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    r2_score,
)
import io  # for in-memory file downloads

def main():
    # ----------------------------------------------------------------
    # App Title & Introduction
    # ----------------------------------------------------------------
    st.title("Interactive Decision Tree Builder (With Depth=3 Default)")

    st.write(
        """
        **Build and interpret a Decision Tree** quickly and easily!

        **Recommended Default**:  
        - **Max Depth = 3** for a more interpretable tree.  
        - If you choose a depth larger than 3, the visual tree can become very large and hard to read.  
          Therefore, if the trained tree actually ends up with depth > 3, we will also provide a **text-based** breakdown.
        """
    )

    # ----------------------------------------------------------------
    # Step 1: Upload CSV
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
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
        st.write("### Classification or Regression?")
        task_type = st.selectbox(
            "Select the tree type:",
            ["Classification", "Regression"],
            help=(
                "Classification = predict categories (e.g., Pass/Fail). "
                "Regression = predict numeric values (e.g., scores, amounts)."
            )
        )

        # ------------------------------------------------------------
        # Step 3: Target Variable
        # ------------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox(
            "Select your Target (outcome) variable:",
            all_columns,
            help="Which column do you want the model to predict?"
        )

        # ------------------------------------------------------------
        # Step 4: Predictor Variables & Scale
        # ------------------------------------------------------------
        st.write("### Choose Predictor Variables & Their Scale")
        possible_predictors = [col for col in all_columns if col != target_col]
        selected_predictors = st.multiselect(
            "Select columns to use as predictors:",
            possible_predictors,
            default=possible_predictors
        )

        scale_of_measurement = {}
        for pred in selected_predictors:
            scale_choice = st.selectbox(
                f"'{pred}' is:",
                ["numeric", "categorical"],
                key=f"scale_{pred}",
                help="If it's truly numeric (continuous values), choose numeric; otherwise, categorical."
            )
            scale_of_measurement[pred] = scale_choice

        # ------------------------------------------------------------
        # Step 5: Decision Tree Hyperparameters
        # ------------------------------------------------------------
        st.write("### Decision Tree Hyperparameters")

        # Criterion
        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion",
                ["gini", "entropy", "log_loss"],
                help="How to measure purity for classification splits."
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                help="How to measure split quality for regression."
            )

        # Max Depth - default to 3 for interpretability
        max_depth = st.slider(
            "Max Depth (default = 3 for interpretability)",
            min_value=0,
            max_value=20,
            value=3,
            step=1,
            help=(
                "Deeper trees can capture more complexity but may be harder to interpret. "
                "0 = no limit."
            )
        )
        max_depth = None if max_depth == 0 else max_depth

        # Min Samples Split
        min_samples_split = st.slider(
            "Minimum Samples per Split",
            min_value=2,
            max_value=50,
            value=2,
            step=1,
            help="At least this many samples in a node before the tree will consider splitting further."
        )

        # ------------------------------------------------------------
        # Step 6: Train Model
        # ------------------------------------------------------------
        if st.button("Train Model"):
            if not selected_predictors:
                st.error("No predictors selected. Please select at least one.")
                return

            # Build X (features) with encoding for categorical
            X_list = []
            col_names = []
            for col in selected_predictors:
                if scale_of_measurement[col] == "categorical":
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    X_list.append(dummies)
                    col_names.extend(dummies.columns.tolist())
                else:
                    X_list.append(df[[col]])
                    col_names.append(col)

            if not X_list:
                st.error("Error building feature matrix. Check your predictor selections.")
                return

            X = pd.concat(X_list, axis=1)
            y = df[target_col]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Build the decision tree model
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

            # ----------------------------------------------------------
            # Model Performance
            # ----------------------------------------------------------
            st.subheader("Model Performance")
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy**: {acc:.4f}")
                st.write(
                    "Accuracy = fraction of test samples predicted correctly."
                )

                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)

                cr = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report**:")
                st.write(pd.DataFrame(cr).transpose())
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**RMSE**: {rmse:.4f}")
                st.write(f"**R²**: {r2:.4f}")

            # ----------------------------------------------------------
            # Feature Importance
            # ----------------------------------------------------------
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

            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            buf_imp = io.BytesIO()
            fig_imp.savefig(buf_imp, format="png", dpi=300, bbox_inches="tight")
            buf_imp.seek(0)
            st.download_button(
                label="Download Feature Importance (PNG)",
                data=buf_imp,
                file_name="feature_importance.png",
                mime="image/png"
            )

            # ----------------------------------------------------------
            # Decision Tree Figure
            # ----------------------------------------------------------
            st.subheader("Decision Tree Figure")
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
                ax=ax_tree
            )
            st.pyplot(fig_tree)

            st.write(
                """
                **How to read this tree figure**:

                - Each **node** shows:
                  - **samples**: How many training samples are in that node.\n
                  - **value**:\n
                    - Classification: distribution of samples in each class.\n
                    - Regression: the average target value in that node.\n
                  - **impurity**:
                    - Classification (Gini or Entropy): measures how mixed the classes are.\n
                    - Regression (MSE or other): indicates how spread out the target values are in the node.\n
                - **Leaf nodes** (no more splits) display the final prediction for that subgroup.\n
                - A tree depth of **3** is often more interpretable. If you increase it, the diagram can get large.\n
                - Use the color intensity to see how pure a node is or how high/low the predicted value is.
                """
            )

            # Download the tree figure
            buf_tree = io.BytesIO()
            fig_tree.savefig(buf_tree, format="png", dpi=300, bbox_inches="tight")
            buf_tree.seek(0)
            st.download_button(
                label="Download Decision Tree Figure (PNG)",
                data=buf_tree,
                file_name="decision_tree_plot.png",
                mime="image/png"
            )

            # ----------------------------------------------------------
            # If the final model depth is > 3, display text-based tree
            # ----------------------------------------------------------
            final_depth = model.get_depth()
            if final_depth > 3:
                st.warning(
                    f"Your trained tree ended up with an actual depth of {final_depth}, which is > 3. "
                    "Below is a text-based summary to help interpret deeper trees more easily."
                )
                from sklearn.tree import export_text
                tree_rules = export_text(model, feature_names=col_names)
                st.code(tree_rules)

                st.write(
                    """
                    **How to read this**:
                    - Each `|---` indicates another level of splitting.\n
                    - `samples` = how many training samples in that node.\n
                    - `value` = For classification, distribution of classes; for regression, average target.\n
                    - `impurity` = measure of how mixed (classification) or how spread out (regression) the node is.
                    """
                )

    else:
        st.info("Upload a CSV file to begin.")

if __name__ == "__main__":
    main()

