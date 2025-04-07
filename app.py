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
import io

def main():
    # ----------------------------------------------------------------
    # Title and Intro
    # ----------------------------------------------------------------
    st.title("Decision Tree Builder with Depth Suggestion & Text Breakdown")

    st.write(
        """
        **This app** helps you build and interpret a **Decision Tree**.  
        
        ### Why suggest a max depth of 3?
        - Depth 3 often produces a tree that is **easier to interpret**.
        - Very deep trees (>3) can become hard to read and might overfit.
        
        But you can still override this suggestion if you want a deeper (or shallower) tree.
        """
    )

    # ----------------------------------------------------------------
    # CSV Upload
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write(f"**Rows**: {df.shape[0]} | **Columns**: {df.shape[1]}")

        # ------------------------------------------------------------
        # 1. Classification or Regression
        # ------------------------------------------------------------
        st.write("### 1) Is Your Target Categorical or Numeric?")
        task_type = st.selectbox(
            "Select the type of Decision Tree:",
            ["Classification", "Regression"],
            help=(
                "Classification = target is a discrete category (e.g., pass/fail). "
                "Regression = target is a continuous numeric variable (e.g., exam score)."
            )
        )

        # ------------------------------------------------------------
        # 2. Select Target
        # ------------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox(
            "Select your Target (outcome) variable:",
            all_columns,
            help="Which column do you want to predict?"
        )

        # ------------------------------------------------------------
        # 3. Select Predictors and Scale
        # ------------------------------------------------------------
        st.write("### 2) Select Predictor Variables & Their Scale")
        possible_predictors = [c for c in all_columns if c != target_col]
        selected_predictors = st.multiselect(
            "Pick your predictor columns:",
            possible_predictors,
            default=possible_predictors
        )

        scale_of_measurement = {}
        for pred in selected_predictors:
            scale_choice = st.selectbox(
                f"'{pred}' is:",
                ["numeric", "categorical"],
                key=f"scale_{pred}"
            )
            scale_of_measurement[pred] = scale_choice

        # ------------------------------------------------------------
        # 4. Decision Tree Hyperparameters
        # ------------------------------------------------------------
        st.write("### 3) Decision Tree Hyperparameters")

        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion",
                ["gini", "entropy", "log_loss"],
                help=(
                    "How the tree measures the purity of a node. "
                    "'gini' and 'entropy' are common, 'log_loss' is another option."
                )
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                help=(
                    "For regression trees, 'squared_error' is typical. 'absolute_error' can help with outliers."
                )
            )

        # **Default** max depth is 3
        max_depth = st.slider(
            "Max Depth (0 = No Limit):",
            min_value=0,
            max_value=20,
            value=3,  # SUGGESTED DEPTH = 3
            step=1,
            help=(
                "We recommend 3 for easier interpretation, but you can override if you want a deeper or shallower tree."
            )
        )
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=50,
            value=2,
            step=1,
            help=(
                "A larger number helps reduce overfitting by preventing very small splits. "
                "2 is the most flexible setting."
            )
        )

        # ------------------------------------------------------------
        # 5. Train Model Button
        # ------------------------------------------------------------
        if st.button("Train Model"):
            if not selected_predictors:
                st.error("No predictors selected. Please choose at least one.")
                return

            # Construct feature matrix X
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
                st.error("Failed to create the feature matrix. Check your selections.")
                return

            X = pd.concat(X_list, axis=1)
            y = df[target_col]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Build and train
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

            # ----------------------------------------------------------
            # Model Performance
            # ----------------------------------------------------------
            st.subheader("Model Performance")
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy**: {accuracy:.4f}")
                st.write("Accuracy = fraction of correct predictions (1.0 means perfect).")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)
                st.write(
                    "Rows are actual classes; columns are predicted classes. Diagonal = correct predictions."
                )

                # Classification report
                cr = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report**:")
                st.write(pd.DataFrame(cr).transpose())
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**RMSE**: {rmse:.4f}")
                st.write(
                    "Root Mean Squared Error = on average, how far the predictions deviate from the actual values."
                )
                st.write(f"**R²**: {r2:.4f}")
                st.write(
                    "Coefficient of Determination = fraction of target variance explained by the model."
                )

            # ----------------------------------------------------------
            # Feature Importance
            # ----------------------------------------------------------
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            sorted_features = np.array(col_names)[sorted_idx]
            sorted_importances = importances[sorted_idx]

            fi_df = pd.DataFrame({"Feature": sorted_features, "Importance": sorted_importances})
            st.write("**Ranking of Predictors (High to Low)**:")
            st.write(fi_df)

            # Bar chart
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            # Download the feature importance figure
            buf_imp = io.BytesIO()
            fig_imp.savefig(buf_imp, format="png", dpi=300, bbox_inches="tight")
            buf_imp.seek(0)
            st.download_button(
                label="Download Feature Importance Plot (PNG)",
                data=buf_imp,
                file_name="feature_importance_plot.png",
                mime="image/png"
            )

            # ----------------------------------------------------------
            # Decision Tree Figure
            # ----------------------------------------------------------
            st.subheader("Decision Tree Figure")

            # Make the figure high resolution
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

            # Guidance on reading the figure
            st.write(
                """
                **How to interpret each node**:
                - **samples**: Number of training records that fall into this node.\n
                - **value**:\n
                  - Classification: shows how many samples of each class are in that node.\n
                  - Regression: shows the mean (or average) target value in that node.\n
                - **impurity**:\n
                  - For Classification: Gini or Entropy. Lower = more "pure" (mostly one class).\n
                  - For Regression: MSE or similar, showing how spread out values are.\n
                - **Color saturation** often indicates the level or majority class.\n
                """
            )
            st.write(
                "**Tip**: Because we suggested a max depth of 3, the tree might be easier to read. "
                "If you use a larger depth, it can become very complex. However, we still provide a "
                "text-based summary (below) if the depth goes beyond 3."
            )

            # Download button for the tree figure
            buf_tree = io.BytesIO()
            fig_tree.savefig(buf_tree, format="png", dpi=300, bbox_inches="tight")
            buf_tree.seek(0)
            st.download_button(
                label="Download Decision Tree Plot (PNG)",
                data=buf_tree,
                file_name="decision_tree_plot.png",
                mime="image/png"
            )

            # ----------------------------------------------------------
            # If the Final Tree Depth > 3, Show Text-Based Rules
            # ----------------------------------------------------------
            final_depth = model.get_depth()
            if final_depth > 3:
                st.warning(
                    f"The final tree depth is **{final_depth}**, which is > 3. "
                    "This might be hard to visualize. Here's a text-based breakdown too:"
                )

                tree_text = export_text(model, feature_names=col_names)
                st.code(tree_text)

                st.write(
                    "**How to read these text rules**:\n"
                    "- Each `|---` indicates another level of splitting.\n"
                    "- `samples` = how many training samples in that node.\n"
                    "- `value` = for classification, distribution of classes; for regression, the mean target.\n"
                    "- `gini`/`entropy`/`mse` indicates how mixed the node is.\n"
                    "A node without further splits is a **leaf**."
                )

    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()

