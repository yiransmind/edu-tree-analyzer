import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text  # for textual interpretation
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt

def main():
    st.title("Decision Tree App for Educational / Social Science Research")

    st.write(
        """
        **How to Use**:
        1. Upload a CSV file.\n
        2. Select whether you want a **Classification** or **Regression** tree.\n
        3. Pick your **Target (outcome)** variable.\n
        4. Select which columns will serve as **Predictors (features)**.\n
        5. Specify the **Scale of Measurement** for each predictor (categorical vs. numeric).\n
        6. Adjust any **hyperparameters** (e.g., criterion, max depth).\n
        7. Click **Train Model** to see:\n
           - Performance metrics (Accuracy, Confusion Matrix, RMSE, R²)\n
           - Tree structure and rules (interpretation)\n
           - Variable importance\n
           - A tree visualization\n
        """
    )

    # -----------------------------------------------------------
    # Step 1: Upload CSV
    # -----------------------------------------------------------
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head(10))

        # -----------------------------------------------------------
        # Step 2: Choose task type (Classification or Regression)
        # -----------------------------------------------------------
        task_type = st.selectbox(
            "Select the type of Decision Tree",
            ["Classification", "Regression"]
        )

        # -----------------------------------------------------------
        # Step 3: Choose the target (outcome) variable
        # -----------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox("Select your target (outcome) variable", all_columns)

        # -----------------------------------------------------------
        # Step 4: Choose predictor variables
        # -----------------------------------------------------------
        possible_predictors = [col for col in all_columns if col != target_col]
        selected_predictors = st.multiselect(
            "Select your predictor variables",
            possible_predictors,
            default=possible_predictors  # or leave blank if you prefer
        )

        # -----------------------------------------------------------
        # Step 5: Choose scale of measurement for each predictor
        # -----------------------------------------------------------
        st.write("### Specify Scale of Measurement for Each Predictor")
        scale_of_measurement = {}
        for predictor in selected_predictors:
            scale = st.selectbox(
                f"Variable: {predictor}",
                ("categorical", "numeric"),
                key=f"scale_{predictor}"
            )
            scale_of_measurement[predictor] = scale

        # -----------------------------------------------------------
        # Step 6: Select model hyperparameters
        # -----------------------------------------------------------
        if task_type == "Classification":
            criterion = st.selectbox(
                "Select split criterion",
                ["gini", "entropy", "log_loss"],
            )
        else:
            criterion = st.selectbox(
                "Select split criterion",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            )

        max_depth = st.slider(
            "Max depth of the tree (0 = unlimited)",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider(
            "Minimum samples required to split an internal node",
            min_value=2,
            max_value=50,
            value=2,
            step=1
        )

        # -----------------------------------------------------------
        # Step 7: Train Model Button
        # -----------------------------------------------------------
        if st.button("Train Model"):
            st.subheader("Model Results")

            # -----------------------------------------------------------
            # Build feature matrix X with appropriate encoding
            # -----------------------------------------------------------
            X_list = []
            col_names = []

            for col in selected_predictors:
                if scale_of_measurement[col] == "categorical":
                    # One-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    X_list.append(dummies)
                    col_names.extend(dummies.columns.tolist())
                else:
                    # Numeric
                    X_list.append(df[[col]])
                    col_names.append(col)

            if not X_list:
                st.error("No predictors selected. Please select at least one.")
                return

            X = pd.concat(X_list, axis=1)
            y = df[target_col]

            # -----------------------------------------------------------
            # Train-test split
            # -----------------------------------------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # -----------------------------------------------------------
            # Initialize and fit the Decision Tree
            # -----------------------------------------------------------
            if task_type == "Classification":
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )
            else:
                model = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )

            model.fit(X_train, y_train)

            # -----------------------------------------------------------
            # Basic Performance Metrics
            # -----------------------------------------------------------
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy:** {accuracy:.4f}")

                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix:")
                st.write(cm)

                report = classification_report(y_test, y_pred, output_dict=True)
                st.write("Classification Report (Precision, Recall, F1-score):")
                st.write(pd.DataFrame(report).transpose())

            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**R² (Coefficient of Determination):** {r2:.4f}")

            # -----------------------------------------------------------
            # Tree Statistics & Interpretation
            # -----------------------------------------------------------
            st.write("### Decision Tree Statistics & Interpretation")
            # Number of leaves & depth
            n_leaves = model.get_n_leaves()
            depth = model.get_depth()
            st.write(f"- **Number of leaves**: {n_leaves}")
            st.write(f"- **Max depth**: {depth}")

            # Textual description of the tree (rules)
            st.write("**Tree Rules (Split-by-Split Interpretation):**")
            tree_rules = export_text(model, feature_names=col_names)
            st.code(tree_rules)

            st.markdown(
                """
                **How to read this**:
                - Each `|---` indicates a split level in the tree.
                - `gini` (or other impurity/criterion) describes the impurity at that node.
                - `samples` is how many training samples passed through that node.
                - `value` is either the distribution of classes (classification) 
                  or the mean value (regression).
                - Leaf nodes (no further splits) show the final prediction.
                """
            )

            # -----------------------------------------------------------
            # Variable Importance
            # -----------------------------------------------------------
            st.write("### Variable Importance")
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_idx]
            sorted_features = np.array(col_names)[sorted_idx]

            importance_df = pd.DataFrame({
                "Feature": sorted_features,
                "Importance": sorted_importances
            })
            st.write(importance_df)

            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            # -----------------------------------------------------------
            # Tree Visualization
            # -----------------------------------------------------------
            st.write("### Decision Tree Plot")
            fig, ax = plt.subplots(figsize=(12, 8))
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
                ax=ax
            )
            st.pyplot(fig)

if __name__ == "__main__":
    main()


