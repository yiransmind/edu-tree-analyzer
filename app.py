import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_text,
    plot_tree
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
import io  # To handle in-memory bytes for downloads

def main():
    # ----------------------------------------------------------------
    # Title and High-Level Description
    # ----------------------------------------------------------------
    st.title("Interactive Decision Tree App (Non-Coder Friendly)")

    st.write(
        """
        #### Welcome!  
        Use this app to create and interpret **Decision Trees** for your data.
        
        **Here's how to proceed**:
        1. **Upload** a CSV file of your data.\n
        2. Choose **Classification** or **Regression**.\n
        3. Select your **Target** (outcome) variable.\n
        4. Pick your **Predictors** (features) and their scale (numeric/categorical).\n
        5. Adjust any **Hyperparameters** you want.\n
        6. Click **Train Model**.\n
        
        The app will show you:
        - **Model Performance** (Accuracy and Confusion Matrix for classification, or RMSE and R² for regression)
        - **Decision Tree Rules** (step-by-step breakdown of each split in plain text)
        - **Feature Importance** (which predictors matter most)
        - **High-Resolution Decision Tree Figure** (with a detailed explanation and separate download options)
        """
    )

    # ----------------------------------------------------------------
    # 1. CSV Upload
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the CSV
        df = pd.read_csv(uploaded_file)

        # A quick look at the DataFrame
        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write("**Data Dimensions**:")
        st.write(f"- Rows: {df.shape[0]}")
        st.write(f"- Columns: {df.shape[1]}")

        # ----------------------------------------------------------------
        # 2. Classification or Regression
        # ----------------------------------------------------------------
        task_type = st.selectbox(
            "Is this a Classification or Regression problem?",
            ["Classification", "Regression"]
        )

        # ----------------------------------------------------------------
        # 3. Target Variable
        # ----------------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox("Which column is your Target (outcome)?", all_columns)

        # ----------------------------------------------------------------
        # 4. Predictors & Scale of Measurement
        # ----------------------------------------------------------------
        possible_predictors = [c for c in all_columns if c != target_col]
        selected_predictors = st.multiselect(
            "Select your Predictor (feature) columns:",
            possible_predictors,
            default=possible_predictors,  # or empty if you prefer
        )

        st.write("### Specify the Scale of Measurement for Each Predictor")
        scale_of_measurement = {}
        for pred in selected_predictors:
            scale_choice = st.selectbox(
                f"'{pred}' is:",
                ["numeric", "categorical"],
                key=f"scale_{pred}"
            )
            scale_of_measurement[pred] = scale_choice

        # ----------------------------------------------------------------
        # 5. Hyperparameters
        # ----------------------------------------------------------------
        st.write("### Decision Tree Hyperparameters")

        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion (How the tree splits nodes)",
                ["gini", "entropy", "log_loss"]
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion (How the tree splits nodes)",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"]
            )

        max_depth = st.slider(
            "Maximum Depth of the Tree (0 = No Limit)",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
        # If 0, we use None (meaning unlimited depth)
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider(
            "Minimum Samples Required to Split a Node",
            min_value=2,
            max_value=50,
            value=2,
            step=1
        )

        # ----------------------------------------------------------------
        # 6. Train Model
        # ----------------------------------------------------------------
        if st.button("Train Model"):
            # ----------------------------------------
            # Basic Input Validation
            # ----------------------------------------
            if len(selected_predictors) == 0:
                st.error("No predictor columns selected. Please choose at least one.")
                return

            # ----------------------------------------
            # Construct Feature Matrix X
            # ----------------------------------------
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

            if len(X_list) == 0:
                st.error("Something went wrong creating the feature matrix. Check your data.")
                return

            X = pd.concat(X_list, axis=1)
            y = df[target_col]

            # ----------------------------------------
            # Train-Test Split
            # ----------------------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # ----------------------------------------
            # Model Initialization & Fitting
            # ----------------------------------------
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

            # ----------------------------------------------------------------
            # Model Performance
            # ----------------------------------------------------------------
            st.subheader("Model Performance")
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy**: {accuracy:.4f}")
                st.write(
                    "**Explanation**: Accuracy is the fraction of predictions the model got right. "
                    "An accuracy of 1.0 means 100% correct predictions, whereas 0.0 means no correct predictions at all."
                )

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)
                st.write(
                    "**How to read**: Rows often represent actual classes; columns represent predicted classes. "
                    "The diagonal entries show the number of correct predictions for each class."
                )

                # Classification Report
                cr = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report** (Precision, Recall, F1-score):")
                st.write(pd.DataFrame(cr).transpose())
                st.write(
                    "**Precision**: Within the samples predicted as a certain class, how many are correct?\n"
                    "**Recall**: Out of all samples that truly belong to a class, how many did we predict correctly?\n"
                    "**F1-score**: Harmonic mean of Precision and Recall."
                )
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**RMSE (Root Mean Squared Error)**: {rmse:.4f}")
                st.write(
                    "**Explanation**: On average, the model's predictions deviate "
                    "from the actual values by this amount."
                )
                st.write(f"**R² (Coefficient of Determination)**: {r2:.4f}")
                st.write(
                    "**Explanation**: R² measures how well the model explains "
                    "the variability in the target. 1.0 is perfect, 0.0 means no explanatory power."
                )

            # ----------------------------------------------------------------
            # Decision Tree Interpretation (Rules & Node Stats)
            # ----------------------------------------------------------------
            st.subheader("Decision Tree Interpretation")

            n_leaves = model.get_n_leaves()
            depth = model.get_depth()
            st.write(f"**Number of Leaves**: {n_leaves}")
            st.write(f"**Depth of the Tree**: {depth}")
            st.write(
                "A **leaf** is a terminal node that makes a final prediction. "
                "Depth is how many times we can follow a split from the root node down to a leaf."
            )

            # Show textual rules
            tree_rules = export_text(model, feature_names=col_names)
            st.write("**Tree Rules** (Text Description):")
            st.code(tree_rules)
            st.write(
                "**How to read these text rules**:\n"
                "- Each `|---` indicates another level of splitting.\n"
                "- `samples` = number of training data points that reach that node.\n"
                "- `value` = \n"
                "  - For Classification: distribution of samples among classes in that node.\n"
                "  - For Regression: the mean (or average) target value at that node.\n"
                "- `gini`, `entropy`, `mse`, or `mae` (depending on your criterion) show **impurity** in the node.\n"
                "- A node without further splits is a **leaf** (final prediction)."
            )
            st.write(
                "For **Regression Trees**, you may see `mse` (mean squared error) or similar terms. "
                "Occasionally, you'll see something like `SE` or standard error in other outputs. "
                "They measure how spread out the target values are in that node. Lower is more homogeneous."
            )

            # ----------------------------------------------------------------
            # Feature Importance
            # ----------------------------------------------------------------
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]  # Descending order
            sorted_importances = importances[sorted_idx]
            sorted_features = np.array(col_names)[sorted_idx]

            importance_df = pd.DataFrame({
                "Feature": sorted_features,
                "Importance": sorted_importances
            })
            st.write("**Ranking** (highest = most important):")
            st.write(importance_df)

            st.write(
                "**Interpretation**: Feature Importance indicates how much each predictor "
                "contributes to the tree’s decisions. A higher value = more importance."
            )

            # Create a separate figure for the bar chart
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            st.write(
                "**How to read**: Each bar represents a feature. Taller bars mean the feature "
                "is used more frequently (or more effectively) to split the data."
            )

            # Provide a download button for the bar chart
            buf_imp = io.BytesIO()
            fig_imp.savefig(buf_imp, format="png", dpi=300, bbox_inches="tight")
            buf_imp.seek(0)
            st.download_button(
                label="Download Feature Importance Plot (PNG)",
                data=buf_imp,
                file_name="feature_importance_plot.png",
                mime="image/png"
            )

            # ----------------------------------------------------------------
            # Decision Tree Figure
            # ----------------------------------------------------------------
            st.subheader("Decision Tree Figure (High Resolution)")
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
                "### How to read the Decision Tree figure:\n"
                "- Each **box** represents a node. The **top box** is the **root** (all training samples).\n"
                "- A node splits into branches based on a condition (e.g., `feature <= value`).\n"
                "- **samples**: how many training examples are in that node.\n"
                "- **impurity**: measure of homogeneity (e.g., Gini for classification, MSE for regression).\n"
                "- **value**:\n"
                "  - For Classification, shows how many samples of each class are in the node.\n"
                "  - For Regression, shows the average target value in that node.\n"
                "- The **color** saturation sometimes reflects the majority class (for classification) "
                "or the magnitude of the mean target (for regression)."
            )

            # Provide a download button for the tree figure
            buf_tree = io.BytesIO()
            fig_tree.savefig(buf_tree, format="png", dpi=300, bbox_inches="tight")
            buf_tree.seek(0)
            st.download_button(
                label="Download Decision Tree Plot (PNG)",
                data=buf_tree,
                file_name="decision_tree_plot.png",
                mime="image/png"
            )

            st.write(
                "**Tip**: You can use these images in presentations or reports to communicate how "
                "the tree makes predictions."
            )

    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()



