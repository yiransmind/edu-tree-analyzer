import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree
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
    # Introduction / App Title
    # ----------------------------------------------------------------
    st.title("Interactive Decision Tree Builder")

    st.write(
        """
        **This app helps you create and visualize a Decision Tree** – ideal for non-coders!\n
        **How it works**:
        1. Upload a **CSV** file.\n
        2. Choose if your outcome (target) variable is **Classification** or **Regression**.\n
        3. Select your **Target** and **Predictor** variables, plus whether each predictor is numeric or categorical.\n
        4. Adjust **Tree Hyperparameters**.\n
        5. Click **Train Model**.\n
        The app will show:\n
        - **Performance** (accuracy, confusion matrix, classification report for classification; RMSE and R² for regression).\n
        - A **Feature Importance** chart (which predictors matter most).\n
        - A **High-Resolution Decision Tree Figure** – plus thorough guidance on how to read the node statistics.\n
        """
    )

    # ----------------------------------------------------------------
    # CSV File Upload
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write("**Data Dimensions**:")
        st.write(f"- Rows: {df.shape[0]}")
        st.write(f"- Columns: {df.shape[1]}")

        # ------------------------------------------------------------
        # Step 1: Classification or Regression
        # ------------------------------------------------------------
        st.write("### 1) Classification or Regression?")
        task_type = st.selectbox(
            "Select the type of Decision Tree:",
            ["Classification", "Regression"],
            help=(
                "Classification = Target is a discrete category (e.g., Pass/Fail). "
                "Regression = Target is a continuous numeric variable (e.g., a test score)."
            )
        )

        # ------------------------------------------------------------
        # Step 2: Select Target Variable
        # ------------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox(
            "Select your Target (outcome) variable:",
            all_columns,
            help="Which column do you want the model to predict?"
        )

        # ------------------------------------------------------------
        # Step 3: Select Predictor Variables
        # ------------------------------------------------------------
        st.write("### 2) Choose Predictor Variables & Their Scale")
        possible_predictors = [col for col in all_columns if col != target_col]
        selected_predictors = st.multiselect(
            "Select the columns to use as predictors:",
            possible_predictors,
            default=possible_predictors,
            help="Pick the features/columns that influence or predict your target."
        )

        scale_of_measurement = {}
        for pred in selected_predictors:
            scale_choice = st.selectbox(
                f"'{pred}' is:",
                ["numeric", "categorical"],
                key=f"scale_{pred}",
                help="Numeric for continuous values; categorical for discrete groups."
            )
            scale_of_measurement[pred] = scale_choice

        # ------------------------------------------------------------
        # Step 4: Decision Tree Hyperparameters
        # ------------------------------------------------------------
        st.write("### 3) Decision Tree Hyperparameters")
        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion",
                ["gini", "entropy", "log_loss"],
                help=(
                    "How the tree measures the purity of each node. 'gini' and 'entropy' are common. "
                    "'log_loss' is another option."
                )
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                help=(
                    "For regression, 'squared_error' is typical. 'absolute_error' can reduce the impact of outliers."
                )
            )

        max_depth = st.slider(
            "Max Depth (0 = unlimited)",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="How many splits deep the tree can go. Larger = more complex. 0 = no limit."
        )
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider(
            "Minimum Samples per Split",
            min_value=2,
            max_value=50,
            value=2,
            step=1,
            help=(
                "At least this many samples are required in a node to consider splitting it. "
                "Increasing can reduce overfitting."
            )
        )

        # ------------------------------------------------------------
        # Step 5: Train the Model
        # ------------------------------------------------------------
        if st.button("Train Model"):
            if len(selected_predictors) == 0:
                st.error("No predictors selected. Please choose at least one.")
                return

            # Build X with appropriate encoding
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
                st.error("Failed to build feature matrix. Check your predictor selections.")
                return

            X = pd.concat(X_list, axis=1)
            y = df[target_col]

            # Train-test split (70/30)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Initialize Decision Tree
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

            # Fit model
            model.fit(X_train, y_train)

            # ----------------------------------------------------------
            # Model Performance
            # ----------------------------------------------------------
            st.subheader("Model Performance")
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy**: {accuracy:.4f}")
                st.write(
                    "How many predictions the model got right overall.\n"
                    "An accuracy of 1.0 means 100% correct predictions."
                )

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)
                st.write(
                    "Rows represent true classes; columns represent predicted classes. "
                    "The diagonal shows correct predictions."
                )

                # Classification report
                cr = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report**:")
                st.write(pd.DataFrame(cr).transpose())
                st.write(
                    "- **Precision**: Out of all predicted positives, how many were correct?\n"
                    "- **Recall**: Out of all true positives, how many did we predict correctly?\n"
                    "- **F1-score**: A balance of precision and recall."
                )
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.write(f"**RMSE**: {rmse:.4f}")
                st.write(
                    "Root Mean Squared Error: on average, how far the predictions are from the actual values."
                )
                st.write(f"**R²**: {r2:.4f}")
                st.write(
                    "Coefficient of Determination: how much variance in the target is explained by the model. "
                    "1.0 = perfect fit, 0 = no explanatory power."
                )

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

            st.write("**Ranking of Predictors** (Higher = More Important):")
            st.write(fi_df)

            st.write(
                "A higher importance means that feature was more influential in splitting the data "
                "to reduce impurity (for classification) or reduce error (for regression)."
            )

            # Plot feature importance
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            # Download button for the feature importance plot
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
            st.subheader("Decision Tree Figure (High-Resolution)")

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
                ### How to Read the Decision Tree Figure:

                - **Nodes**: Each box is a node that represents a group of samples.\n
                - **Root Node**: The topmost node, containing all your training samples.\n
                - **Splits**: Nodes split into branches based on a rule, for example "Predictor <= some value". If true, samples go left; if false, samples go right (or vice versa).\n
                - **samples**: Shows how many training records (rows) are in that node.\n
                - **value**:\n
                  - For **Classification**: The number of samples belonging to each class in that node. The predicted class is often indicated too.\n
                  - For **Regression**: The average (mean) target value of the samples in that node.\n
                - **impurity**:\n
                  - For **Classification**: Typically Gini or Entropy. Lower means the node is more "pure" (dominated by one class).\n
                  - For **Regression**: Often MSE (mean squared error) – lower means predictions are more similar to each other.\n
                - **Color Saturation**: Usually indicates the node's predominant class (classification) or the magnitude of the predicted value (regression).\n
                
                ### Finding Interesting Patterns:
                - Look at **Leaf Nodes** (where no more splitting occurs):
                  - For **Classification**: A leaf node that has a high proportion of one class suggests that combination of feature conditions strongly predicts that class.\n
                  - For **Regression**: A leaf node with a very high or very low mean target value indicates a unique subgroup with extreme outcomes.\n
                - **Compare Leaves**: Are there leaves with significantly different predicted outcomes? That might indicate a meaningful subgroup.\n
                - **Sample Sizes**: If a leaf has a small number of samples, it might be a niche pattern or potentially overfitting.\n
                - **Trace the Path**: By following splits from the root to a leaf, you can see exactly which conditions lead to that prediction. This often uncovers actionable insights in educational or business contexts.\n

                > **Tip**: If the tree is very deep and complex, consider limiting the max depth or increasing min samples split to simplify it.
                """
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

            st.write(
                "Use this figure in reports or presentations to illustrate how your model makes predictions!"
            )

    else:
        st.info("Upload a CSV file to get started.")

if __name__ == "__main__":
    main()
