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
import io  # For managing in-memory file downloads

def main():
    # ----------------------------------------------------------------
    # Page Title & Introduction
    # ----------------------------------------------------------------
    st.title("Interactive Decision Tree Builder for Non-Coders")

    st.write(
        """
        **This app helps you build and interpret Decision Trees without writing code.**  
        
        ### Step-by-Step Instructions
        
        1. **Upload Your Data**: Provide a CSV file containing your dataset.\n
        2. **Decide: Classification or Regression?**  
           - **Classification**: Your target (outcome) variable has distinct categories (e.g., "Pass"/"Fail", "High"/"Medium"/"Low", "Yes"/"No").\n
           - **Regression**: Your target variable is numeric/continuous (e.g., scores, incomes, amounts) and can take many possible values.\n
        3. **Select Your Target (Outcome) Variable**.\n
        4. **Select Predictor Variables (Features)** and tell the app which are **numeric** (continuous) vs. **categorical** (labels, groups).\n
        5. **Adjust Decision Tree Hyperparameters**:\n
           - **Criterion**: For Classification, `gini`, `entropy`, or `log_loss`; for Regression, `squared_error`, `absolute_error`, etc.\n
           - **Max Depth**: How many times the tree can split. Higher = more complex tree. 0 = no limit.\n
           - **Min Samples Split**: How many samples must be in a node before the tree considers splitting.\n
        6. **Click Train Model**. The app will:\n
           - Show **Model Performance** (Accuracy, Confusion Matrix for Classification or RMSE/R² for Regression)\n
           - Display **Decision Tree Rules** (a textual breakdown)\n
           - Show a **Feature Importance** ranking and bar chart\n
           - Generate a **High-Resolution Decision Tree Figure** with a download button\n
        
        ### How to Decide Between Classification or Regression
        - If your **target variable** is **categorical** (like "Pass" vs. "Fail" or "Class A" vs. "Class B"), you have a **Classification** problem.\n
        - If your **target variable** is **numeric** (like exam scores, prices, or amounts) and can take many values, it's **Regression**.\n
        
        ### Why Adjust Hyperparameters?
        - **Max Depth**: A deeper tree can find more complex patterns but might overfit (not generalize well). If you see very high accuracy on training but poor performance on test data, consider lowering max depth.\n
        - **Min Samples Split**: If you increase this value, the tree won't split on very small subsets, which can reduce overfitting.\n
        
        Let's get started!
        """
    )

    # ----------------------------------------------------------------
    # 1. CSV Upload
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Convert the uploaded CSV into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write("**Data Dimensions**:")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        # ----------------------------------------------------------------
        # 2. Classification or Regression
        # ----------------------------------------------------------------
        st.write("### Is Your Outcome Variable Categorical or Numeric?")
        task_type = st.selectbox(
            "Classification or Regression?",
            ["Classification", "Regression"],
            help=(
                "Choose Classification if your target is categorical "
                "(e.g., pass/fail, yes/no, categories). "
                "Choose Regression if your target is numeric (e.g., scores, amounts)."
            )
        )

        # ----------------------------------------------------------------
        # 3. Target (Outcome) Variable
        # ----------------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox(
            "Which column is your Target (outcome)?",
            all_columns,
            help="Select the variable you want to predict."
        )

        # ----------------------------------------------------------------
        # 4. Predictor (Feature) Variables & Scale of Measurement
        # ----------------------------------------------------------------
        st.write("### Select Predictor (Feature) Variables")
        possible_predictors = [c for c in all_columns if c != target_col]
        selected_predictors = st.multiselect(
            "Choose the columns to use as predictors:",
            possible_predictors,
            default=possible_predictors,
            help="Pick the features (columns) that you think influence or predict the target."
        )

        st.write("### Indicate if Each Predictor is Numeric or Categorical")
        scale_of_measurement = {}
        for pred in selected_predictors:
            scale_choice = st.selectbox(
                f"'{pred}' is:",
                ["numeric", "categorical"],
                key=f"scale_{pred}",
                help=(
                    "If the column contains numbers like scores, amounts, or continuous data, choose numeric. "
                    "If it has discrete categories, choose categorical."
                )
            )
            scale_of_measurement[pred] = scale_choice

        # ----------------------------------------------------------------
        # 5. Decision Tree Hyperparameters
        # ----------------------------------------------------------------
        st.write("### Decision Tree Hyperparameters")

        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion",
                ["gini", "entropy", "log_loss"],
                help=(
                    "Decides how the tree measures 'purity' of a node. "
                    "'gini' is common, 'entropy' is from information theory, 'log_loss' can also be used."
                )
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                help=(
                    "For regression trees, these criteria measure how good a split is. "
                    "'squared_error' is typical, 'absolute_error' is more robust to outliers."
                )
            )

        max_depth = st.slider(
            "Max Depth (0 = unlimited)",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help=(
                "Deeper trees can capture more complex relationships but risk overfitting. "
                "Set 0 for no limit."
            )
        )
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider(
            "Min Samples Split (2 = very flexible)",
            min_value=2,
            max_value=50,
            value=2,
            step=1,
            help=(
                "Minimum number of samples required to split an internal node. "
                "A larger number can help reduce overfitting by preventing too many tiny splits."
            )
        )

        # ----------------------------------------------------------------
        # 6. Train Model Button
        # ----------------------------------------------------------------
        if st.button("Train Model"):
            # ----------------------------------------
            # Validate that we have predictors
            # ----------------------------------------
            if len(selected_predictors) == 0:
                st.error("No predictors were selected. Please choose at least one.")
                return

            # ----------------------------------------
            # Build X with proper encoding
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
                st.error("Failed to build your feature matrix. Check your selections.")
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
            # Initialize and Fit the Decision Tree
            # ----------------------------------------
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

            # ----------------------------------------------------------------
            # Model Performance Section
            # ----------------------------------------------------------------
            st.subheader("Model Performance")
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                # Accuracy
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy**: {accuracy:.4f}")
                st.write(
                    "Accuracy is the fraction of test samples the model predicts correctly.\n"
                    "1.0 means perfectly correct predictions; 0.0 means no correct predictions."
                )

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)
                st.write(
                    "Rows often represent the **true classes**, columns represent the **predicted classes**. "
                    "Diagonal cells (top-left to bottom-right) are correct predictions."
                )

                # Classification Report
                cr = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report** (Precision, Recall, F1-score):")
                st.write(pd.DataFrame(cr).transpose())
                st.write(
                    "**Precision**: Out of the predicted positives, how many were correct?\n"
                    "**Recall**: Out of the actual positives, how many did we correctly predict?\n"
                    "**F1-score**: The harmonic mean of Precision and Recall, balancing both."
                )

            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.write(f"**RMSE (Root Mean Squared Error)**: {rmse:.4f}")
                st.write(
                    "On average, how far off your predictions are from the actual values."
                )
                st.write(f"**R² (Coefficient of Determination)**: {r2:.4f}")
                st.write(
                    "How much variance in the target is explained by the model. "
                    "1.0 = perfect fit, 0.0 = no explanatory power."
                )

            # ----------------------------------------------------------------
            # Decision Tree Interpretation
            # ----------------------------------------------------------------
            st.subheader("Decision Tree Interpretation (Rules & Structure)")

            # Number of leaves and depth
            n_leaves = model.get_n_leaves()
            depth = model.get_depth()
            st.write(f"**Number of Leaves**: {n_leaves}")
            st.write(f"**Max Depth**: {depth}")
            st.write(
                "A **leaf** is an end node (where no more splitting occurs). "
                "The **depth** is how many splits from the root to the deepest leaf."
            )

            # Textual Breakdown
            st.write("**Text-Based Tree Rules**:")
            tree_text = export_text(model, feature_names=col_names)
            st.code(tree_text)

            st.write(
                "**How to read this**:\n"
                "- `|---` indicates deeper levels of splitting.\n"
                "- `samples` = how many training data points fall into that node.\n"
                "- `value` = \n"
                "  - For Classification: how many samples of each class are in that node.\n"
                "  - For Regression: the average (mean) target value in that node.\n"
                "- `gini`, `entropy`, or `mse` are measures of impurity. **Lower** = more homogeneous.\n"
                "If you see 'mse', it's the mean squared error within that node. Some packages show `SE` (standard error)."
            )

            # Tips for finding interesting patterns
            st.write(
                """
                **Tips for Identifying Interesting Patterns**:
                - Look for **leaf nodes** where the predicted value or majority class is **very different** from the overall average. 
                  That suggests a unique subgroup.\n
                - Examine the **sequence of splits** leading to that leaf. For example, if certain splits produce a subgroup that 
                  does extremely well (or poorly) on the target, that path may represent an actionable insight.\n
                - Compare the **number of samples** (in `samples=...`) for each leaf. A large leaf with a distinctly different 
                  predicted value might be very important for real-world decisions.\n
                - In **Classification Trees**, watch for leaves with a high purity for a particular class. This can signal 
                  strong indicators or risk factors.\n
                - In **Regression Trees**, check for large negative or positive leaf predictions to see which subgroup 
                  is driving extreme outcomes.
                """
            )

            # ----------------------------------------------------------------
            # Feature Importance
            # ----------------------------------------------------------------
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_idx]
            sorted_features = np.array(col_names)[sorted_idx]

            fi_df = pd.DataFrame({
                "Feature": sorted_features,
                "Importance": sorted_importances
            })
            st.write("**Ranking of Features** (Higher = More Important):")
            st.write(fi_df)

            st.write(
                "A higher importance value means that feature is used more often (or more effectively) "
                "to reduce impurity in the tree's splits. It's a rough guide to which features matter most."
            )

            # Feature importance bar chart
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

            # ----------------------------------------------------------------
            # Decision Tree Figure
            # ----------------------------------------------------------------
            st.subheader("High-Resolution Decision Tree Figure")
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
                "**How to read this figure**:\n"
                "- Each **box** is a **node** in the tree. The top box is the **root node** (includes all training data).\n"
                "- A node splits into **branches** based on a condition (e.g., `Feature <= value`).\n"
                "- **samples**: how many training samples are in that node.\n"
                "- **value** (Classification): distribution of samples across classes.\n"
                "- **value** (Regression): the average target value in that node.\n"
                "- **impurity**: how mixed the node is (e.g., Gini/Entropy for classification, MSE for regression).\n"
                "- The **color** or saturation often indicates the majority class (classification) or the magnitude of the predicted value (regression)."
            )

            # Download button for the tree plot
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
                "You can download this figure and include it in your reports or presentations!"
            )

    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()



