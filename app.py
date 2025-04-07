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
    r2_score
)
import io  # for enabling file downloads in-memory

def main():
    # ----------------------------------------------------------------
    # App Title & Introductory Description
    # ----------------------------------------------------------------
    st.title("Decision Tree Builder & Interpreter (For Non-Coders)")

    st.write(
        """
        ### Welcome to the Decision Tree Builder!  

        This app will guide you to:
        1. **Upload** a CSV data file.\n
        2. Choose **Classification** (if your target is categorical) or **Regression** (if your target is numeric).\n
        3. Select your **Target** variable (the outcome you want to predict) and **Predictor** variables (features).\n
        4. Decide how each predictor is **scaled** (numeric or categorical).\n
        5. Adjust **hyperparameters** to control how the decision tree is grown.\n
        6. **Train** the model and view:\n
           - **Performance metrics** (accuracy, confusion matrix for classification; RMSE, R² for regression)\n
           - **Feature Importance** (which predictors matter most)\n
           - A **high-resolution Tree Figure** for direct interpretation\n
           - **Text-based** tree splits if the tree ends up very deep\n

        #### How to Decide Between Classification and Regression
        - **Classification**: Your target variable is a **category** (e.g., pass/fail, low/medium/high, yes/no).\n
        - **Regression**: Your target variable is **numeric** and continuous (e.g., a score, an amount, or a range of values).

        #### How to Use Predictor Scales
        - **Numeric**: Any continuous number field (e.g., scores, ages, incomes).\n
        - **Categorical**: Discrete labels/groups (e.g., gender, level of education, region).  
          The app automatically **one-hot encodes** these for you.

        #### Guidance on Hyperparameters
        - **Criterion**: Defines how splits are measured.  
          - Classification: "gini", "entropy", or "log_loss".  
          - Regression: "squared_error", "absolute_error", etc.  
        - **Max Depth**: Upper limit on how many times the tree can split. Larger = more complex but can overfit.\n
        - **Min Samples Split**: The minimum number of samples a node must have before it’s eligible for further splitting.

        Let's get started!
        """
    )

    # ----------------------------------------------------------------
    # Step 1: CSV Upload
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Step 1: Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write("**Data Dimensions**:")
        st.write(f"- **Rows**: {df.shape[0]}")
        st.write(f"- **Columns**: {df.shape[1]}")

        # ------------------------------------------------------------
        # Step 2: Classification or Regression
        # ------------------------------------------------------------
        st.write("### Step 2: Classification or Regression?")
        task_type = st.selectbox(
            "Choose the problem type:",
            ["Classification", "Regression"],
            help=(
                "Classification if your target is categorical (e.g., pass/fail). "
                "Regression if your target is numeric (e.g., test score, amount)."
            )
        )

        # ------------------------------------------------------------
        # Step 3: Select Target Variable
        # ------------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox(
            "Step 3: Select your Target (outcome) variable",
            all_columns,
            help="Which column do you want the model to predict?"
        )

        # ------------------------------------------------------------
        # Step 4: Select Predictor Variables & Scale
        # ------------------------------------------------------------
        st.write("### Step 4: Choose Your Predictor Variables & Scale")
        possible_predictors = [col for col in all_columns if col != target_col]
        selected_predictors = st.multiselect(
            "Predictors (features) that might influence the target:",
            possible_predictors,
            default=possible_predictors,
            help=(
                "Pick columns that could help predict or explain your target. "
                "You can select none, some, or all from the dataset."
            )
        )

        st.write("#### Indicate Numeric or Categorical for Each Selected Predictor")
        scale_of_measurement = {}
        for pred in selected_predictors:
            scale_choice = st.selectbox(
                f"Scale for '{pred}':",
                ["numeric", "categorical"],
                key=f"scale_{pred}",
                help="Is this feature truly numeric (continuous values) or categorical (labels/groups)?"
            )
            scale_of_measurement[pred] = scale_choice

        # ------------------------------------------------------------
        # Step 5: Decision Tree Hyperparameters
        # ------------------------------------------------------------
        st.write("### Step 5: Decision Tree Hyperparameters")

        # Criterion
        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion (Classification)",
                ["gini", "entropy", "log_loss"],
                help=(
                    "Measures how 'pure' each node is after a split. "
                    "gini and entropy are most common. log_loss is another metric."
                )
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion (Regression)",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                help=(
                    "For regression, typical choices include squared_error or absolute_error. "
                    "squared_error is the default in many tree implementations."
                )
            )

        max_depth = st.slider(
            "Max Depth (0 = no limit)",
            min_value=0,
            max_value=20,
            value=3,  # A moderate default
            step=1,
            help=(
                "Maximum number of levels the tree can have. A small number can be more interpretable; "
                "a larger number might capture more complexity but can overfit."
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
                "A node must have at least this many samples before it's considered for splitting. "
                "Increasing this can reduce overfitting by preventing very narrow splits."
            )
        )

        # ------------------------------------------------------------
        # Train Model Button
        # ------------------------------------------------------------
        if st.button("Step 6: Train the Decision Tree"):
            # Validate that we have predictors selected
            if not selected_predictors:
                st.error("No predictors were selected. Please choose at least one.")
                return

            # Build X using appropriate encodings
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
                st.error("Something went wrong while building your feature matrix.")
                return

            X = pd.concat(X_list, axis=1)
            y = df[target_col]

            # Split data into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Initialize the Decision Tree
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

            # ------------------------------------
            # Performance Metrics
            # ------------------------------------
            st.subheader("Model Performance")
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                # Accuracy
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy**: {accuracy:.4f}")
                st.write(
                    "Proportion of test samples predicted correctly. "
                    "An accuracy of 1.0 means perfect predictions on the test set."
                )

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)
                st.write(
                    "Each row shows the actual class; each column shows the predicted class. "
                    "Numbers on the diagonal represent correct predictions."
                )

                # Classification Report
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report** (Precision, Recall, F1-score):")
                st.write(pd.DataFrame(report).transpose())
                st.write(
                    """
                    - **Precision**: Among the items labeled as a certain class, how many are truly that class?\n
                    - **Recall**: Among the items that are truly a certain class, how many did the model correctly detect?\n
                    - **F1-score**: Harmonic mean of Precision and Recall, balances both.\n
                    """
                )
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**RMSE (Root Mean Squared Error)**: {rmse:.4f}")
                st.write(
                    "On average, how far off your predictions are from the actual values. "
                    "Lower is generally better."
                )
                st.write(f"**R² (Coefficient of Determination)**: {r2:.4f}")
                st.write(
                    "Proportion of variance in the target explained by the model. 1.0 = perfect, 0.0 = none."
                )

            # ------------------------------------
            # Feature Importance
            # ------------------------------------
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_idx]
            sorted_features = np.array(col_names)[sorted_idx]

            fi_df = pd.DataFrame({
                "Feature": sorted_features,
                "Importance": sorted_importances
            })

            st.write("**Which features were used the most by the model?**")
            st.write(fi_df)

            st.write(
                "A higher 'importance' suggests that feature was more critical in splitting the nodes "
                "to reduce impurity (classification) or reduce error (regression)."
            )

            # Plot the importance
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            # Download button for Feature Importance
            buf_imp = io.BytesIO()
            fig_imp.savefig(buf_imp, format="png", dpi=300, bbox_inches="tight")
            buf_imp.seek(0)
            st.download_button(
                label="Download Feature Importance Plot (PNG)",
                data=buf_imp,
                file_name="feature_importance_plot.png",
                mime="image/png"
            )

            # ------------------------------------
            # Decision Tree Figure
            # ------------------------------------
            st.subheader("Decision Tree Figure")
            st.write(
                "This is a high-resolution visualization of your decision tree. "
                "For large trees, the diagram can be big. Scroll or zoom as needed."
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
                ax=ax_tree
            )
            st.pyplot(fig_tree)

            # Explanation of the figure:
            st.write(
                """
                ### How to Interpret the Tree Figure:
                
                - Each **box** (node) represents a group of samples from your training data.\n
                - **samples**: how many training examples ended up in that node.\n
                - **value**:\n
                  - **Classification**: distribution of samples across classes in that node (e.g., [50, 10]).\n
                  - **Regression**: the average target value of that node.\n
                - **impurity**:\n
                  - **Classification**: Gini or Entropy, a measure of how mixed the classes are (0 = perfectly pure, 1 = very mixed).\n
                  - **Regression**: MSE or another error measure for how spread out the target values are.\n
                - **Splits**:\n
                  - A node typically splits into two child nodes (for binary trees) based on a condition like `Feature <= some_value`.\n
                  - Samples that satisfy the condition go to the left branch, others go to the right.\n
                - **Leaf nodes** are nodes with no further splits (final predictions). For classification, it shows which class it predicts; for regression, a numeric estimate.\n
                
                #### Finding Patterns:
                - **Leaf Nodes** with **high purity** (classification) or **extreme values** (regression) can be very revealing.\n
                - You can trace a path from the **root** (top node) down to a **leaf** to see exactly which rules define that subgroup.\n
                - Check **sample sizes** in each node: a tiny node might indicate a niche pattern, while a large node suggests a broad segment of the data.\n
                - If your tree is extremely large (many levels), you can either limit `max_depth` or increase `min_samples_split` to keep it more interpretable.\n
                """
            )

            # Download the tree figure
            buf_tree = io.BytesIO()
            fig_tree.savefig(buf_tree, format="png", dpi=300, bbox_inches="tight")
            buf_tree.seek(0)
            st.download_button(
                label="Download Decision Tree (PNG)",
                data=buf_tree,
                file_name="decision_tree_plot.png",
                mime="image/png"
            )

            # ------------------------------------
            # Text-based Tree if Depth > 3
            # ------------------------------------
            final_depth = model.get_depth()
            if final_depth and final_depth > 3:
                st.warning(
                    f"Your tree has an actual depth of {final_depth}, which can be hard to visualize. "
                    "Below is a text-based summary for clarity:"
                )
                tree_text = export_text(model, feature_names=col_names)
                st.code(tree_text)
                st.write(
                    """
                    **Reading the text-based tree**:\n
                    - `|---` indicates deeper splits.\n
                    - `samples` = how many training samples in that node.\n
                    - `value` = For classification, distribution of classes; for regression, the mean.\n
                    - `impurity` = Gini/Entropy or MSE.\n
                    Follow the splits from top to bottom to see how the data is routed.
                    """
                )

    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()


