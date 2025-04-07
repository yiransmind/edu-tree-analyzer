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
    st.title("EduTree")

    st.write(
        """
        ### Welcome to the EduTree for decision tree analysis!

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
           - A **Text-Based Tree** if the tree is deeper than 3 levels\n

        #### Important Notes:
        - For **Regression Trees**, scikit-learn’s figure may display node impurity as "squared_error," 
          which is effectively **MSE** under the hood.  
        - For **Regression Trees**, the text-based tree (`export_text`) does **not** show the node’s impurity. 
          That’s just a scikit-learn quirk.
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

        st.markdown(
    """
    Hyperparameters control how your decision tree is built and can help prevent overfitting. 
    Use the controls below to select the splitting **criterion**, **maximum depth**, and **minimum samples** to split.

    **Guidelines for Choosing Hyperparameters:**
    - **Criterion (Classification)**:
      - *gini*: Often the default; measures how often a randomly chosen element would be incorrectly labeled.
      - *entropy*: Similar to gini, but can produce slightly different trees.
      - *log_loss*: Another impurity measure, sometimes slower to compute; not as common as gini or entropy.
    - **Criterion (Regression)**:
      - *squared_error*: Standard MSE-based splitting; often a good default.
      - *friedman_mse*: A variation on MSE that can improve performance in some cases.
      - *absolute_error*: Focuses on median-based splits, more robust to outliers.
      - *poisson*: Used for count data where the target is non-negative.
    - **Max Depth**:
      - A lower max depth (e.g., 3-5) produces simpler, more interpretable trees but may underfit.
      - A higher max depth can capture more nuances in the data but risks overfitting.
      - Setting this to 0 in the slider means no limit (i.e., the tree can grow until all leaves are pure).
    - **Minimum Samples to Split**:
      - Controls how many samples a node must have before a further split is considered.
      - Increasing this number generally reduces overfitting (by preventing very specific splits) 
        but might cause underfitting if set too high.
    """
)

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
                    "How to measure node quality for regression. 'squared_error' means MSE internally, "
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
                    "Proportion of correct predictions on the test set (range: 0 to 1)."
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
                    "RMSE (Root Mean Squared Error) is the square root of the average squared difference "
                    "between predicted and actual values."
                )
                st.write(f"**R²**: {r2_val:.4f}")
                st.write(
                    "R² (Coefficient of Determination) measures how much variance is explained by the model. "
                    "1.0 = perfect prediction, 0.0 = model explains nothing."
                )

            # --------------------------------------------------------
            # Extra Guidance on Interpreting Metrics
            # --------------------------------------------------------
            st.subheader("Interpreting These Metrics")
            if task_type == "Classification":
                st.markdown(
                    """
                    - **Accuracy**: How often the model correctly predicts the class.  
                      - *Interpretation*: 0.90 means 90% of the time, the model's prediction matches the true label.
                    - **Confusion Matrix**: Breaks down predictions vs. actual classes.  
                      - *Interpretation*: Helps to see if the model confuses certain classes more than others.
                    - **Classification Report** (Precision, Recall, F1-Score):  
                      - *Precision*: Of all predicted positives, how many are truly positive?  
                      - *Recall (Sensitivity)*: Of all actual positives, how many did we catch?  
                      - *F1-Score*: The harmonic mean of precision & recall.
                    """
                )
            else:
                st.markdown(
                    """
                    - **RMSE**: How far off predictions are, on average.  
                      - *Interpretation*: If RMSE is 10, the model’s predictions deviate from true values by about 10 units on average.
                    - **R²**: Proportion of variance in the target explained by the model.  
                      - *Interpretation*: An R² of 0.75 means the model explains 75% of the variability in the outcome.
                    """
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
                "to split nodes and reduce error (regression) or increase purity (classification)."
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
                impurity=True,  # Displays impurity in the figure
                ax=ax_tree
            )
            st.pyplot(fig_tree)

            st.write(
    """
    ### How to Read This Tree Figure

    This diagram shows how the Decision Tree splits your data step-by-step, from the **root node** at the top (all samples) down to **leaf nodes** with no further splits. Each node in the tree provides statistics about the subset of data that reached that node.

    **Key Elements to Look For:**

    - **samples**:  
      The total number of training observations (rows) that have flowed into this node after all previous splits.  
      - *Interpretation:* If a node says "samples = 120," it means 120 rows from your training set meet the conditions leading to that node.

    - **value**:  
      - **Classification**: A list (or array) indicating how many samples in this node belong to each class.  
        - *Example:* `value = [40, 80]` could mean there are 40 samples of class A and 80 samples of class B in this node.  
        - *Interpretation:* Whichever class count is larger typically indicates the node’s “majority class.”  
      - **Regression**: A single number showing the **mean** (average) target value among the samples in that node.  
        - *Example:* `value = 23.5` indicates that on average, the target value is 23.5 for this subset.

    - **impurity**:  
      Shows how “pure” (homogeneous) the node is, according to your chosen splitting criterion.  
      - **Classification**:  
        - If you selected "gini," it displays the Gini impurity (0 = perfectly pure, 0.5 for a 2-class problem = maximum impurity).  
        - If "entropy," it displays the entropy measure (0 = perfectly pure, higher values indicate more mixed classes).  
        - If "log_loss," it displays a loss-based impurity measure.  
      - **Regression**:  
        - If you chose "squared_error," it’s effectively displaying MSE at that node (how far off, on average, the predictions are from the true values in that node).  
        - Other criteria ("friedman_mse," "absolute_error," "poisson") have their own ways of measuring node quality.

    - **Splits**:  
      Each node has a rule like "`Feature <= x`," meaning all rows where that feature’s value is less than or equal to `x` go to the left child node, and rows above `x` go to the right child.  
      - *Interpretation:* This rule is chosen by the algorithm to best separate the data (reduce impurity or error).

    - **Leaf nodes**:  
      Nodes with no further splits. The **value** of a leaf is the prediction the tree makes for all samples that fall into that leaf.  
      - For classification, the final predicted class is typically the majority class in that leaf.  
      - For regression, the prediction is the average target value of that leaf.

    **Colors and Shading** (for classification trees):
    - Often, the background color of a node (if `filled=True`) indicates which class is most prevalent, and the intensity of the color can show how “pure” that node is. A deeper shade suggests the node is more dominated by a single class.

    **Overall Interpretation**:
    - Starting at the top, follow the splits (true/false or ≤ / >) that match a given row’s feature values until you reach a leaf node. That leaf’s **value** is then your model’s prediction for that row.
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
                    #### Note
                    - For Regression Trees, scikit-learn **does not show** the impurity measure (like MSE) in the text format.  
                    - For Classification, you'll see Gini/Entropy/Log Loss values in the text output.
                    """
                )

    else:
        st.info("Please upload a CSV file to start.")

if __name__ == "__main__":
    main()


