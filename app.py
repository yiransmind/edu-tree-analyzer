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
    # ---------------------------------------------------------
    # 1. App Introduction and Detailed Guidance
    # ---------------------------------------------------------
    st.title("Comprehensive Decision Tree Builder & Interpreter")
    
    st.write("""
    ### Welcome!
    
    This app is designed for anyone—especially non-coders—who wants to build, explore, and understand Decision Trees. 
    It will walk you through every step, from uploading your data to interpreting the final model.
    
    **What does this app do?**
    - **Upload Your Data:** Provide a CSV file containing your dataset.
    - **Choose Problem Type:** Decide whether your problem is a **Classification** problem (predicting categories like Pass/Fail) or a **Regression** problem (predicting continuous numbers like test scores).
    - **Select Variables:** Pick the target (what you want to predict) and the predictors (the features that might explain the target). You will also indicate whether each predictor is numeric (continuous data) or categorical (discrete groups).
    - **Set Hyperparameters:** Adjust settings such as the splitting criterion, maximum depth of the tree, and the minimum number of samples required to split a node. These decisions affect how complex your tree will be.
    - **Train the Model:** The app will train a Decision Tree based on your settings and show you performance metrics.
    - **Interpret Results:** Detailed explanations will help you understand:
         - **Model Performance:** Metrics like Accuracy, RMSE, R², etc.
         - **Feature Importance:** Which predictors were most influential.
         - **The Decision Tree Figure:** How to read each node (what “samples”, “value”, and “impurity” mean) and how to trace a path from the root to a leaf.
         - **Text-Based Summary:** If your tree is very deep, a text-based breakdown is provided for clarity.
    
    **Guidance for Decision Making:**
    - **Classification vs. Regression:** 
      - Use **Classification** if your target is a category (for example, “Pass” vs. “Fail”, “Yes” vs. “No”).
      - Use **Regression** if your target is a continuous number (for example, exam scores, incomes, temperatures).
    - **Choosing Predictors:** Think about what factors might logically influence your target. The app even explains how to handle categorical data by converting it into dummy variables.
    - **Setting Hyperparameters:** 
      - **Splitting Criterion:** This measures the quality of a split. For classification, options like 'gini' or 'entropy' are common. For regression, 'squared_error' is used—this is equivalent to Mean Squared Error (MSE) even though it is labeled as “squared_error”.
      - **Max Depth:** Controls the complexity of the tree. A shallower tree (e.g., a depth of 3) is easier to interpret, while a deeper tree might capture more detail but can be overwhelming.
      - **Min Samples Split:** This prevents the tree from splitting on very few samples, reducing the risk of overfitting.
    
    Follow the steps in order, and detailed instructions will appear at each stage. Enjoy exploring your data!
    """)

    # ---------------------------------------------------------
    # 2. Data Upload
    # ---------------------------------------------------------
    st.header("Step 1: Upload Your Data")
    st.write("""
    **Instructions:**
    - Click the 'Browse' button below to select your CSV file.
    - The app will display a preview of your data along with its dimensions.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error("Error reading CSV file. Please check the file format and try again.")
            st.error(e)
            return

        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")
    else:
        st.info("Please upload a CSV file to continue.")
        return

    # ---------------------------------------------------------
    # 3. Choose Problem Type (Classification vs. Regression)
    # ---------------------------------------------------------
    st.header("Step 2: Choose Your Problem Type")
    st.write("""
    **How to Decide:**
    - **Classification:** Use this option if your target variable is categorical. 
      For example, if you are predicting if a student passes or fails, use Classification.
    - **Regression:** Use this option if your target variable is numeric and continuous.
      For example, if you are predicting a student’s exam score, use Regression.
    """)
    
    task_type = st.selectbox(
        "Select Problem Type",
        ["Classification", "Regression"],
        help="Select 'Classification' if your target variable consists of categories; select 'Regression' if it is numeric."
    )

    # ---------------------------------------------------------
    # 4. Select Target and Predictor Variables
    # ---------------------------------------------------------
    st.header("Step 3: Select Variables")
    st.write("""
    **Target Variable:**
    - Choose the column you want to predict (your outcome).
    
    **Predictor Variables:**
    - Choose the columns that might influence the target.
    - For each predictor, indicate whether it is **numeric** (e.g., scores, age) or **categorical** (e.g., gender, region).
      The app will automatically convert categorical predictors into a format suitable for the model (using one-hot encoding).
    """)
    
    all_columns = df.columns.tolist()
    target_col = st.selectbox(
        "Select your Target Variable",
        all_columns,
        help="This is the outcome you want to predict."
    )
    
    possible_predictors = [col for col in all_columns if col != target_col]
    selected_predictors = st.multiselect(
        "Select Predictor Variables",
        possible_predictors,
        default=possible_predictors,
        help="Select one or more columns that could be used to predict your target."
    )
    
    st.write("#### Specify the Data Type for Each Predictor")
    scale_info = {}
    for pred in selected_predictors:
        user_scale = st.selectbox(
            f"Is '{pred}' Numeric or Categorical?",
            ["numeric", "categorical"],
            key=f"scale_{pred}",
            help="Select 'numeric' if this column contains continuous values; 'categorical' if it contains labels or groups."
        )
        scale_info[pred] = user_scale

    # ---------------------------------------------------------
    # 5. Set Decision Tree Hyperparameters
    # ---------------------------------------------------------
    st.header("Step 4: Set Hyperparameters")
    st.write("""
    **Hyperparameters Control Your Tree's Complexity:**
    
    - **Splitting Criterion:** 
      - For Classification: Options include **gini** (measures impurity) and **entropy** (information gain).
      - For Regression: The option **squared_error** is commonly used, which is equivalent to Mean Squared Error (MSE). Note that scikit-learn labels it as 'squared_error'.
    
    - **Max Depth:** 
      - This defines how many splits (levels) the tree can have.
      - A smaller max depth (e.g., 3 or 4) makes the tree easier to interpret.
      - A larger max depth might capture more detail but can be harder to understand.
    
    - **Min Samples Split:** 
      - This is the minimum number of samples required in a node before a split is attempted.
      - Increasing this value can reduce overfitting by preventing splits on very few samples.
    
    Adjust these values based on the complexity you expect in your data.
    """)
    
    if task_type == "Classification":
        criterion = st.selectbox(
            "Choose Splitting Criterion (Classification)",
            ["gini", "entropy", "log_loss"],
            help="Select a criterion to measure node purity for classification."
        )
    else:
        criterion = st.selectbox(
            "Choose Splitting Criterion (Regression)",
            ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            help=("Select a criterion for regression. 'squared_error' is commonly used and represents MSE, "
                  "even though it is labeled as such in the tree.")
        )
    
    max_depth = st.slider(
        "Set Maximum Tree Depth (0 for no limit)",
        min_value=0,
        max_value=20,
        value=3,
        help="A smaller depth leads to a simpler and more interpretable tree."
    )
    max_depth = None if max_depth == 0 else max_depth
    
    min_samples_split = st.slider(
        "Set Minimum Samples Required to Split a Node",
        min_value=2,
        max_value=50,
        value=2,
        help="Higher values prevent splits on very few samples, reducing overfitting."
    )

    # ---------------------------------------------------------
    # 6. Train the Decision Tree Model
    # ---------------------------------------------------------
    st.header("Step 5: Train Your Model")
    st.write("""
    Click the button below to train the decision tree model using your selected settings. 
    The app will then display performance metrics, feature importance, and a high-resolution tree figure.
    """)
    
    if st.button("Train Decision Tree"):
        if not selected_predictors:
            st.error("Please select at least one predictor variable.")
            return

        # Build feature matrix X with proper encoding
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

        if len(X_parts) == 0:
            st.error("Error constructing feature matrix. Please check your selections.")
            return
        
        X = pd.concat(X_parts, axis=1)
        y = df[target_col]

        # Split data into training and testing sets (70/30 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Initialize and train the decision tree model
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
        st.success("Model training complete!")
        
        # ---------------------------------------------------------
        # 7. Display Model Performance Metrics
        # ---------------------------------------------------------
        st.header("Model Performance Metrics")
        y_pred = model.predict(X_test)
        if task_type == "Classification":
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write("""
            **Interpretation:** Accuracy measures the proportion of correct predictions. 
            A value of 1.0 means perfect prediction, while 0.0 indicates that none of the predictions were correct.
            """)
            
            cm = confusion_matrix(y_test, y_pred)
            st.write("**Confusion Matrix:**")
            st.write(cm)
            st.write("""
            **How to Read the Confusion Matrix:**
            - Rows represent the actual classes.
            - Columns represent the predicted classes.
            - Diagonal elements indicate correct predictions.
            """)
            
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write("**Classification Report:**")
            st.write(pd.DataFrame(report).transpose())
            st.write("""
            **Key Metrics:**
            - **Precision:** The proportion of positive identifications that were actually correct.
            - **Recall:** The proportion of actual positives that were identified correctly.
            - **F1-score:** The harmonic mean of precision and recall.
            """)
        else:
            mse_val = mean_squared_error(y_test, y_pred)
            rmse_val = np.sqrt(mse_val)
            r2_val = r2_score(y_test, y_pred)
            st.write(f"**RMSE:** {rmse_val:.4f}")
            st.write("""
            **Interpretation:** RMSE (Root Mean Squared Error) represents the average magnitude of the prediction errors. 
            Lower RMSE indicates better predictive performance.
            """)
            st.write(f"**R² (Coefficient of Determination):** {r2_val:.4f}")
            st.write("""
            **Interpretation:** R² measures the proportion of variance in the target variable that is explained by the model. 
            An R² of 1.0 indicates perfect prediction, while 0.0 indicates that the model does not explain any of the variability.
            """)
            st.write("""
            **Note for Regression:** In the decision tree figure, you may see nodes labeled with "squared_error". 
            This is the criterion used and it represents the Mean Squared Error (MSE) for that node.
            """)

        # ---------------------------------------------------------
        # 8. Display Feature Importance
        # ---------------------------------------------------------
        st.header("Feature Importance")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_features = np.array(col_names)[sorted_idx]
        
        fi_df = pd.DataFrame({
            "Feature": sorted_features,
            "Importance": sorted_importances
        })
        st.write("The table below ranks features by their importance in the model:")
        st.write(fi_df)
        st.write("""
        **How to Interpret Feature Importance:**
        - A higher importance value indicates that a feature was used more frequently or more effectively 
          to make splits that reduce error (in regression) or increase class purity (in classification).
        """)
        
        # Plot Feature Importance
        fig_imp, ax_imp = plt.subplots()
        ax_imp.bar(range(len(sorted_features)), sorted_importances)
        ax_imp.set_xticks(range(len(sorted_features)))
        ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
        ax_imp.set_ylabel("Importance")
        ax_imp.set_title("Feature Importance Bar Chart")
        st.pyplot(fig_imp)
        
        buf_imp = io.BytesIO()
        fig_imp.savefig(buf_imp, format="png", dpi=300, bbox_inches="tight")
        buf_imp.seek(0)
        st.download_button(
            label="Download Feature Importance Plot (PNG)",
            data=buf_imp,
            file_name="feature_importance.png",
            mime="image/png"
        )
        
        # ---------------------------------------------------------
        # 9. Display Decision Tree Figure
        # ---------------------------------------------------------
        st.header("Decision Tree Figure")
        st.write("""
        **Understanding the Tree Figure:**
        
        This high-resolution diagram shows the structure of your decision tree.
        
        **Key Elements in Each Node:**
        - **samples:** The number of training samples that reach the node.
        - **value:** 
          - In Classification: The distribution of samples among the different classes.
          - In Regression: The average (mean) target value in that node.
        - **impurity:** 
          - For Classification: Displays the impurity measure (e.g., Gini or Entropy) which indicates how mixed the classes are.
          - For Regression: Displays "squared_error" (which is the same as MSE).
        - **Splitting Rule:** The condition used to split the node (e.g., "Feature <= 2.5"). Samples satisfying the condition move to the left; others move to the right.
        - **Leaf Nodes:** These are terminal nodes where no further splitting occurs. They provide the final prediction for that subgroup.
        
        **How to Use This Information:**
        - **Trace the Path:** Follow the branches from the root to a leaf to understand the decision-making process.
        - **Identify Patterns:** Look at leaf nodes with high purity (in classification) or extreme average values (in regression). These can indicate interesting subgroups or insights.
        - **Node Sample Sizes:** Consider how many samples are in each node—a very small sample might indicate an outlier or overfitting.
        """)
        
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
            impurity=True,
            ax=ax_tree
        )
        st.pyplot(fig_tree)
        
        buf_tree = io.BytesIO()
        fig_tree.savefig(buf_tree, format="png", dpi=300, bbox_inches="tight")
        buf_tree.seek(0)
        st.download_button(
            label="Download Decision Tree Plot (PNG)",
            data=buf_tree,
            file_name="decision_tree_plot.png",
            mime="image/png"
        )
        
        # ---------------------------------------------------------
        # 10. Provide Text-Based Tree if the Tree is Deep
        # ---------------------------------------------------------
        final_depth = model.get_depth()
        if final_depth and final_depth > 3:
            st.warning(
                f"Your decision tree has a depth of {final_depth}, which can be challenging to interpret visually. "
                "A text-based breakdown is provided below for clarity."
            )
            tree_text = export_text(model, feature_names=col_names)
            st.code(tree_text)
            st.write("""
            **How to Read the Text-Based Tree:**
            - Each level of indentation (represented by "|---") shows a deeper split.
            - **samples:** The number of training samples in that node.
            - **value:** 
              - For Classification: The distribution of class labels.
              - For Regression: The mean target value. Note that, for regression trees, scikit-learn's text output does not display an impurity value.
            - This format helps you follow the sequence of splits even if the visual diagram is too complex.
            """)
    else:
        st.info("Upload a CSV file to start building your Decision Tree.")

if __name__ == "__main__":
    main()


