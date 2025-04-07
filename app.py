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
import io  # For handling in-memory file downloads

def main():
    # ----------------------------------------------------------------
    # Title and Description
    # ----------------------------------------------------------------
    st.title("Interactive Decision Tree App for Non-Coders")
    st.write(
        """
        ### This app helps you build and interpret Decision Trees without needing to code.
        Simply:
        1. Upload a CSV file of your dataset.\n
        2. Choose **Classification** or **Regression**.\n
        3. Select your **Target (outcome)** variable.\n
        4. Pick your **Predictor** variables and specify if they're **numeric** or **categorical**.\n
        5. Adjust any **Hyperparameters** you want.\n
        6. Click **Train Model**.\n
        The app will show you:
        - **Model Performance** (accuracy, confusion matrix for classification OR RMSE, R² for regression)\n
        - **Decision Tree Rules** (text interpretation of each split)\n
        - **Variable Importance** (which predictors matter most)\n
        - **A High-Resolution Tree Plot** (with a download option)\n
        """
    )

    # ----------------------------------------------------------------
    # Step 1: Upload CSV
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the CSV into a DataFrame
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.write(df.head(10))

        st.write("**Number of Rows**:", df.shape[0])
        st.write("**Number of Columns**:", df.shape[1])

        # ----------------------------------------------------------------
        # Step 2: Choose Classification or Regression
        # ----------------------------------------------------------------
        task_type = st.selectbox(
            "Select the type of Decision Tree you want to build:",
            ["Classification", "Regression"]
        )

        # ----------------------------------------------------------------
        # Step 3: Select Target Variable
        # ----------------------------------------------------------------
        all_columns = df.columns.tolist()
        target_col = st.selectbox("Which column is your Target (outcome)?", all_columns)

        # ----------------------------------------------------------------
        # Step 4: Select Predictors and Scale of Measurement
        # ----------------------------------------------------------------
        possible_predictors = [col for col in all_columns if col != target_col]
        selected_predictors = st.multiselect(
            "Select your Predictor Variables:",
            possible_predictors,
            default=possible_predictors  # or leave empty if you prefer
        )

        st.write("### Indicate Scale of Measurement for Each Predictor")
        scale_of_measurement = {}
        for pred in selected_predictors:
            user_choice = st.selectbox(
                f"{pred}",
                ["numeric", "categorical"],
                key=f"scale_{pred}"
            )
            scale_of_measurement[pred] = user_choice

        # ----------------------------------------------------------------
        # Step 5: Hyperparameters
        # ----------------------------------------------------------------
        st.write("### Decision Tree Hyperparameters")

        if task_type == "Classification":
            criterion = st.selectbox(
                "Splitting Criterion (How the tree decides splits)",
                ["gini", "entropy", "log_loss"],
            )
        else:
            criterion = st.selectbox(
                "Splitting Criterion (How the tree decides splits)",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            )

        max_depth = st.slider(
            "Maximum Depth of the Tree (0 = No Limit)",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
        max_depth = None if max_depth == 0 else max_depth

        min_samples_split = st.slider(
            "Minimum Samples Required to Split a Node",
            min_value=2,
            max_value=50,
            value=2,
            step=1
        )

        # ----------------------------------------------------------------
        # Step 6: Train Model Button
        # ----------------------------------------------------------------
        if st.button("Train Model"):

            # ------------------------------------------------------------
            # Validate Input
            # ------------------------------------------------------------
            if not selected_predictors:
                st.error("No predictors selected. Please choose at least one.")
                return

            # ------------------------------------------------------------
            # Build Feature Matrix (X) with proper encoding
            # ------------------------------------------------------------
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
                st.error("Error building feature matrix. Check your selections.")
                return

            X = pd.concat(X_list, axis=1)
            y = df[target_col]

            # ------------------------------------------------------------
            # Train-Test Split
            # ------------------------------------------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # ------------------------------------------------------------
            # Initialize and Fit the Decision Tree
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # Model Performance
            # ------------------------------------------------------------
            st.subheader("Model Performance")

            y_pred = model.predict(X_test)

            if task_type == "Classification":
                # Accuracy
                accuracy = accuracy_score(y_test, y_pred)
                st.write(
                    f"**Accuracy**: {accuracy:.4f}\n\n"
                    "**Explanation**: Accuracy is the proportion of correct predictions over all predictions."
                )

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("**Confusion Matrix**:")
                st.write(cm)
                st.write(
                    "**Explanation**: The Confusion Matrix shows how many samples were correctly "
                    "or incorrectly classified. Rows often represent Actual classes, and columns "
                    "represent Predicted classes."
                )

                # Classification Report
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write("**Classification Report** (Precision, Recall, F1-score):")
                st.write(pd.DataFrame(report).transpose())
                st.write(
                    "**Explanation**:\n"
                    "- **Precision**: Out of the samples predicted to be a certain class, how many are correct?\n"
                    "- **Recall**: Out of the samples that actually are a certain class, how many did we predict correctly?\n"
                    "- **F1-score**: Harmonic mean of Precision and Recall, balances both.\n"
                )
            else:
                # Regression Metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.write(
                    f"**Root Mean Squared Error (RMSE)**: {rmse:.4f}\n\n"
                    "**Explanation**: On average, the model's predictions deviate from the actual values by this amount."
                )
                st.write(
                    f"**R² (Coefficient of Determination)**: {r2:.4f}\n\n"
                    "**Explanation**: R² indicates how much of the variation in the target is explained by the model "
                    "(1.0 is a perfect fit, 0.0 means no explanatory power)."
                )

            # ------------------------------------------------------------
            # Decision Tree Interpretation
            # ------------------------------------------------------------
            st.subheader("Decision Tree Interpretation")

            # Number of leaves and depth
            n_leaves = model.get_n_leaves()
            depth = model.get_depth()
            st.write(
                f"**Number of Leaves**: {n_leaves}\n\n"
                f"**Depth of the Tree**: {depth}"
            )
            st.write(
                "**Explanation**:\n"
                "- **Leaves** are the final segments of the tree (where no further splitting happens).\n"
                "- **Depth** is how many layers of splits the tree has."
            )

            # Text-based rules
            st.write("**Tree Rules (Text Description)**:")
            tree_rules = export_text(model, feature_names=col_names)
            st.code(tree_rules)

            st.write(
                "**How to read these rules**:\n"
                "- Each `|---` indicates a deeper level of splitting.\n"
                "- For Classification, `value = [x, y, ...]` shows how many samples of each class are in that node.\n"
                "- For Regression, `value = [prediction_value]` is the average predicted value at that node.\n"
                "- `samples` shows how many samples from the training set reach that split or leaf.\n"
                "- `gini`/`entropy`/etc. indicates how pure the node is. **Lower** means more homogeneous data in that node."
            )

            # ------------------------------------------------------------
            # Variable Importance
            # ------------------------------------------------------------
            st.subheader("Variable (Feature) Importance")
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]  # Descending
            sorted_importances = importances[sorted_idx]
            sorted_features = np.array(col_names)[sorted_idx]

            importance_df = pd.DataFrame({
                "Feature": sorted_features,
                "Importance": sorted_importances
            })
            st.write(importance_df)

            st.write(
                "**Explanation**: These values indicate how much each feature "
                "contributes to splitting and improving predictions in the tree. "
                "Higher = more important."
            )

            # Make a bar plot of importance
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Feature Importances")
            # We'll keep the plot separate from the tree to comply with single-plot instructions
            st.pyplot(fig_imp)

            # ------------------------------------------------------------
            # Decision Tree Visualization
            # ------------------------------------------------------------
            st.subheader("Visual Decision Tree (High-Resolution)")

            # Create the figure
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # Higher DPI for better resolution
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

            st.write(
                "**Explanation**: Each box is a node. "
                "The top box is the root (includes all samples). "
                "Each subsequent split divides the data based on a feature/value. "
                "Leaf nodes show final predictions."
            )

            # ------------------------------------------------------------
            # Provide Download Button for the Figure
            # ------------------------------------------------------------
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label="Download Tree Plot (PNG)",
                data=buf,
                file_name="decision_tree_plot.png",
                mime="image/png"
            )

            st.write("Click the button above to download the tree plot as a PNG image.")
    
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()


