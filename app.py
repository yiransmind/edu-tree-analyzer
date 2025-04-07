import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt

def main():
    st.title("Decision Tree App for Educational (Social Science) Research")

    st.write(
        """
        **How to Use**:
        1. Upload a CSV file.\n
        2. Select whether you want a **Classification** or **Regression** tree.\n
        3. Pick your **Target (outcome)** variable.\n
        4. Select which columns will serve as **Predictors (features)**.\n
        5. Choose the **Scale of Measurement** for each predictor (categorical vs. numeric).\n
        6. Adjust any **hyperparameters** (e.g., criterion, max depth).\n
        7. Click **Train Model** to see the results:\n
           - Performance metrics\n
           - Tree visualization\n
           - Variable importance\n
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
        # Step 3: Choose the target variable
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
            default=possible_predictors  # You can leave default empty if you prefer
        )

        # -----------------------------------------------------------
        # Step 5: Choose scale of measurement for each predictor
        # -----------------------------------------------------------
        # We'll store the scale in a dictionary, e.g. {"var1": "categorical", "var2": "numeric", ...}
        scale_of_measurement = {}
        st.write("### Specify Scale of Measurement for Each Predictor")
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
        # Splitting criterion
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

        # Max depth
        max_depth = st.slider(
            "Max depth of the tree (0 = unlimited)",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
        max_depth = None if max_depth == 0 else max_depth

        # Min samples split
        min_samples_split = st.slider(
            "Minimum samples required to split an internal node",
            min_value=2,
            max_value=50,
            value=2,
            step=1
        )

        # -----------------------------------------------------------
        # Button to train model
        # -----------------------------------------------------------
        if st.button("Train Model"):
            st.subheader("Model Results")

            # -----------------------------------------------------------
            # Construct X (predictors) based on scale of measurement
            # -----------------------------------------------------------
            # We'll build a new DataFrame (X) that properly encodes
            # each selected predictor as numeric or categorical (dummies).
            X_list = []
            col_names = []

            for col in selected_predictors:
                if scale_of_measurement[col] == "categorical":
                    # One-hot encode the categorical column
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    X_list.append(dummies)
                    col_names.extend(dummies.columns.tolist())
                else:
                    # Numeric, use as-is
                    X_list.append(df[[col]])
                    col_names.append(col)

            # Concatenate all feature columns
            if X_list:
                X = pd.concat(X_list, axis=1)
            else:
                st.error("No predictors selected.")
                return

            # Define y (target)
            y = df[target_col]

            # -----------------------------------------------------------
            # Create train-test split
            # -----------------------------------------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # -----------------------------------------------------------
            # Build and fit the Decision Tree
            # -----------------------------------------------------------
            if task_type == "Classification":
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )
            else:  # Regression
                model = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )

            model.fit(X_train, y_train)

            # -----------------------------------------------------------
            # Evaluate the model
            # -----------------------------------------------------------
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy:** {accuracy:.4f}")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix:")
                st.write(cm)

                # Classification report (precision, recall, F1-score, etc.)
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write("Classification Report:")
                st.write(pd.DataFrame(report).transpose())

            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**R² (Coefficient of Determination):** {r2:.4f}")

            # -----------------------------------------------------------
            # Variable Importance
            # -----------------------------------------------------------
            st.write("### Variable Importance")
            importances = model.feature_importances_
            # Sort in descending order
            sorted_idx = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_idx]
            sorted_features = np.array(col_names)[sorted_idx]

            # Display as a table
            importance_df = pd.DataFrame({
                "Feature": sorted_features,
                "Importance": sorted_importances
            })
            st.write(importance_df)

            # Display as a bar chart
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(sorted_features)), sorted_importances)
            ax_imp.set_xticks(range(len(sorted_features)))
            ax_imp.set_xticklabels(sorted_features, rotation=45, ha="right")
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

            # -----------------------------------------------------------
            # Visualize the Decision Tree
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



