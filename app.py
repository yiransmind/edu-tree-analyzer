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
    st.title("Decision Tree App for Educational Research")

    st.write(
        """
        **Instructions**:
        1. Upload a CSV file containing your dataset.\n
        2. Select if you want to do a classification or regression tree.\n
        3. Choose your target variable (the outcome you want to predict).\n
        4. Optionally adjust the decision tree parameters.\n
        5. Click "Train Model" to see results!\n
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
        target_col = st.selectbox("Select your target variable", all_columns)

        # For convenience, define feature columns as everything except the target
        feature_cols = [col for col in all_columns if col != target_col]

        # -----------------------------------------------------------
        # Step 4: Select model hyperparameters
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
            "Max depth of the tree (0 is unlimited)",
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

        # Button to train model
        if st.button("Train Model"):
            st.subheader("Model Results")

            # -----------------------------------------------------------
            # Separate data into features (X) and target (y)
            # -----------------------------------------------------------
            X = df[feature_cols]
            y = df[target_col]

            # Handle potential non-numeric features by one-hot encoding
            # (in a real-world scenario, you may want more robust preprocessing)
            X = pd.get_dummies(X, drop_first=True)

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

                # Classification report (precision, recall, etc.)
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
            # Visualize the Decision Tree
            # -----------------------------------------------------------
            # We’ll generate a simple matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(
                model,
                feature_names=X_train.columns,
                class_names=(
                    [str(cls) for cls in model.classes_] 
                    if task_type == "Classification" 
                    else None
                ),
                filled=True,
                rounded=True,
                ax=ax
            )
            st.pyplot(fig)

if __name__ == "__main__":
    main()



