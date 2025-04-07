import streamlit as st
import pandas as pd
import numpy as np

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

###################################
# 1. Define Helper Functions
###################################

def freq_percent(column: pd.Series):
    """
    Returns a DataFrame with frequency counts and percentages
    for a categorical column.
    """
    freq = column.value_counts(dropna=False)
    percent = freq / freq.sum() * 100
    return pd.DataFrame({
        'Frequency': freq,
        'Percentage': percent.round(2)
    })

def model_evaluate(model, X, y):
    """
    Returns evaluation metrics (R2, MSE, RMSE, MAE) on full data.
    """
    preds = model.predict(X)
    r2_val = r2_score(y, preds)
    mse_val = mean_squared_error(y, preds)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y, preds)
    
    return {
        'R2': r2_val,
        'MSE': mse_val,
        'RMSE': rmse_val,
        'MAE': mae_val
    }

def cross_validate_tree(X, y):
    """
    Performs 10-fold cross-validation with a DecisionTreeRegressor
    to get mean RMSE and mean R2 across folds.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # For MSE/RMSE
    mse_scores = -cross_val_score(
        DecisionTreeRegressor(random_state=42),
        X, y,
        scoring='neg_mean_squared_error',
        cv=kf
    )
    rmse_scores = np.sqrt(mse_scores)
    
    # For R2
    r2_scores = cross_val_score(
        DecisionTreeRegressor(random_state=42),
        X, y,
        scoring='r2',
        cv=kf
    )
    
    return {
        'CV_RMSE': np.mean(rmse_scores),
        'CV_R2': np.mean(r2_scores)
    }


###################################
# 2. Streamlit App
###################################

st.title("Decision Tree Analysis (Similar to English2.R)")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("## Data Preview")
    st.dataframe(data.head())

    ############################
    # 2a. Frequency Stats
    ############################
    # Provide a multiselect for the user to pick which categorical columns
    st.write("### Frequency Tables for Selected Categorical Variables")
    all_cols = data.columns.tolist()
    cat_cols = st.multiselect(
        "Select categorical columns to view Frequency & Percentage:",
        options=all_cols
    )

    for col in cat_cols:
        st.write(f"**Column:** {col}")
        st.table(freq_percent(data[col]))

    ############################
    # 2b. Create Composite Variables
    ############################
    # Example structure, adapt to your own column names
    st.write("### Creating Composite Scores")
    st.write(
        "Below, we assume you have sub-items like `CSR1, CSR2, CSR3` "
        "that combine into a `CSR` composite score, etc. Adapt these lines to fit your dataset."
    )
    
    # Check if user’s dataset has these columns; otherwise, skip
    needed_cols = [
        ["CSR1", "CSR2", "CSR3"],
        ["MSR1", "MSR2", "MSR3"],
        ["MCSR1", "MCSR2", "MCSR3"],
        ["SSR1", "SSR2", "SSR3"],
        ["BSR1", "BSR2", "BSR3"],
        ["SE1", "SE2", "SE3"],
        ["GR1", "GR2", "GR3"],
        ["GM1", "GM2", "GM3"],
        ["IM1", "IM2", "IM3"],
        ["EM1", "EM2", "EM3"],
        ["AT1", "AT2", "AT3", "AT4"]
    ]

    # Create composites only if columns exist
    for group in needed_cols:
        missing = [col for col in group if col not in data.columns]
        if missing:
            st.warning(f"Skipping {group}, missing: {missing}")
        else:
            base_name = group[0][:-1]  # e.g., CSR from CSR1
            # If the first item ends with a digit, remove it:
            # e.g., 'CSR1' -> 'CSR', 'BSR1' -> 'BSR', ...
            composite_name = base_name
            data[composite_name] = data[group].mean(axis=1, skipna=True)
            st.write(f"Created composite: **{composite_name}** from {group}")

    # Optional: show all newly created columns
    st.write("**Current Columns in Data**:", data.columns.tolist())

    ############################
    # 2c. Summaries of Composites
    ############################
    st.write("### Descriptive Statistics for Composite Variables")
    # Identify all composites by checking if they are exactly 3 or 4 letter columns (e.g., CSR, MSR, MCSR?)
    # Or you can define them directly:
    composites = ["CSR", "MSR", "MCSR", "SSR", "BSR", "SE", "GR", "GM", "IM", "EM", "AT"]
    existing_composites = [c for c in composites if c in data.columns]
    
    if existing_composites:
        st.dataframe(data[existing_composites].describe())
    else:
        st.write("No valid composite columns found.")

    ############################
    # 3. Fit Multiple Decision Trees
    ############################
    st.write("## Fitting Multiple Decision Trees")

    # Pick your predictor set (here, we mimic the R code: SE, GR, GM, IM, EM, AT)
    # but only if they are in the dataset
    possible_predictors = ["SE", "GR", "GM", "IM", "EM", "AT"]
    predictors = [p for p in possible_predictors if p in data.columns]
    
    if not predictors:
        st.error("No valid predictors found (like SE, GR, GM, IM, EM, AT). Please adapt the code to your dataset.")
    else:
        # List of target variables (CSR, MSR, MCSR, SSR, BSR) if they exist
        target_vars = ["CSR", "MSR", "MCSR", "SSR", "BSR"]
        existing_targets = [t for t in target_vars if t in data.columns]

        results = []
        model_dict = {}

        for target in existing_targets:
            # Drop rows with missing predictor/target data
            sub_df = data[predictors + [target]].dropna()
            X = sub_df[predictors]
            y = sub_df[target]

            # Fit a decision tree regressor
            tree_model = DecisionTreeRegressor(random_state=42)
            tree_model.fit(X, y)

            # Evaluate on full data
            full_eval = model_evaluate(tree_model, X, y)

            # Cross-validation evaluation
            cv_res = cross_validate_tree(X, y)

            results.append({
                "Model": target,
                "R2": round(full_eval["R2"], 4),
                "MSE": round(full_eval["MSE"], 4),
                "RMSE": round(full_eval["RMSE"], 4),
                "MAE": round(full_eval["MAE"], 4),
                "CV_RMSE": round(cv_res["CV_RMSE"], 4),
                "CV_R2": round(cv_res["CV_R2"], 4)
            })

            # Store model for later reporting & plotting
            model_dict[target] = tree_model
        
        if results:
            st.write("### Model Performance Table")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # ------------------------------------------------
            # 3a. Summaries & Plots of Each Tree
            # ------------------------------------------------
            st.write("### Decision Tree Summaries & Plots")

            for target, model in model_dict.items():
                st.subheader(f"Decision Tree for {target}")

                # Text-based summary (like 'summary(model)' in R)
                # We'll limit depth for readability
                tree_rules = export_text(model, feature_names=predictors, max_depth=3)
                st.text(tree_rules)

                # Plot the tree
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_tree(
                    model,
                    feature_names=predictors,
                    filled=True,
                    max_depth=3
                )
                st.pyplot(fig)

                # Variable importance
                st.write("**Variable Importance**")
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Predictor": predictors,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)
                st.table(importance_df.reset_index(drop=True))

        else:
            st.warning("No target columns found to build decision trees.")

