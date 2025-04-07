import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Streamlit App Title
# ------------------------------------------------
st.title("🎓 Easy Decision Tree Analyzer for Education Research")

# ------------------------------------------------
# File Upload
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read data
    df = pd.read_csv(uploaded_file)
    
    st.write("### 1. Preview of your data")
    st.write("This table shows the first few rows of your uploaded dataset.")
    st.dataframe(df.head())

    # ------------------------------------------------
    # Column Selections
    # ------------------------------------------------
    st.write("### 2. Select your target (outcome) variable and predictors")
    all_columns = df.columns.tolist()

    # Choose target and define type
    target = st.selectbox("🎯 Select your target (outcome) variable", all_columns)
    target_type = st.radio(
        "🔎 What type of variable is the target?",
        ["Categorical (e.g., pass/fail, yes/no)", "Numerical (continuous values)"]
    )

    # Choose predictor variables
    predictors = st.multiselect(
        "🧩 Select one or more predictor (explanatory) variables", 
        [col for col in all_columns if col != target]
    )

    # Define each predictor's type
    predictor_types = {}
    for col in predictors:
        predictor_types[col] = st.selectbox(
            f"📌 Data type for predictor `{col}`",
            ["Categorical", "Numerical"],
            key=col
        )

    # ------------------------------------------------
    # Once the user has chosen a target & predictors...
    # ------------------------------------------------
    if target and predictors:
        # Separate features (X) and target (y)
        X = df[predictors].copy()
        y = df[target].copy()

        # Convert predictor types
        for col, vtype in predictor_types.items():
            if vtype == "Categorical":
                X[col] = X[col].astype("category")

        # Decide if it's classification or regression
        if "Categorical" in target_type:
            y = y.astype("category")
            problem_type = "classification"
        else:
            y = pd.to_numeric(y, errors="coerce")
            problem_type = "regression"

        # ------------------------------------------------
        # Data Cleaning: drop rows with missing values
        # ------------------------------------------------
        data = pd.concat([X, y], axis=1).dropna()
        X = data[predictors]
        y = data[target]

        # Encode categorical predictors (turn categories into 0/1 dummy codes)
        X = pd.get_dummies(X, drop_first=True)

        st.write(f"**You have selected a {problem_type.upper()} problem** based on your target variable type.")

        # ------------------------------------------------
        # Train-Test Split
        # ------------------------------------------------
        st.write("### 3. Train-Test Split")
        st.write("We split the data into training (70%) and test (30%) to evaluate how well the model generalizes.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # ------------------------------------------------
        # Model Building and Fitting
        # ------------------------------------------------
        st.write("### 4. Train the Decision Tree Model")
        st.write("We will create a decision tree with a maximum depth of 4. This limit helps make the tree easier to interpret.")
        if problem_type == "classification":
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=4, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ------------------------------------------------
        # Model Performance
        # ------------------------------------------------
        st.write("### 5. Model Performance on Test Data")

        if problem_type == "classification":
            score = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy**: The model correctly classifies {score:.2f} (out of 1.00) of the test data.")
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"**Mean Squared Error**: {mse:.2f}")

        # ------------------------------------------------
        # Display Tree Rules
        # ------------------------------------------------
        st.write("### 6. Decision Tree Rules")
        st.write("These rules show how the decision tree splits the data step by step:")
        tree_rules = export_text(model, feature_names=X.columns.tolist())
        st.code(tree_rules)

        # ------------------------------------------------
        # Plot the Tree
        # ------------------------------------------------
        st.write("### 7. Decision Tree Diagram")
        st.write("Below is a diagram of the decision tree. Each node shows the splitting condition and the outcome.")
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(
            model,
            feature_names=X.columns,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)

        # ------------------------------------------------
        # Feature Importance
        # ------------------------------------------------
        st.write("### 8. Feature Importance")
        st.write(
            "This chart shows which predictors are most important in splitting the data. "
            "A higher value means the feature contributed more to the decision rules."
        )
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig2, ax2 = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax2)
        ax2.set_xlabel("Importance Score")
        ax2.set_ylabel("Feature Name")
        st.pyplot(fig2)

        # ------------------------------------------------
        # Interpretation
        # ------------------------------------------------
        st.write("### 9. Interpretation and Next Steps")
        st.write(
            "Use the results above to understand how each predictor influences the outcome. "
            "Here are some basic tips for interpretation:"
        )
        
        top_feat = importance_df.iloc[0]
        st.markdown(f"""
        - **Most important predictor**: `{top_feat.Feature}`.  
          This means this feature plays a central role in how the tree splits and makes decisions.
        - **Tree Rules**: Start from the top of the tree and follow the splits to see which conditions lead to different outcomes.
        - **Keep it Simple**: A maximum depth of 4 is used here to keep the tree easy to interpret. You can adjust the depth to see if performance improves, but the tree may become more complex.
        - **Next Steps**: Consider collecting more data, testing other model types (e.g., Random Forest), or applying cross-validation to see if your results hold up.
        """)

