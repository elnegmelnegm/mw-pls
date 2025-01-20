def main():
    st.title("Moving Window PLS (MWPLS) vs. PLS")

    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your data in CSV format", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.header("Data Preview")
        st.sidebar.dataframe(data.head())

        # Data Splitting
        X_cols = st.sidebar.multiselect("Select features (X variables):", data.columns)
        y_col = st.sidebar.selectbox("Select target (y variable):", data.columns)

        if not X_cols or not y_col:
            st.error("Please select at least one feature and a target variable.")
            return

        X = data[X_cols].values
        y = data[y_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Parameters
        st.sidebar.header("Model Parameters")
        win_size = st.sidebar.slider("Window Size", min_value=2, max_value=X_train.shape[1], value=10)
        max_comps_mwpls = st.sidebar.slider("Max PLS Components (MWPLS)", min_value=1, max_value=min(20, win_size), value=5)
        cv_splits_mwpls = st.sidebar.number_input("Cross-validation Splits (MWPLS)", min_value=2, value=5)

        max_comps_pls = st.sidebar.slider("Max PLS Components (PLS)", min_value=1, max_value=X_train.shape[1], value=10)

        # Run Models
        if st.sidebar.button("Run Models"):
            # MWPLS
            st.header("Moving Window PLS (MWPLS)")
            best_win_pos, best_n_comp, min_rmse = mwpls(X_train, y_train, win_size, max_comps_mwpls, cv_splits_mwpls, plot=True)

            pls_mw = PLSRegression(n_components=best_n_comp)
            pls_mw.fit(X_train[:, best_win_pos:best_win_pos + win_size], y_train)

            y_train_pred_mw = pls_mw.predict(X_train[:, best_win_pos:best_win_pos + win_size])
            y_test_pred_mw = pls_mw.predict(X_test[:, best_win_pos:best_win_pos + win_size])

            # PLS
            st.header("Partial Least Squares (PLS)")
            pls = PLSRegression(n_components=max_comps_pls)
            pls.fit(X_train, y_train)

            y_train_pred_pls = pls.predict(X_train)
            y_test_pred_pls = pls.predict(X_test)

            # Evaluate and Display Metrics
            st.subheader("Model Evaluation")
            metrics = pd.DataFrame({
                'Metric': ['R2', 'MSE', 'MAE'],
                'MWPLS Train': [r2_score(y_train, y_train_pred_mw),
                                mean_squared_error(y_train, y_train_pred_mw),
                                mean_absolute_error(y_train, y_train_pred_mw)],
                'MWPLS Test': [r2_score(y_test, y_test_pred_mw),
                               mean_squared_error(y_test, y_test_pred_mw),
                               mean_absolute_error(y_test, y_test_pred_mw)],
                'PLS Train': [r2_score(y_train, y_train_pred_pls),
                              mean_squared_error(y_train, y_train_pred_pls),
                              mean_absolute_error(y_train, y_train_pred_pls)],
                'PLS Test': [r2_score(y_test, y_test_pred_pls),
                             mean_squared_error(y_test, y_test_pred_pls),
                             mean_absolute_error(y_test, y_test_pred_pls)]
            })
            st.table(metrics.set_index('Metric'))

if __name__ == "__main__":
    main()
