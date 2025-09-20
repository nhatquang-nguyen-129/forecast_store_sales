class RecursiveFeatureEliminator(BaseEstimator, TransformerMixin):
    '''
    A transformer class for selecting features based on the Recursive Feature Elimination (RFE) technique.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by recursively selecting the features with highest feature 
        importances until a final desired number of features is obtained through time series rolling window
        cross validation.
    '''

    def __init__(self, 
                 estimator=LGBMRegressor(verbosity=-1), 
                 scoring='neg_mean_squared_error', 
                 n_folds=3,
                 test_size=1*93*50*10, 
                 gap=1*7*50*10):
        '''
        Initialize the Recursive Feature Elimination (RFE) transformer.
        
        Args:
            estimator (object, default=LGBMRegressor): The model to obtain feature importances.
            scoring (object, default='neg_mean_squared_error'): The scoring for time series rolling window cross-validation.
            n_folds (int, default=5): The number of folds for time series rolling window cross validation.
            test_size (int, default=1*93*50*10): The size of the test for time series rolling window cross validation.
            gap (int, default=1*7*50*10): The gap between training and test for time series rolling window cross validation.
            
        '''
        # Get sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_folds, 
                               test_size=test_size, 
                               gap=gap)
        
        self.rfe = RFECV(estimator=estimator, 
                         cv=tscv,
                         scoring=scoring)

    def fit(self, X, y):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns an instance of self.
        '''
        # Save the date indexes.
        date_idx = X.index
        
        self.rfe.fit(X, y)
        
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by recursively selecting the features with highest feature 
        importances.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after recursively selecting the features with highest feature 
            importances.
        '''
        # Recursively select the features with highest feature importances.
        X_selected = self.rfe.transform(X)

        # Create a dataframe for the final selected features.
        selected_df = pd.DataFrame(X_selected,
                                  columns=self.rfe.get_feature_names_out(),
                                  )

        return selected_df


def plot_sales_forecast_items_stores(y_true, y_pred, data):
    '''
    Plots the sales forecast for each item per store.

    Args:
        y_true (array-like): True sales values.
        y_pred (array-like): Predicted sales values.
        Data (DataFrame): DataFrame containing the features over time.

    Returns:
        None
        
    Raises:
        CustomException: An error occurred during the plotting process
    '''
    try:
        actual_pred_data = data.copy()
        actual_pred_data['actual'] = np.expm1(y_true)
        actual_pred_data['pred'] = np.expm1(y_pred)

        fig, axes = plt.subplots(10, 5, figsize=(50, 50))

        # Lists to store legend handles and labels
        legend_handles = []
        legend_labels = []

        for i in range(1, 51):
            item_i_actual_pred = actual_pred_data.loc[actual_pred_data['item'] == i]

            # Determine subplot indices
            row_index = (i - 1) // 5
            col_index = (i - 1) % 5

            # Plot on the appropriate subplot
            ax = axes[row_index, col_index]

            # Iterate over each store and plot predicted sales
            for store in item_i_actual_pred['store'].unique():
                store_data = item_i_actual_pred[item_i_actual_pred['store'] == store]
                line = sns.lineplot(data=store_data, x=store_data.index, y='pred', label=f'Store {store:.0f}', ax=ax)
                if i == 1:  # Only need to collect handles and labels once
                    legend_handles.append(line.lines[0])
                    legend_labels.append(f'Store {store:.0f}')

            # Set labels and title
            ax.set_xlabel('Date', loc='left', labelpad=25)
            ax.set_ylabel('Sales', loc='top', labelpad=25)
            ax.set_title(f'Item {i} sales forecast per store', fontweight='bold', fontsize=25, pad=25)
            ax.grid(True)
            ax.legend().remove()  # Remove legend from each subplot

        # Create a single legend outside the loop
        leg = fig.legend(handles=legend_handles, labels=legend_labels, loc='upper left')
        for i in range(0, 10):
            leg.legendHandles[i].set_color(ts_palette[i])

        # Adjust layout
        plt.tight_layout()

        # Show or save the plot
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)