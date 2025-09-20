# 1.1 Estimate financial results based on predicted and true values 
def estimate_financial_results(data, y_true, y_pred, per_store=False, per_store_item=False):
    '''
    Args:
        data (DataFrame): The input data containing date, store, and item information.
        y_true (Series): The true sales values.
        y_pred (Series): The predicted sales values.
        per_store (bool, optional): If True, calculate results per store. Defaults to False.
        per_store_item (bool, optional): If True, calculate results per store and item. Defaults to False.

    Returns:
        DataFrame: Estimated financial results.

    Raises:
        CustomException: If an error occurs during the calculation.

    Note:
        This function estimates financial results based on predicted and true sales values. 
        It calculates various scenarios such as total predicted sales, average predicted sales per day,
        daily Mean Absolute Error (MAE), worst and best average sales scenarios, worst and best total 
        sales scenarios, etc. If no specific result is determined, it just estimates the overall total financial result.
    '''
    try:
        financial_results = data.copy().reset_index()[['date', 'store', 'item']]
        financial_results['sales'] = np.expm1(y_true.reset_index(drop=True))
        financial_results['predictions'] = np.expm1(y_pred) 
        
        if per_store:
            # Placeholder for storing results.
            results_stores = []

            # Iterate over each store.
            for store_id in range(1, 11):
                # Obtain data for store i.
                store_i_data = financial_results.loc[financial_results['store'] == store_id]
                
                # Total predictions for the store.
                store_i_total_predictions = store_i_data['predictions'].sum()
                
                # Average predictions per day for the store.
                store_i_avg_predictions = store_i_total_predictions / 93
                
                # MAE per day.
                daily_pred_sales = store_i_data.groupby(['date'])[['sales', 'predictions']].sum().reset_index()
                daily_mae = mean_absolute_error(daily_pred_sales['sales'], daily_pred_sales['predictions'])
                
                # Worst and best daily scenario for average prediction.
                store_i_worst_scenario_avg = store_i_avg_predictions - daily_mae
                store_i_best_scenario_avg = store_i_avg_predictions + daily_mae
                
                # Worst and best total scenarios.
                store_i_worst_scenario_total = store_i_total_predictions - (daily_mae * 93)
                store_i_best_scenario_total = store_i_total_predictions + (daily_mae * 93)
                
                # Append results to the list.
                results_stores.append({
                    'Store': store_id,
                    'Total predicted sales': store_i_total_predictions,
                    'Average predicted sales (daily)': store_i_avg_predictions,
                    'Daily MAE': daily_mae,
                    'Worst average sales scenario (daily)': store_i_worst_scenario_avg,
                    'Best average sales scenario (daily)': store_i_best_scenario_avg,
                    'Worst total sales scenario': store_i_worst_scenario_total,
                    'Best total sales scenario': store_i_best_scenario_total
                })

            # Create DataFrame from results.
            stores_results = round(pd.DataFrame(results_stores))
            stores_results
            
            return stores_results
        
        elif per_store_item:
            sum_pred_items = financial_results.groupby(['store', 'item'])['predictions'].sum().reset_index()
            avg_pred_items = financial_results.groupby(['store', 'item'])['predictions'].mean().reset_index()
            mae_items = financial_results.groupby(['store', 'item']).apply(lambda x: mean_absolute_error(x['sales'], x['predictions'])).reset_index().rename(columns={0: 'MAE'})
            sum_avg = pd.merge(sum_pred_items, avg_pred_items, how='inner', on=['store', 'item']).rename(columns={'predictions_x': 'Total predicted sales', 'predictions_y': 'Average predicted sales (daily)'})
            items_results = pd.merge(sum_avg, mae_items, how='inner', on=['store', 'item'])
            items_results['Worst average sales scenario (daily)'] = items_results['Average predicted sales (daily)'] - items_results['MAE']
            items_results['Best average sales scenario (daily)'] = items_results['Average predicted sales (daily)'] + items_results['MAE']
            items_results['Worst total sales scenario'] = items_results['Total predicted sales'] - items_results['MAE'] * 93
            items_results['Best total sales scenario'] = items_results['Total predicted sales'] + items_results['MAE'] * 93
            items_results = items_results.rename(columns={'store': 'Store', 'item': 'Item'})
            items_results = round(items_results)
            
            return items_results
        
        
        # Total predicted sales and average predicted sales (daily).
        sum_pred = financial_results['predictions'].sum()
        avg_pred = sum_pred / 93

        # MAE per day.
        daily_pred_sales = financial_results.groupby(['date'])[['sales', 'predictions']].sum().reset_index()
        daily_mae = mean_absolute_error(daily_pred_sales['sales'], daily_pred_sales['predictions'])
            
        # Worst and best daily scenario for average prediction.
        worst_scenario_avg = avg_pred - daily_mae
        best_scenario_avg = avg_pred + daily_mae
            
        # Worst and best total scenarios.
        worst_scenario_total = sum_pred - (daily_mae * 93)
        best_scenario_total = sum_pred + (daily_mae * 93)

        overall_results_df = pd.DataFrame({
            'Overall total predicted sales': [sum_pred],
            'Overall average predicted sales (daily)': [avg_pred],
            'Overall daily MAE': [daily_mae],
            'Overall worst average sales scenario (daily)': [worst_scenario_avg],
            'Overall best average sales scenario (daily)': [best_scenario_avg],
            'Overall worst total sales scenario': [worst_scenario_total],
            'Overall best total sales scenario': [best_scenario_total]
        })

        overall_results_df = round(overall_results_df)
        
        return overall_results_df

    except Exception as e:
        raise CustomException(e, sys)