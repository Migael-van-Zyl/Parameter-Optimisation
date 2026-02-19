from model_dev import Model
from handle_data import DataPreprocessing

from pandas.tseries.offsets import DateOffset

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print('Starting training_pipeline...')
    dp = DataPreprocessing()
    
    # Load the data
    df = dp.load_data(r'..\data\customer_data.xlsx', sheet_name='History_5y')
    # print(df.head()) # UNCOMMENT FOR DEBUGGING
    
    # Perform exploratory data analysis
    processed_df = dp.process_data(df)
    # print(processed_df.head()) # UNCOMMENT FOR DEBUGGING
    
    # Define the start dates for training and testing datasets
    train_start_date = processed_df['ds'].min()
    test_start_date = processed_df['ds'].max() - DateOffset(years=2)
    
    # Split the data into training and test datasets
    train_df, test_df = dp.data_partition(
        df=processed_df,
        train_start_date=train_start_date, 
        test_start_date=test_start_date)
    # print(train_data.shape) # UNCOMMENT TO DEBUG
    # print(test_data.shape) # UNCOMMENT WHEN DEBUG
    
    arima_params = Model().hyperparameter_tuning(train_df=train_df.loc[(train_df['unique_id'] != '03000') & (train_df['unique_id'] != 'PRPD')], model_name='arima')    
    
    # Fit the models
    model = Model().train(train_df=train_df) # train the models
    model_preds = Model().predict(model=model, forecast_horizon=24) # Predict the next 24 months 
    model_eval = Model().evaluate(test_df=test_df, y_pred=model_preds) # evaluate the model
    
    # Save the dataset
    final_df = model_preds.merge(test_df, how='inner', on=['ds', 'unique_id'])
    final_df.to_csv(r'..\data\model_results.csv')
    
    print('Ending training_pipeline...')