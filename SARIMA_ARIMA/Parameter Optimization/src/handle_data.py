import logging 
import pandas as pd
from typing import Union
    
class DataPreprocessing:
    def load_data(self, file_path, sheet_name=None):
        try:
            print('Data loading started...')
            if sheet_name is None:
                df = pd.read_excel(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print('Data loading complete...')
            logging.info('Data loading complete...')
            
            return df
        except Exception as e:
            print('Error in load_data: {}'.format(e))
            logging.error('Error in load_data: {}'.format(e))
    
    def process_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            print('Data preprocessing started...')            
            df_copy = df.copy()
            
            # REMOVE UNNECESSARY COLUMNS
            df_copy = df_copy.drop(columns=['CustomerID', 
                                            'PodID', 
                                            'CustomerName', 
                                            'OperatingUnit',
                                            'OffPeakConsumption',
                                            'PeakConsumption',
                                            'StandardConsumption',
                                            'Category',
                                            'SubSIC',
                                            ], axis=1)
            
            # FILL MISSING VALUES
            df_copy['TotalConsumption'].fillna(0, inplace=True)
            
            # CONVERT THE CONSUMPTION FROM 'kWh' to 'GWh'
            df_copy['TotalConsumption'] = df_copy['TotalConsumption'] / 1000000
            
            # RENAME THE 'ReportingMonth' COLUMN TO 'ds'
            df_copy.rename(
                columns=
                {
                    'ReportingMonth':'ds',
                    'SICID':'unique_id',
                    'TotalConsumption':'y'
                }, 
            inplace=True)
            
            # CHANGE THE DATA TYPE FOR 'ReportingMonth column TO DATETIME
            df_copy['ds'] = pd.to_datetime(df_copy['ds'])
            
            # AGGREGATE THE DATA
            df_copy = df_copy.groupby(['ds', 'unique_id'], as_index=False)[['y']].sum()
            
            # SORT THE DATA ACCORDING TO THE unique_id and ds COLUMNS
            df_copy = df_copy.sort_values(['unique_id', 'ds']).reset_index(drop=True)
            
            df_copy['unique_id'] = df_copy['unique_id'].astype(str)
            
            print('Data preprocessing complete...')
            logging.info('Data preprocessing complete...')
            return df_copy
        except Exception as e:
            print('Error in process_data: {}'.format(e))
            logging.error('Error in process_data: {}'.format(e))
            
    def data_partition(self, 
                        df: pd.DataFrame, 
                        train_start_date: pd.DataFrame, 
                        test_start_date: pd.DataFrame
                        ):
        try:
            print('Data partition started...')
            train_data = df.loc[(df['ds'] >= train_start_date) & (df['ds'] < test_start_date)]
            test_data = df.loc[(df['ds'] >= test_start_date)]

            print('Data partition complete...')
            logging.info('Data Partitioning Complete...')
            return train_data, test_data
        except Exception as e:
            print('Error in data_partition: {}'.format(e))
            logging.error('Error in data_partition: {}'.format(e))
