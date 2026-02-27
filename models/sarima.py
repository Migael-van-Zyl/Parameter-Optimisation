import logging
import numpy as np
import pandas as pd
import itertools

from statsforecast import StatsForecast 
from statsforecast.models import AutoARIMA 
from utilsforecast.losses import mae, mse, mape 
from utilsforecast.evaluation import evaluate

from abc import ABC, abstractmethod

class AModel(ABC):
    """
    This abstract class serves as a blueprint that will encapsulate all the functionality that any type of model will have e.g train, predict etc.
    
    Key considerations:
        - Any subclass that inherits this class MUST implement the methods defined in this class.
        - This class CANNOT be instantiated to create an object.
    """
    @abstractmethod
    def train(self, train_df: pd.DataFrame):
        pass
    
    @abstractmethod
    def predict(self, model, test_df: pd.DataFrame):
        pass
    
    def evaluate(self, test_df, y_pred, metrics=None):
        try:
            print('Evaluation in progress...')
            if metrics is None:
                metrics=[mae, mse, mape]
            
            y_pred['ds'] = y_pred['ds'].apply(lambda x: x.replace(day=1)) 
                
            eval_df = pd.merge(test_df, y_pred, how='inner', on=['ds', 'unique_id'])
            evaluation = evaluate(
                eval_df,
                metrics=metrics
            )
            
            print('Evaluation complete...')
            logging.info('Evaluation complete...')
            return evaluation
        except Exception as e:
            print('Error in evaluate: {}'.format(e))
            logging.error('Error in evaluate: {}'.format(e))
            return None
        
    def hyperparameter_tuning(self, train_df, model_name:str, param_grid=None):
        print('Starting hyperparameter tuning...')
        try:
            results = []
            
            if model_name.lower()=='arima' and param_grid is None:
                param_grid = {
                    'max_p': [2,3],
                    'max_d': [0,2],
                    'max_q': [2,3]
                }
                
                for max_p, max_d, max_q in itertools.product(
                        param_grid['max_p'],
                        param_grid['max_d'],
                        param_grid['max_q'],
                ):
                  model = AutoARIMA(
                      season_length=12,
                      max_p = max_p,
                      max_d = max_d,
                      max_q = max_q,
                      seasonal=False,
                      alias=model_name
                  )
                  sf = StatsForecast([model], freq='MS')
                  cv = sf.cross_validation(df=train_df, h=12, n_windows=3)
                  
                  rsme = np.sqrt(((cv['y'] - cv[model_name])**2).mean())
                  results.append((max_p, max_d, max_q, rsme))
                  
                  print(f'{(max_p, max_d, max_q, rsme)}')
            elif model_name=='sarima':
                param_grid = {
                    'max_p': [2,3],
                    'max_d': [0,2],
                    'max_q': [2,3],
                    'max_P': [0,1],
                    'max_D': [0,1],
                    'max_Q': [0,1]
                }
                
                for max_p, max_d, max_q, max_P, max_D, max_Q in itertools.product(
                        param_grid['max_p'],
                        param_grid['max_d'],
                        param_grid['max_q'],
                        param_grid['max_P'],
                        param_grid['max_D'],
                        param_grid['max_Q']
                ):
                  model = AutoARIMA(
                      max_p = max_p,
                      max_d = max_d,
                      max_q = max_q,
                      max_P=max_P,
                      max_D=max_D,
                      max_Q=max_Q,
                      m=12,
                      seasonal=True,
                      alias=model_name
                  )
                  sf = StatsForecast([model], freq='MS')
                  cv = sf.cross_validation(df=train_df, h=12, n_windows=3)
                  
                  rsme = np.sqrt(((cv['y'] - cv[model_name])**2).mean())
                  results.append((max_p, max_d, max_q, rsme))
                  
                  print(f'{(max_p, max_d, max_q, rsme)}')
            else:
                print('Error model does not exist')
                exit
            
            return results
              
        except Exception as e:
            print('Error in hyperparameter_tuning: {}'.format(e))
            return None

class Model(AModel):
    
    def train(self, train_df):
        try:
            print('Model training started...')
            sf = StatsForecast(
            models = [
                AutoARIMA(season_length=12, seasonal=False, alias='ARIMA'),
                AutoARIMA(season_length=12, seasonal=True, alias='SARIMA'),
            ],
            freq='MS',
            n_jobs = 1
            )
            
            sf.fit(df=train_df)
            
            print('Model training complete...')
            logging.info('Model training complete...')
            return sf
        except Exception as e:
            print('Error in train: {}'.format(e))
            logging.error('Error in train: {}'.format(e))
            return None
    
    def predict(self, model, forecast_horizon):
        try:
            print('Model predictions started...')
            y_pred = model.predict(h=forecast_horizon)
            
            print('Model predictions complete...')
            logging.info('Model predictions complete...')
            return y_pred    
        except Exception as e:
            print('Error in Model predict: {}'.format(e))
            logging.info('Error in Model predict: {}'.format(e))
            return None

class SARIMA(AModel):
    def train(self, train_df):
        try:
            print('Training SARIMA model...')
            sf = StatsForecast(
                models = [
                    AutoARIMA(season_length=12, seasonal=True, alias='SARIMA')
                ],
                freq='MS'
            )
            
            sf.fit(df=train_df)
            
            print('SARIMA training complete...')
            logging.info('SARIMA training complete...')
            return sf
        except Exception as e:
            print('Error in SARIMA train: {}'.format(e))
            logging.info('Error in SARIMA train: {}'.format(e))
            return None
    
    def predict(self, model, forecast_horizon) -> np.array:
        try:
            print('SARIMA predictions started...')
            y_pred = model.predict(h=forecast_horizon)
            
            print('SARIMA predictions completed...')
            logging.info('SARIMA predictions complete...')
            return y_pred 
        except Exception as e:
            print('Error in SARIMA predict: {}'.format(e))
            logging.error('Error in SARIMA predict: {}'.format(e))
            return None