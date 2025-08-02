class modelos():
    """
    A class to manage and evaluate different machine learning models.
    
    Attributes:
        models (dict): Dictionary to store model names and their evaluation metrics (mse, training time).
    """

    def __init__(self, x_test=None, y_test=None, x_train=None, y_train=None):
        self.models = {'modelo':[], 'mse_train':[],'mse_test':[],'tiempo_entrenamiento':[]}
        self.x_test = x_test
        self.y_test = y_test    
        self.x_train = x_train
        self.y_train = y_train
    
    def add_model(self):
        model_xgb = xgb.XGBRegressor(random_state=12345)
        time_start_xgb = time.time()
        param_grid_gxb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        cv_xgb = GridSearchCV(model_xgb, param_grid_gxb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_xgb.fit(self.x_train, self.y_train)
        best_model_xgb = cv_xgb.best_estimator_ 
        mse_xgb_train = mean_squared_error(self.y_train, best_model_xgb.predict(self.x_train))
        predictions_xgb = best_model_xgb.predict(self.x_test)  
        mse_xgb_test = mean_squared_error(self.y_test, predictions_xgb) 
        time_end_xgb = time.time() - time_start_xgb
        self.models['modelo'].append('XGBoost')
        self.models['mse_train'].append(mse_xgb_train)      
        self.models['mse_test'].append(mse_xgb_test)
        self.models['tiempo_entrenamiento'].append(time_end_xgb)

        model_lgb = lgb.LGBMRegressor(random_state=12345, force_col_wise=True)
        time_start_lgb = time.time()    
        param_grid_lgb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        cv_lgb = GridSearchCV(model_lgb, param_grid_lgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_lgb.fit(self.x_train, self.y_train)  
        best_model_lgb = cv_lgb.best_estimator_
        mse_lgb_train = mean_squared_error(self.y_train, best_model_lgb.predict(self.x_train))
        predictions_lgb = best_model_lgb.predict(self.x_test)
        mse_lgb_test = mean_squared_error(self.y_test, predictions_lgb)
        time_end_lgb = time.time() - time_start_lgb
        self.models['modelo'].append('LightGBM')    
        self.models['mse_train'].append(mse_lgb_train)
        self.models['mse_test'].append(mse_lgb_test)
        self.models['tiempo_entrenamiento'].append(time_end_lgb)

        model_cb = cb.CatBoostRegressor(random_state=12345, verbose=0)
        time_start_cb = time.time()
        param_grid_cb = {
            'n_estimators': [100, 200],
            'depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }   
        cv_cb = GridSearchCV(model_cb, param_grid_cb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_cb.fit(self.x_train, self.y_train)
        best_model_cb = cv_cb.best_estimator_
        mse_cb_train = mean_squared_error(self.y_train, best_model_cb.predict(self.x_train))
        predictions_cb = best_model_cb.predict(self.x_test)
        mse_cb_test = mean_squared_error(self.y_test, predictions_cb)
        time_end_cb = time.time() - time_start_cb
        self.models['modelo'].append('CatBoost')
        self.models['mse_train'].append(mse_cb_train)
        self.models['mse_test'].append(mse_cb_test)
        self.models['tiempo_entrenamiento'].append(time_end_cb)

        model_rf = RandomForestRegressor(random_state=12345)
        time_start_rf = time.time()
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [2, 5]
        }
        cv_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rf.fit(self.x_train, self.y_train)   
        best_model_rf = cv_rf.best_estimator_
        mse_rf_train = mean_squared_error(self.y_train, best_model_rf.predict(self.x_train))
        predictions_rf = best_model_rf.predict(self.x_test)
        mse_rf_test = mean_squared_error(self.y_test, predictions_rf)
        time_end_rf = time.time() - time_start_rf
        self.models['modelo'].append('RandomForest')
        self.models['mse_train'].append(mse_rf_train)
        self.models['mse_test'].append(mse_rf_test)
        self.models['tiempo_entrenamiento'].append(time_end_rf)
        mode = pd.DataFrame(self.models)
        return mode, best_model_xgb, best_model_lgb, best_model_cb, best_model_rf
    




        
    