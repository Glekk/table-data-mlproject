import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param_grid):
    try:
        results = []
        for name, model in models.items():
            params = param_grid[name]

            random_search = RandomizedSearchCV(
                model,
                params,
                n_iter=20,
                scoring='f1',
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X_train, y_train)

            model.set_params(**random_search.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            results.append([name,
                            f1_score(y_train, y_train_pred),
                            f1_score(y_test, y_test_pred)
                            ])
            logging.info(f'Ended training {name}')
        return results
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)