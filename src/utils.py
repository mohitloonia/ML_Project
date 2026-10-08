import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train, y_train, X_test, y_test, models, param
):
    try:
        # Imported here to avoid circular imports.
        from src.components.data_transformation import DataTransformation

        report = {}

        cv = KFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        for name, model in list(models.items()):
            print(f"Training {name}...", flush=True)

            preprocessor = (
                DataTransformation().get_data_transformer_object()
            )

            preprocessor.set_params(
                cat_pipelines__one_hot_encoder__handle_unknown="ignore"
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            parameter_grid = {
                f"model__{key}": values
                for key, values in param[name].items()
            }

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                cv=cv,
                scoring="r2",
                refit=True,
                error_score="raise"
            )

            search.fit(X_train, y_train)

            models[name] = search.best_estimator_
            report[name] = float(search.best_score_)

            print(
                f"{name}: CV R² = {report[name]:.4f}",
                flush=True
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)