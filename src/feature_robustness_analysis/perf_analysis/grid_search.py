import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import pipeline
from mri_radiomics_toolkit.feature_selection import preliminary_feature_filtering
from mri_radiomics_toolkit.model_building import cv_grid_search, neg_log_loss
from mri_radiomics_toolkit.models.cards import multi_class_cv_grid_search_card
from typing import Union, Optional, Any, Tuple


def get_best_hyperparameters(X: pd.DataFrame,
                             gt: Union[pd.DataFrame, pd.Series]) -> Tuple[dict, dict, pd.Index]:
    r"""Get the best hyperparameters for a given dataset and ground truth labels using cross-validated grid
    search.

    The function first standardizes the input data using RobustScaler, then applies preliminary feature
    filtering. After that, it performs cross-validated grid search to find the best hyperparameters for the
    model, and return the instance of the model created using these best hyperparameters.

    Args:
        X (pd.DataFrame):
            The input data, where each row is a sample and each column is a feature.
        gt (Union[pd.DataFrame, pd.Series]):
            The ground truth labels for each sample.

    Returns:
        Tuple[dict, dict, pd.Index]:
            A tuple containing three elements:
                - best_params: A dictionary of the best hyperparameters found by the grid search.
                - best_estimators: A dictionary of the best models for each class, with classes as keys.
                - X_train.index: The index of the samples in the training set after feature filtering.

    Raises:
        ValueError:
            If the input parameters are not in the expected format or if the grid search does not converge.
    """
    # Zscore normalization
    zscored = StandardScaler().fit_transform(X.T).T
    X = pd.DataFrame(data=zscored, index=X.index, columns=X.columns)

    # Use custom pipeline that changes the standardization
    clf = pipeline.Pipeline([
        ('standardization', StandardScaler()),
        ('classification', 'passthrough')
    ])

    # Feature selection
    X_train, _ = preliminary_feature_filtering(X, None, gt.to_frame(), p_thres=0.01)

    # Perform CV grid search for best hyperparams
    best_params, results, predict_table, best_estimators = cv_grid_search(
        X_train.T, gt,
        param_grid_dict=multi_class_cv_grid_search_card,
        scoring=neg_log_loss,
        clf=clf
    )
    return best_params, best_estimators, X_train.index

