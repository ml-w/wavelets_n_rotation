import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import pipeline
from mri_radiomics_toolkit.feature_selection import supervised_features_selection, preliminary_feature_filtering
from mri_radiomics_toolkit.model_building import cv_grid_search, neg_log_loss
from mri_radiomics_toolkit.models.cards import multi_class_cv_grid_search_card
from mri_radiomics_toolkit.perf_metric import top_k_accuracy_score_
from typing import Union, Tuple, Optional, Any

def normalize_n_feature_selection(X: pd.DataFrame,
                                  y: Union[pd.Series, pd.DataFrame],
                                  X_hold_out: pd.DataFrame,
                                  y_hold_out: Union[pd.Series, pd.DataFrame],
                                  **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    r"""Perform normalization and supervised feature selection on the given datasets.

    This function first normalizes the given feature sets, `X` and `X_hold_out`, using the StandardScaler.
    It then performs a supervised feature selection using the ElasticNet model, selecting features
    whose coefficients are non-zero. The selected features are limited to a reasonable number.

    Note: This function assumes the availability of a function named `supervised_features_selection`.

    Args:
        X (pd.DataFrame):
            The training feature set to be normalized and for feature selection.
        y (Union[pd.Series, pd.DataFrame]):
            The target values corresponding to `X`.
        X_hold_out (pd.DataFrame):
            The hold-out feature set to be normalized according to `X`'s normalization and for feature selection.
        y_hold_out (Union[pd.Series, pd.DataFrame]):
            The target values corresponding to `X_hold_out`.
        **kwargs:
            Additional keyword arguments to be passed to the `supervised_features_selection` function.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Index]: A tuple containing three items:
            - The normalized and feature-selected version of `X`,
            - The normalized and feature-selected version of `X_hold_out`,
            - The Index object of the feature-selected `X`.

    Raises:
        Any exception that `StandardScaler.fit_transform`, `StandardScaler.transform` or
        `supervised_features_selection` may raise due to invalid input or during execution.
    """
    # Perform normalization
    normalizer = StandardScaler()
    X_normed = normalizer.fit_transform(X.T).T
    X = pd.DataFrame(data=X_normed, index=X.index, columns=X.columns)
    X_hold_out_normed = normalizer.transform(X_hold_out.T).T
    X_hold_out = pd.DataFrame(data=X_hold_out_normed, index=X_hold_out.index, columns=X_hold_out.columns)

    # Before K-fold, perform fine feature selection using all training data, features are selected
    # if its ENet coefficients are non-zero. Note: Set `n_trials`=1 to use one ElasticNet
    sup_selected_feats = supervised_features_selection(
        X,
        y,
        alpha=0.01,
        l1_ratio=0.9,
        n_trials=1,
        boosting=False,
        n_features=25   # Keep the number of selected features limited to a reasonable number
    )
    X = sup_selected_feats # Note that output has its index sorted
    X_hold_out = X_hold_out.loc[X.index] # Also apply data to X_hold_out
    return X, X_hold_out, sup_selected_feats.index


def K_fold_training(X: pd.DataFrame,
                    y: pd.Series,
                    X_hold_out: pd.DataFrame,
                    y_hold_out: pd.Series,
                    model: sklearn.pipeline.Pipeline,
                    K: Optional[int] = 5,
                    **kwargs) -> Tuple[pd.DataFrame, pd.Index]:
    r"""Performs K-fold training and hold out validation using the specified model.

    Args:
        X (pd.DataFrame):
            The input features for the training set.
        y (pd.Series):
            The targets for the training set.
        X_hold_out (pd.DataFrame):
            The input features for the hold out set.
        y_hold_out (pd.Series):
            The targets for the hold out set.
        model (sklearn.pipeline.Pipeline):
            The machine learning model to use, must be compatible with sklearn.pipeline.Pipeline.
        K (int, optional):
            The number of folds for the K-fold cross validation, by default 5.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[pd.DataFrame, pd.Index]:
            A tuple containing a DataFrame of the scores from the K-fold training,
            and the index of the best model from the hold out validation.

    Raises:
        ValueError:
            If the input parameters are not in the expected format or if the
            model is not compatible with sklearn.pipeline.Pipeline.
    """
    # Check if the target is multi-class or binary
    assert isinstance(y, pd.Series) and isinstance(y_hold_out, pd.Series)

    if len(y.unique()) > 2:
        # For multi-class targets, perform one-hot encoding
        y_onehot = pd.get_dummies(y)
        y_hold_out_onehot = pd.get_dummies(y_hold_out)
        score_func = top_k_accuracy_score_
    else:
        y_onehot = y.copy()
        y_hold_out_onehot = y_hold_out.copy()
        score_func = sklearn.metrics.roc_auc_score

    scores = pd.DataFrame()
    kfold_splitter = StratifiedKFold(n_splits=K, shuffle=True)
    for fold, (train, test) in auto.tqdm(enumerate(kfold_splitter.split(X.T, y)), total=K, leave=False,
                                         disable=True):
        X_train = X.T.iloc[train]
        X_test = X_hold_out.T

        y_train = y_onehot.iloc[train]
        y_test = y_hold_out_onehot

        # copy the model instance which is supposed to have best hyperparameters
        _model = sklearn.base.clone(model)
        _model.fit(X_train, y_train)

        # Test the performance of this model
        pred = _model.predict(X_test)
        _score = score_func(y_test, pred)
        row = pd.Series(name=fold)
        row['score'] = _score
        row['model'] = _model
        scores = scores.join(row, how='outer')
    return scores