import mri_radiomics_toolkit
import unittest
from feature_robustness_analysis.perf_analysis.grid_search import *
from feature_robustness_analysis.perf_analysis.k_fold_analysis import *


class TestGridSearch(unittest.TestCase):
    def test_normalizes_feature_sets(self):
        idx = list("12345")
        X_train = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
                                'feature2': [6, 7, 8, 9, 10]},
                               index=idx).T
        y_train = pd.Series([0, 1, 0, 1, 0], index=idx)

        X_hold_out = pd.DataFrame({'feature1': [6, 7, 8, 9, 10],
                                   'feature2': [11, 12, 13, 14, 15]},
                                  index=idx).T
        y_hold_out = pd.Series([1, 0, 1, 0, 1], index=idx)

        normalized_X_train, _, _ = normalize_n_feature_selection(X_train, y_train, X_hold_out, y_hold_out)

        scaler = StandardScaler()
        expected_X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

        assert normalized_X_train.equals(expected_X_train)

    def test_performs_feature_selection(self):
        idx = list("12345")
        X_train = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
                                'feature2': [6, 7, 8, 9, 10]},
                               index=idx).T
        y_train = pd.Series([0, 1, 0, 1, 0], index=idx)

        X_hold_out = pd.DataFrame({'feature1': [6, 7, 8, 9, 10],
                                   'feature2': [11, 12, 13, 14, 15]},
                                  index=idx).T
        y_hold_out = pd.Series([1, 0, 1, 0, 1], index=idx)

        _, _, selected_features = normalize_n_feature_selection(X_train, y_train, X_hold_out, y_hold_out)

        expected_selected_features = pd.Index(['feature1', 'feature2'])

        assert selected_features.equals(expected_selected_features)