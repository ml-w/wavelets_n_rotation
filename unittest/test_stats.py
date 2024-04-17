import pandas as pd
import numpy as np

from feature_robustness_analysis.stats import *
import unittest

class Test_States(unittest.TestCase):
    def test_pct_change_with_R0(self):
        '''
        Tests that the function works correctly when all values in the input dataframe are numeric
        '''
        data = [
            [0.1, 0.1, 0.2],
            [0.2, 0.1, 0.2],
            [0.3, 0.1, 0.2]
        ]
        columns = pd.MultiIndex.from_tuples(
            [('A', 'B', '1'), ('A', 'B', '2'), ('A', 'B', '3')]
        )
        index = pd.Index(['F1', 'F2', 'F3'])
        input_df = pd.DataFrame(data, columns=columns, index=index)
        expected_output = pd.DataFrame({'2': [0., -.5, -2/3.],
                                        '3': [1., 0., -1/3.]}, index=index)
        pd.testing.assert_frame_equal(pct_change_with_R0(input_df),
                                      (expected_output), check_dtype=False)

    def test_identify_trend(self):
        ordinal_var = pd.Series(np.random.randint(0, 5, size=[100]))
        features = pd.Series(np.random.random(size=[100]))
        cor, pval = spearmanr(ordinal_var, features)

    def test_subdf_to_ordinal_coord(self):
        data = [
            [0.1, 0.1, 0.2],
        ]
        columns = pd.MultiIndex.from_tuples(
            [('A', 'B', '1'), ('A', 'B', '2'), ('B', 'A', '1')]
        )
        index = pd.Index(['F1'])
        input_df = pd.DataFrame(data, columns=columns, index=index)
        input_df.groupby(axis=1, level=(0, 1)).apply(subdf_to_ordinal_coords)

    def test_identify_trend_from_df(self):
        data = [
            [0.1, 0.1, 0.2, 0.3],
            [0.2, 0.1, 0.2, 0.5],
            [0.3, 0.1, 0.2, 0.4],
            [0.2, 0.4, 0.5, 0.6]
        ]
        columns = pd.MultiIndex.from_tuples(
            [('A', '', '1'), ('A', '', '2'), ('B', '', '1'), ('B', '', '2')]
        )
        index = pd.Index(['F1', 'F2', 'F3', 'F4'])
        input_df = pd.DataFrame(data, columns=columns, index=index)
        corr_df = identify_trend_from_df(input_df)
        print(corr_df)

