import pandas as pd
import unittest
from feature_robustness_analysis.io import standardize_df

class Test_IO(unittest.TestCase):
    def test_standardize_df(self):
        """Test that the function handles invalid column and index names. Add assertion for column dissection."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        df.index = ['ImagingFilter_Category_Name1', 'ImagingFilter_Category_Name2', 'ImagingFilter_Category_Name3']
        df.columns = ['LUNG1-1-A_R1', 'LUNG1-1-B_R2', 'LUNG1-2_R1']
        standardized_df = standardize_df(df)
        standardized_df.sort_index(inplace=True)
        assert standardized_df.shape == (3, 3)
        self.assertListEqual(['LUNG1-1', 'LUNG1-1', 'LUNG1-2'],
                             standardized_df.columns.get_level_values(0).to_list())
        self.assertListEqual(['A', 'B', ''],
                             standardized_df.columns.get_level_values(1).to_list())
        self.assertListEqual(['R1', 'R2', 'R1'],
                             standardized_df.columns.get_level_values(2).to_list())

if __name__ == '__main__':
    unittest.main()
