import matplotlib.pyplot as plt
import unittest
from feature_robustness_analysis.plots import *


class TestPlots(unittest.TestCase):
    def test_plot_pct_diff(self):
        data = [
            [0.1, 0.1, 0.2, 0.3],
            [0.2, 0.1, 0.2, 0.5],
            [0.3, 0.1, 0.2, 0.4],
            [0.2, 0.4, 0.5, 0.6]
        ]
        columns = pd.MultiIndex.from_tuples(
            [('A', '', '1'),
             ('A', '', '2'),
             ('B', '', '1'),
             ('B', '', '2')],
            names=['Patient', 'Lesion Code', 'Rotation']
        )
        index = pd.MultiIndex.from_tuples([
            ('ImgFilter1', 'Cat1', 'F1'),
            ('ImgFilter1', 'Cat1', 'F2'),
            ('ImgFilter2', 'Cat2', 'F1'),
            ('ImgFilter3', 'Cat3', 'F1')
        ], names=['Filter', 'Category', 'Name'])

        input_df = pd.DataFrame(data, columns=columns, index=index)

        # Test without providing ax
        ax, summary_df = plot_percentage_difference(input_df)

        # Test with providing ax
        fig, ax = plt.subplots(1, 1)
        plot_percentage_difference(input_df, ax=ax)