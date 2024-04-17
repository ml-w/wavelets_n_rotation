import pandas as pd
import numpy as np
import itertools
from scipy.stats import spearmanr
from typing import List, Optional, Sequence, Tuple, Union

def pct_change_with_R0(x: pd.DataFrame) -> pd.DataFrame:
    r"""This function calculates the percentage change between columns in a given DataFrame,
    using the formula (b - a) / (a + 1E-14). It takes the first column as a reference and
    calculates the percentage change for all other columns relative to it. The resulting
    DataFrame has the same column names as the input DataFrame, except for the first column.
    If any errors occur during the calculation, the function returns a copy of the input
    DataFrame with all values set to missing (NA).

    .. note::
        * The expected input must have three column levels. The last column level is being
          compared. The comparison is performed between each column with the first column.
        * To prevent division by zero, a small epsilon 1E-14 is installed into the factor.

    Args:
        x (pd.DataFrame): Input data frame to be normalized using :func:`io.standardize_df`.

    Returns:
        pd.DataFrame: The standardized dataframe with multi-index column and index.

    """
    if not x.columns.nlevels == 3:
        raise ValueError("Input dataframe must have 3 levels of columns. The last column level is being "
                         "compared. The comparison is performed between each column with  the first.")

    # display(x.columns)
    pairs = itertools.product([x.columns[0]], x.columns[1:])
    try:
        out_x = pd.concat([(x[b] - x[a]) / (x[a] + 1E-14) for a, b in pairs], axis=1)

        # reconstruct column names
        out_x.columns = x.columns[1:]
        out_x.columns = out_x.columns.droplevel([0, 1])
        out_x.columns.names = x.columns.names[2:]
    except:
        out_x = x.copy()
        out_x.iloc[:, :] = pd.NA
        out_x.columns = out_x.columns.droplevel([0, 1])
    return out_x

def identify_trend(s1: pd.Series, s2: pd.Series) -> Tuple[float, float]:
    r"""Perform Spearman's rank correlation on two series.

    The intended usage of this function is to compare the correlation between two series. These two series
    should have identical indices and have the same number of elements.

    Args:
        data (pd.DataFrame): Input dataframe.

    """
    # check if two series have the same length
    assert s1.index.identical(s2.index), "Two series has different indices"

    corr, pval = spearmanr(s1, s2)
    return corr, pval


def subdf_to_ordinal_coords(df: pd.DataFrame) -> pd.Series:
    r"""A helper function to convert grouped dataframe into coordinate of values and its column label.

    This function is written to facilitate spearman correlation analysis. This function should only be
    called from `df.groupby().apply()`. The input of this function is assumed to have only one row,
    where rows are the features and the columns are the patient with an ordinal class label embedded into
    the column headers.
    """
    # drop first two levels (PID, Lesion Code) as they are now irrelevant
    df.columns = df.columns.droplevel([0, 1])

    # the df should only have one single row
    out_series = df.iloc[0]
    return out_series

def identify_trend_from_df(df: pd.DataFrame) -> pd.DataFrame:
    r"""Calculates Spearman's rank correlation coefficient and corresponding p-value for each feature in the input
    DataFrame.

    The input DataFrame should have a 3-level column hierarchy, where each row corresponds to a feature and each
    column corresponds to a patient. The output is a DataFrame where each row represents a feature, and columns
    represent the correlation coefficient ('cor') and the p-value ('pval').

    Args:
        df (pd.DataFrame):
            A multi-level DataFrame where rows signify features and columns signify patients. The
            DataFrame should adhere to the format defined in `pct_change_with_R0`, where the last column level
            indicates the class label that is used as the ordinal variable and the row values used as the a
            continuous variable.

    Returns:
        pd.DataFrame: A DataFrame containing Spearman's correlation coefficient and p-value for each feature.

    Raises:
        AssertionError: If the input DataFrame doesn't have 3 levels of columns.

    .. notes::
        Spearman's rank correlation coefficient measures the statistical dependence between the rankings of two
        variables. The p-value roughly indicates the probability of an uncorrelated system producing datasets that
        have a Spearman correlation at least as extreme as the one computed from these datasets.

    Example:
        >>> df = pd.DataFrame({...})
        >>> result = identify_trend_from_df(df)
    """
    # Input must follow the format set in :func:`pct_change_with_R0`
    assert df.columns.nlevels == 3, "Input dataframe must have 3 levels of columns. Please see `pct_change_with_R0`."

    # rows are features and columns are patients
    series = []
    for feat_name, row in df.iterrows():
        # For each patient, convert the values into a lists of points
        coords = subdf_to_ordinal_coords(row.to_frame().T)
        # Get the class labels as a pd.Series
        categories = coords.index.to_series().reset_index(drop=True)
        # Get the features as a pd.Series
        variable = coords.reset_index(drop=True)
        # perform Spearmans
        corr, pval = spearmanr(variable, categories)
        s = pd.Series([corr, pval], name=feat_name, index=['cor', 'pval'])
        series.append(s)

    return pd.concat(series, axis=1).T
