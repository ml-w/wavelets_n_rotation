import re
import pandas as pd
import collections
from typing import Union, Iterable, Any, Hashable, Optional

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    r"""Standardize dataframe by changing the column and index.

    This function takes a pandas DataFrame and standardizes its columns and indices according to certain patterns.

    For columns, it uses a regular expression pattern to extract 'PID', 'LessionCode', and 'Rotation' from the
    original column names. The resulting new columns are a MultiIndex with these three levels. The regex pattern
    used is `(?P<PID>LUNG1-\d+)-?(?P<LESSION_CODE>\w?)_(?P<ROT_DEG>R\d+)`.

    For indices, it splits the original index names on underscore '_' and assumes three parts: 'ImagingFilter',
    'Category', and 'Name'. The resulting new index is a MultiIndex with these three levels.

    Args:
        df (pd.DataFrame):
            The input DataFrame to be standardized. Assumes the columns and index are in a specific format.

    Returns:
        pd.DataFrame: A new DataFrame with standardized column names and index.

    .. notes::
        The column change is based on a regex pattern "(?P<PID>LUNG1-\d+)-?(?P<LESSION_CODE>\w?)_(?P<ROT_DEG>R\d+)".
        The index change is based on splitting the original index names on underscore.
    """
    ori_index = df.index
    ori_col = df.columns
    # regpat for columns
    regpat = "(?P<PID>LUNG1-\d+)-?(?P<LESSION_CODE>\w?)_(?P<ROT_DEG>R\d+)"

    # Define new list
    new_column = pd.MultiIndex.from_tuples(
        ["{PID},{LESSION_CODE},{ROT_DEG}".format(**re.search(regpat, s).groupdict()).split(',') for s in ori_col],
        names=["PID", "LessionCode", "Rotation"]
    )
    new_index = pd.MultiIndex.from_tuples(
        [s.split("_") for s in ori_index],
        names=["ImagingFilter", "Category", "Name"]
    )
    new_df = df.copy()
    new_df.index = new_index
    new_df.columns = new_column
    return new_df


def prepend_index_level(df: pd.DataFrame,
                        idx: Union[Iterable[Any], Hashable],
                        name: Any, axis: Optional[int] = 0) -> pd.DataFrame:
    r"""Prepends a level to the index of a DataFrame.

    Adds a new index level to the given DataFrame `df` at the specified `axis`. The new level will have the values specified by `idx` and the name `name`. If `axis` is 0, the level will be added to the row index; if `axis` is 1, the level will be added to the column index.

    If `idx` is a single hashable value, it will be repeated to match the length of the original index/column. If `idx` is an iterable, it should be of the same length as the original index/column. The new index level will be a tuple resulting from zipping `idx` and the original index/column.

    Args:
        df (pd.DataFrame): The DataFrame to which the index level is to be added.
        idx (Union[Iterable[Any], Hashable]): An iterable of values for the new index level or a single hashable value to be repeated.
        name (Any): The name of the new index level.
        axis (Optional[int]): The axis at which to add the new index level. 0 for rows, 1 for columns. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame with the added index level.

    Raises:
        ValueError: If `axis` is not 0 or 1, or if `idx` is an iterable but its length does not match the length of the original index/column.
    """
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    old_idx = df.index if axis == 0 else df.columns
    if isinstance(idx, collections.abc.Hashable):
        new_idx_values = [idx] * len(old_idx)
    elif isinstance(idx, collections.abc.Iterable):
        new_idx_values = list(idx)
        if len(new_idx_values) != len(old_idx):
            raise ValueError("`idx` must have the same length as the original index/column.")
    else:
        raise ValueError("`idx` must be a hashable or an iterable.")

    # Create new multi-index
    new_idx_tuples = list([_newidx, *_oldidx] for _newidx, _oldidx in zip(new_idx_values, old_idx))
    new_idx = pd.MultiIndex.from_tuples(new_idx_tuples, names=[name] + old_idx.names)

    # Return DataFrame with new index
    if axis == 0:
        return df.set_index(new_idx)
    elif axis == 1:
        return df.T.set_index(new_idx).T