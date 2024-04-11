import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk

from typing import List, Optional, Sequence, Tuple, Union, Iterable
from .stats import pct_change_with_R0


def plot_percentage_difference(df: pd.DataFrame,
                               ax: Optional[plt.Axes] = None,
                               title: Optional[str] = None,
                               **kwargs) -> Tuple[plt.Axes, pd.DataFrame]:
    r"""Plot the percentage change between two columns in a DataFrame.

    This function asserts that the DataFrame and its index both have three levels. It calculates
    the percentage change with respect to the initial value and plots the data in a box plot.

    Args:
        df (pd.DataFrame):
            A pandas DataFrame with three levels in both columns and index.
        ax (Optional[plt.Axes], default=None):
            An optional Matplotlib Axes object to plot the graph on.
            If none is provided, a new Axes object will be created.
        title (Optional[str], default=None):
            An optional title for the plot.
        **kwargs:
            Additional keyword arguments to pass to seaborn's :func:`boxplot` function.

    Returns:
        plt.Axes:
            The Axes object with the plot drawn onto it.
        pd.DataFrame:
            A summary DataFrame with the mean and variance of percentage difference for each feature.

    Raises:
        AssertionError: If the DataFrame 'df' does not have three levels in its columns or index.
    """
    assert df.columns.nlevels == 3, "df must have 3 levels"
    assert df.index.nlevels == 3, "df index must have 3 levels"

    pct = df.groupby(axis=1, level=[0, 1]).apply(pct_change_with_R0)
    pct_melted = pct.reset_index().melt(id_vars=pct.index.names,
                                        var_name=df.columns.names).dropna()

    # Construct output summary of the percentage differences mean/var for each feature
    summary_out = pct_melted.copy()
    summary_out.set_index(df.index.names[0], drop=True, inplace=True)
    summary_out.set_index(df.index.names[1], drop=True, inplace=True, append=True)
    summary_out.set_index(df.index.names[2], drop=True, inplace=True, append=True)
    summary_out.set_index(df.columns.names[-1], drop=True, inplace=True, append=True)
    summary_out['value'] = summary_out['value'].astype('float')
    summary_out = pd.concat([summary_out.groupby(axis=0, level=[0, 1, 2, -1])['value'].median(),
                             summary_out.groupby(axis=0, level=[0, 1, 2, -1])['value'].quantile(0.25),
                             summary_out.groupby(axis=0, level=[0, 1, 2, -1])['value'].quantile(0.75)],
                            axis=1)
    summary_out.columns = ['Median', 'LQ', 'HQ']

    # Creates the ax if its not provided
    if ax is None:
        figsize = kwargs.pop('figsize', (10, 7))
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=False)

    # Generate box plot
    sns.boxplot(pct_melted,
                y='Name', x='value', hue=df.columns.names[-1], showfliers=False, ax=ax, **kwargs)

    # Set up axes labels
    ax.set_ylabel("Feature name")
    ax.set_xlabel("%$\Delta$")

    # Limit xlim to -100% to 100%
    xlim = ax.get_xlim()
    ax.set_xlim(max(xlim[0], -1), min(xlim[1], 1))

    # Aesthetic settings for display
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.set_xticklabels([f"{a * 100:.0f}%" for a in ax.get_xticks()])
    if not title is None:
        ax.set_title(title)
    return ax, summary_out


def display_image_in_3_planes(img: sitk.Image, seg: sitk.Image):
    r"""Display a SimpleITK image in 3 planes (sagittal, coronal, and axial)
    Note that this doesn't account for spacing.
    """
    # Convert SimpleITK image to a numpy array
    img_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(seg)

    # Calculate the center of the image
    center = [size // 2 for size in img.GetSize()]

    # Display the image in 3 planes
    fig, axs = plt.subplots(2, 3, figsize=(10, 4))

    # Sagittal plane
    axs[0][0].imshow(img_array[center[2], :, :], cmap="gray")
    axs[0][0].set_title("Axial plane")
    axs[1][0].imshow(seg_array[center[2], :, :], cmap="gray")
    axs[1][0].set_title("Axial plane")

    # Coronal plane
    axs[0][1].imshow(img_array[:, center[1], :], cmap="gray")
    axs[0][1].set_title("Coronal plane")
    axs[1][1].imshow(seg_array[:, center[1], :], cmap="gray")
    axs[1][1].set_title("Coronal plane")

    # Axial plane
    axs[0][2].imshow(img_array[:, :, center[0]], cmap="gray")
    axs[0][2].set_title("Sagittal plane")
    axs[1][2].imshow(seg_array[:, :, center[0]], cmap="gray")
    axs[1][2].set_title("Sagittal plane")

    # Show the image
    plt.show()
