import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from data_generation import FrankeData
from ordinary_least_squares import OrdinaryLeastSquares

# TODO: Scale these correctly
#  plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "figure.figsize": (4.7747, 5)})

sns.set()


def side_by_side_line_plot(
    title,
    plot_one_title,
    plot_one_x_data,
    plot_one_y_data,
    plot_one_labels,
    plot_one_x_label,
    plot_one_y_label,
    plot_two_title,
    plot_two_x_data,
    plot_two_y_data,
    plot_two_labels,
    plot_two_x_label,
    plot_two_y_label,
    filename="",
):
    """Plots lines side by side

    Parameters
    ----------
        title : str
            The title of the plots
        plot_one_title : str
            The title of the first plots
        plot_one_x_data : float[]
            The x data for the first plot
        plot_one_y_data : float[]
            The y data for the first plot
        plot_one_labels : str[]
            The labels for the first plot
        plot_one_x_label : str
            The label for the x-axis
        plot_one_y_label : str
            The label for the y-axis
        plot_two_title : str
            The title of the first plots
        plot_two_x_data : float[]
            The x data for the first plot
        plot_two_y_data : float[]
            The y data for the first plot
        plot_two_labels : str[]
            The labels for the first plot
        plot_two_x_label : str
            The label for the x-axis
        plot_two_y_label : str
            The label for the y-axis
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    ax[0].set_title(plot_one_title)
    for x_data, y_data, label in zip(plot_one_x_data, plot_one_y_data, plot_one_labels):
        sns.lineplot(x=x_data, y=y_data, label=label, ax=ax[0])
    ax[0].set(xlabel=plot_one_x_label, ylabel=plot_one_y_label)

    ax[1].set_title(plot_two_title)
    for x_data, y_data, label in zip(plot_two_x_data, plot_two_y_data, plot_two_labels):
        sns.lineplot(x=x_data, y=y_data, label=label, ax=ax[1])
    ax[1].set(xlabel=plot_two_x_label, ylabel=plot_two_y_label)

    plt.legend()
    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    plt.show()


def line_plot(title, x_datas, y_datas, data_labels, x_label, y_label, filename=""):
    """Plots a line plot

    Parameters
    ----------
        title : str
            The title of the plots
        x_datas : float[]
            The x data for the plot
        y_datas : float[]
            The y data for the plot
        data_labels : str[]
            The labels for the plot
        x_label : str
            The label for the x-axis
        y_label : str
            The label for the y-axis
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    plt.title(title)
    for x_data, y_data, label in zip(x_datas, y_datas, data_labels):
        sns.lineplot(x=x_data, y=y_data, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    plt.show()


def heat_plot(
    title,
    table_values,
    xticklabels,
    yticklabels,
    x_label,
    y_label,
    selected_idx=None,
    filename="",
):
    """Plots the heat plot

    Parameters
    ----------
        title : str
            The title of the plots
        table_values : float[][]
            The values of the values for which to plot in the heat plot
        xticklabels : str
            The labels for the ticks for the x-axis
        yticklabels : str
            The labels for the ticks for the y-axis
        x_label : str
            The label for the x-axis
        y_label : str
            The label for the y-axis
        selected_idx : tuple[int, int]
            The index for which to give an extra mark
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    g = sns.heatmap(table_values, xticklabels=xticklabels, yticklabels=yticklabels)
    from matplotlib.patches import Rectangle

    if selected_idx:
        g.add_patch(Rectangle(selected_idx, 1, 1, fill=False, edgecolor="blue", lw=3))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    plt.show()


def surface_plot_raveled(title, x, y, z_arr, subtitles, dimensions, filename=""):
    """Plot raveled values in a surface plot by unraveling them

    Parameters
    ----------
        title : str
            The title of the plots
        x : np.array
            The x values for which to plot
        y : np.array
            The y values for which to plot
        z_arr : np.array
            The z values for which to plot
        subtitles : str
            Subtitles for the plot
        dimensions : np.array
            The dimensions of the data
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    x = x.reshape(dimensions[0], dimensions[1])
    y = y.reshape(dimensions[0], dimensions[1])

    for col in range(len(z_arr)):
        for row in range(len(z_arr[0])):
            z_arr[col][row] = z_arr[col][row].reshape(dimensions[0], dimensions[1])

    surface_plot(title, x, y, z_arr, subtitles, filename)


def surface_plot(title, x, y, z_array, subtitles, filename=""):
    """Plot values in a surface plot

    Parameters
    ----------
        title : str
            The title of the plots
        x : np.array
            The x values for which to plot
        y : np.array
            The y values for which to plot
        z_array : np.array
            The z values for which to plot
        subtitles : str
            Subtitles for the plot
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    nrows, ncols = len(z_array[0]), len(z_array)

    fig = plt.figure()

    z_np_arr = np.array(z_array)
    vmin = np.min(z_np_arr)
    vmax = np.max(z_np_arr)

    axes = []
    for row in range(nrows):
        for col in range(ncols):
            ax = fig.add_subplot(ncols, nrows, ncols * col + row + 1, projection="3d")
            axes.append(ax)
            surf = ax.plot_surface(
                x,
                y,
                z_array[col][row],
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            #  ax.set_zlim(-0.10, 1.40) TODO: DO I want this?
            #  ax.set_zlim(vmin, vmax)
            #  ax.zaxis.set_major_locator(LinearLocator(10))
            #  ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

            ax.set_title(subtitles[col][row])

    cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes])
    plt.colorbar(surf, cax=cax, **kw)

    fig.suptitle(title)

    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    plt.show()


def plot_confidence_interval(title, x, y, y_err, x_label, y_label, filename=""):
    """Plot an error bar plot showing the confidence intervals

    Parameters
    ----------
        title : str
            The title of the plots
        x : np.array
            The x values for which to plot
        y : np.array
            The y values for which to plot
        y_err : np.array
            The errors for the confidence intervals
        x_label : str
            The label for the x-axis
        y_label : str
            The label for the y-axis
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    plt.title(title)
    plt.errorbar(x, y, y_err, fmt="o")

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    plt.show()
