import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pathlib


path_style = pathlib.Path(__file__).parent.joinpath(
    "fig.mplstyle"
)  # path to the matplotlib style guide
path_renders = pathlib.Path(__file__).parent.parent.parent.joinpath("renders")


class StyleConfig(object):
    """
    This class provides a consistent figure formatting and useful functions such as,
        - creating figure axes with very specific widths (determined by the published figure guidelines)
        - stores color maps, color palettes, line styles, etc. that can be reused across figures

    """

    def __init__(self):
        plt.style.use(str(path_style))  # load up the matplotlib style guide

        # color definitions
        self.color = ColorManager()

        # default image size
        self.two_col_width = self.mm2in(180)
        self.one_col_width = self.mm2in(88)

        # rasterization settings
        self.dpi = 1200
        self.bbox_inches = None
        self.force_dpi = True

        # output settings
        self.fig_exts = ["pdf", "png"]
        self.save_dir = path_renders

        # various kw_arg style dictionaries
        self.fill_style = dict(alpha=0.5)
        self.line_style = dict(alpha=1.0, ls="-")
        self.pmesh_style = dict(rasterized=True)
        self.contour_style = dict(linewidths=1.75, alpha=1.0, linestyles="-")

        return

    @staticmethod
    def mm2in(val):
        return val / 25.4

    def figure_single_axis(
        self, width_ax=60, height_ax=30, left=15, right=5, bottom=15, top=5
    ):
        """
        Creates a figure with one axis, where the axis size can be precisely set and the figure is based accordingly
        :param width_ax: width of the axis (not figure) in mm
        :param height_ax: height of the axis (not figure) in mm
        :param left: left margin for the axis
        :param right: right margin for the axis
        :param top: top margin for the axis
        :param bottom: bottom margin for the axis
        :return:
        """
        fig = plt.figure(
            figsize=[
                self.mm2in(width_ax + left + right),
                self.mm2in(height_ax + top + bottom),
            ]
        )
        hor = [
            Size.Fixed(self.mm2in(left)),
            Size.Fixed(self.mm2in(width_ax)),
            Size.Fixed(self.mm2in(right)),
        ]
        ver = [
            Size.Fixed(self.mm2in(bottom)),
            Size.Fixed(self.mm2in(height_ax)),
            Size.Fixed(self.mm2in(top)),
        ]

        divider = Divider(fig, (0, 0, 1, 1), hor, ver, aspect=False)
        ax = fig.add_axes(
            divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
        )
        return fig, ax

    def add_axis(
        self, fig, width_ax=60, height_ax=30, left=15, right=5, bottom=15, top=5
    ):
        hor = [
            Size.Fixed(self.mm2in(left)),
            Size.Fixed(self.mm2in(width_ax)),
            Size.Fixed(self.mm2in(right)),
        ]
        ver = [
            Size.Fixed(self.mm2in(bottom)),
            Size.Fixed(self.mm2in(height_ax)),
            Size.Fixed(self.mm2in(top)),
        ]

        divider = Divider(fig, (0, 0, 1, 1), hor, ver, aspect=False)
        ax = fig.add_axes(
            divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
        )
        return ax

    def grid_axes(
        self,
        nrows=3,
        ncols=3,
        width_ax=30,
        height_ax=30,
        left=15,
        right=5,
        bottom=15,
        top=5,
        width_space=10,
        height_space=10,
        sharex=False,
        sharey=False,
        squeeze=True,
    ):

        fig_width = ncols * (width_ax + width_space) + left + right
        fig_height = nrows * (height_ax + height_space) + top + bottom

        left = left / fig_width
        right = (fig_width - right) / fig_width
        top = (fig_height - top) / fig_height
        bottom = bottom / fig_height

        width_space = width_space / fig_height
        height_space = height_space / fig_height

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=[self.mm2in(fig_width), self.mm2in(fig_height)],
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze,
            gridspec_kw=dict(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                wspace=width_space,
                hspace=height_space,
            ),
        )
        return fig, axs

    @staticmethod
    def change_number_of_ticks(ax, num_xticks=None, num_yticks=None):
        for num, axis in zip([num_xticks, num_yticks], [ax.xaxis, ax.yaxis]):
            if num is not None:
                axis.set_major_locator(MaxNLocator(num))
        return

    def save_figure(self, fig, filename):
        kwargs = {"dpi": self.dpi, "transparent": True}
        metadata = {
            "Author": "Benjamin MacLellan",
            "Subject": "QOQI causal model selection",
        }
        self.save_dir.joinpath(filename).parent.mkdir(parents=True, exist_ok=True)

        # adds a rasterized, (almost) transparent rectangle below all layers to force the DPI when importing in Affinity
        if self.force_dpi:
            fig.patches.extend(
                [
                    plt.Rectangle(
                        (0.5, 0.5),
                        0.1,
                        0.1,
                        fill=True,
                        color="w",
                        alpha=0.01,
                        zorder=-999,
                        rasterized=True,
                        transform=fig.transFigure,
                    )
                ]
            )
        print(f"Saving to {str(self.save_dir.joinpath(filename))}")
        for ext in self.fig_exts:
            fig.savefig(
                self.save_dir.joinpath(filename + "." + ext),
                bbox_inches=self.bbox_inches,
                format=ext,
                metadata=metadata,
                **kwargs,
            )
        return


class ColorManager(object):
    """
    Manages global color schemes for the figures.
    Different types of color maps/palettes are stored here, along with helper functions.
    """

    def __init__(self):
        self.cat = sns.color_palette()
        self.map = sns.cubehelix_palette(
            start=0.0, rot=-0.75, reverse=False, as_cmap=True
        )
        self.pairs = sns.color_palette("Paired")

        # matplotlib.rcParams['axes.prop_cycle'] = cycler('color', self.cat)

    def info(self):
        print(f"ColorManager has the following color schemes available:")
        for pal in [self.cat, self.map, self.pairs]:
            print(pal)
