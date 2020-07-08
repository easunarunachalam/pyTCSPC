from mpl_toolkits.axes_grid1 import make_axes_locatable

# for creating appropriately-sized colorbars for subplots
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)