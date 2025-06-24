import matplotlib as mpl

from numpy import abs, arange, arctan, arctan2, asarray, atleast_2d, concatenate, cos, diff
from numpy import float32, float64, gradient, hstack, isnan, linspace, log10, meshgrid, nanmax
from numpy import nanmin, ndim, ones, ones_like, pi, unique, s_, sin, size, sqrt, where, zeros
from numpy import all as npall

from numpy.ma import masked_invalid

from matplotlib import colormaps
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, Colormap

import matplotlib.gridspec as gridspec

from ..base import utilities
from .colormaps import wellSeparated, tatarize, white_to_colour
from copy import copy

def pretty(ax):
    """Make a plot with nice axes.

    Removes fluff from the axes.

    Parameters
    ----------
    ax : matplotlib .Axes
        A .Axes class from for example ax = plt.subplot(111), or ax = plt.gca()

    """
    # Remove the plot frame lines.
    ax = visible_border(ax, 'sw')
    ax = visible_ticks(ax, 'sw')
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()

    return ax

def visible_ticks(ax, sides='ws'):
    """Set the ticks to be visible on the specified sides of the plot.

    Parameters
    ----------
    sides : str
        'ws' for west and south, 'en' for east and north, 'all' for all sides.

    """
    if 'n' in sides:
        ax.get_xaxis().tick_top()
    if 's' in sides:
        ax.get_xaxis().tick_bottom()
    if 'e' in sides:
        ax.get_yaxis().tick_right()
    if 'w' in sides:
        ax.get_yaxis().tick_left()

    return ax

def visible_border(ax, sides='ws'):

    ax.spines["top"].set_visible('n' in sides)
    ax.spines["bottom"].set_visible('s' in sides)
    ax.spines["right"].set_visible('e' in sides)
    ax.spines["left"].set_visible('w' in sides)

    return ax


def filter_color_kwargs(kwargs):
    defaults = {'alpha' : 1.0,
                'alpha_color' : [1, 1, 1],
                'cax' : None,
                'clabel' : None,
                'clim_scaling' : None,
                'cmap' : None,
                'cmapIntervals' : None,
                'cbar_title' : None,
                'colorbar' : True,
                'equalize' : False,
                'nBins' : 256,
                'orientation' : 'vertical',
                'ticks' : None,
                'wrap_clabel' : False}

    out, kwargs = _filter_kwargs(kwargs, defaults)

    if out['cmap'] is None:
        out['cmap'] = 'cividis'
    out['cmap'] = copy(colormaps.get_cmap(out['cmap']))
    # out['cmap'].set_bad(color='white')
    return out, kwargs

def filter_plotting_kwargs(kwargs):
    defaults = {'ax' : None,
                'flip' : False,
                'flipX' : False,
                'flipY' : False,
                'grid' : False,
                'hillshade' : None,
                'labels' : True,
                'legend_size' : None,
                'log' : None,
                'logX' : None,
                'logY' : None,
                'trim' : None,
                'reciprocateX' : False,
                'reciprocateY' : False,
                'shading' : 'auto',
                'transpose' : False,
                'width' : None,
                'wrap_xlabel' : False,
                'wrap_ylabel' : False,
                'xlabel' : True,
                'xlim' : None,
                'xscale' : 'linear',
                'ylabel' : True,
                'ylim' : None,
                'yscale': 'linear'}

    out, kwargs = _filter_kwargs(kwargs, defaults)
    if out['grid']:
        out['color'] = 'k'
    if out['ax'] is None:
        out['ax'] = plt.gca()

    if out['transpose']:
        out['flipX'], out['flipY'] = out['flipY'], out['flipX']
        out['xlim'], out['ylim'] = out['ylim'], out['xlim']
        out['wrap_xlabel'], out['wrap_ylabel'] = out['wrap_ylabel'], out['wrap_xlabel']
        out['xlabel'], out['ylabel'] = out['ylabel'], out['xlabel']
        out['logX'], out['logY'] = out['logY'], out['logX']
        out['flipX'], out['flipY'] = out['flipY'], out['flipX']
        out['reciprocateX'], out['reciprocateY'] = out['reciprocateY'], out['reciprocateX']
        out['xscale'], out['yscale'] = out['yscale'], out['xscale']


    return out, kwargs

def _filter_kwargs(kwargs, defaults):
    tmp = copy(kwargs)
    subset = {k: tmp.pop(k, defaults[k]) for k in defaults.keys()}
    return subset, tmp

def clabel(cb, label, length=20, wrap=False, **kwargs):
    """Create a colourbar label with default fontsizes

    Parameters
    ----------
    cb : matplotlib.colorbar.Colorbar
        A colourbar to label
    label : str
        The colourbar label

    """
    if label == False:
        return
    if wrap:
        label = utilities.wrap_string(label, length)
    cb.ax.set_ylabel(label, **kwargs)

def generate_subplots(n, ax=None):
    """Generates subplots depending on whats given

    Parameters
    ----------
    n : int
        number of subplots
    ax : variable, optional
        List of subplots.
        gridspec.GridSpec or gridspec.Subplotspec
        list of gridspec.Subplotspec
        Defaults to None.

    """
    if ax is None:
        if n > 1:
            ax = [plt.subplot(1, n, p+1) for p in range(n)]
    else:
        if isinstance(ax, (gridspec.GridSpec, gridspec.SubplotSpec)):
            gs = ax.subgridspec(1, n)
            ax = [plt.subplot(gs[:, i]) for i in range(n)]
        else:
            assert len(ax) == n, ValueError("Need {} subplots to match the number of posteriors")
            ax2 = []
            for a in ax:
                if isinstance(a, gridspec.SubplotSpec):
                    ax2.append(plt.subplot(a))
                else:
                    ax2.append(a)
            ax = ax2
    return ax

def hillshade(arr, azimuth=30, altitude=30):
    """Create hillshade from a numpy array containing elevation data.

    Taken from https://github.com/royalosyin/Work-with-DEM-data-using-Python-from-Simple-to-Complicated/blob/master/ex07-Hillshade%20from%20a%20Digital%20Elevation%20Model%20(DEM).ipynb

    Parameters
    ----------
    arr : numpy array of shape (rows, columns)
        Numpy array containing elevation values to be used to created hillshade.
    azimuth : float (default=30)
        The desired azimuth for the hillshade.
    altitude : float (default=30)
        The desired sun angle altitude for the hillshade.

    Returns
    -------
    numpy array
        A numpy array containing hillshade values.

    """
    try:
        x, y = gradient(arr)
    except:
        raise ValueError("Input array should be two-dimensional")

    if azimuth <= 360.0:
        azimuth = 360.0 - azimuth
        azimuthrad = azimuth * pi / 180.0
    else:
        raise ValueError(
            "Azimuth value should be less than or equal to 360 degrees"
        )

    if altitude <= 90.0:
        altituderad = altitude * pi / 180.0
    else:
        raise ValueError(
            "Altitude value should be less than or equal to 90 degrees"
        )

    slope = pi / 2.0 - arctan(sqrt(x * x + y * y))
    aspect = arctan2(-x, y)

    shaded = sin(altituderad) * sin(slope) + cos(altituderad) * cos(slope) * cos((azimuthrad - pi / 2.0) - aspect)

    return 255 * (shaded + 1) / 2

def bar(values, edges, line=None, **kwargs):
    """Plot a bar chart.

    Parameters
    ----------
    values : array_like or StatArray
        Bar values
    edges : array_like or StatArray
        edges of the bars

    Other Parameters
    ----------------

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
        matplotlib.pyplot.hist : For additional keyword arguments you may use.

    """
    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    kwargs['color'] = kwargs.get('color', wellSeparated[0])
    kwargs['linewidth'] = kwargs.get('linewidth', 0.5)
    kwargs['edgecolor'] = kwargs.get('edgecolor', 'k')

    ax = geobipy_kwargs.pop('ax', plt.gca())

    pretty(ax)

    if geobipy_kwargs['reciprocateX']:
        bins = 1.0 / edges

    if geobipy_kwargs['xlabel'] != False:
        geobipy_kwargs['xlabel'] = utilities.getNameUnits(edges)
        # if geobipy_kwargs['logX']:
        #     bins, logLabel = utilities._log(edges, geobipy_kwargs['logX'])
        #     geobipy_kwargs['xlabel'] = logLabel + geobipy_kwargs['xlabel']

    if geobipy_kwargs['ylabel'] != False:
        geobipy_kwargs['ylabel'] = utilities.getNameUnits(values)

    i0 = 0
    i1 = size(values) - 1
    trim = geobipy_kwargs['trim']

    if npall(values == 0):
        trim = None

    if (trim is not None):
        while values[i0] == trim:
            i0 += 1
        while values[i1] == trim:
            i1 -= 1

    if (i1 >= i0):
        values = values[i0:i1+1]
        edges = edges[i0:i1+2]

    width = abs(diff(edges))
    centres = edges[:-1] + 0.5 * (diff(edges))

    if (geobipy_kwargs['transpose']):
        ax.barh(centres, values, height=width, align='center', alpha=color_kwargs['alpha'], **kwargs)
        xl = ylabel
        yl = xlabel
        geobipy_kwargs['xscale'], geobipy_kwargs['yscale'] = geobipy_kwargs['yscale'], geobipy_kwargs['xscale']
    else:
        ax.bar(centres, values, width=width, align='center', alpha=color_kwargs['alpha'], **kwargs)
        xl = xlabel
        yl = ylabel

    xl(ax, geobipy_kwargs['xlabel'], wrap=geobipy_kwargs['wrap_xlabel'])
    yl(ax, geobipy_kwargs['ylabel'], wrap=geobipy_kwargs['wrap_ylabel'])

    if geobipy_kwargs['flipX']:
        ax.invert_xaxis()

    if geobipy_kwargs['flipY']:
        ax.invert_yaxis()

    ax.set_xscale(geobipy_kwargs['xscale'])
    ax.set_yscale(geobipy_kwargs['yscale'])

    if geobipy_kwargs['xscale'] == 'linear':
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 2))
    if geobipy_kwargs['yscale'] == 'linear':
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 2))

    return ax

def pcolor(values, x=None, y=None, **kwargs):
    """Create a pseudocolour plot of a 2D array, Actually uses pcolormesh for speed.

    Create a colour plot of a 2D array.
    If the arrays x, y, and values are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    Parameters
    ----------
    values : array_like or StatArray
        A 2D array of colour values.
    x : 1D array_like or StatArray, optional
        Horizontal coordinates of the values edges.
    y : 1D array_like or StatArray, optional
        Vertical coordinates of the values edges.

    Other Parameters
    ----------------
    alpha : scalar or array_like, optional
        If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
        If array_like, each pixel is given an individual alpha value.
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    equalize : bool, optional
        Equalize the histogram of the colourmap so that all colours have an equal amount.
    nbins : int, optional
        Number of bins to use for histogram equalization.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    grid : bool, optional
        Plot the grid
    noColorbar : bool, optional
        Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
    reciprocateX : bool, optional
        Take the reciprocal of the X axis before other transforms
    reciprocateY : bool, optional
        Take the reciprocal of the Y axis before other transforms
    trim : bool, optional
        Set the x and y limits to the first and last non zero values along each axis.
    classes : dict, optional
            A dictionary containing three entries.
            classes['id'] : array_like of same shape as self containing the class id of each element in self.
            classes['cmaps'] : list of matplotlib colourmaps.  The number of colourmaps should equal the number of classes.
            classes['labels'] : list of str.  The length should equal the number of classes.
            If classes is provided, alpha is ignored if provided.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

    """
    geobipy_kwargs, _ = filter_plotting_kwargs(kwargs)

    if (x is None):
        mx = arange(size(values,-1)+1)
    else:
        mx = asarray(x)
        if ndim(x) < 2:
            if (x.size == values.shape[1]):
                mx = x.edges()
            else:
                assert x.size == values.shape[1]+1, ValueError('x must be size {}. Not {}'.format(values.shape[1]+1, x.size))

    if (y is None):
        my = arange(size(values,0)+1)
    else:
        my = asarray(y)
        if ndim(y) < 2:
            if (y.size == values.shape[0]):
                my = y.edges()
            else:
                assert y.size == values.shape[0]+1, ValueError('y must be size {}. Not {}'.format(values.shape[0]+1, y.size))

    if geobipy_kwargs['transpose']:
        mx, my = my, mx
        values = values.T

    if geobipy_kwargs['reciprocateX']:
        mx = 1.0 / mx

    if geobipy_kwargs['reciprocateY']:
        my = 1.0 / my

    if geobipy_kwargs['xlabel'] != False:
        geobipy_kwargs['xlabel'] = utilities.getNameUnits(x)

    if geobipy_kwargs['ylabel'] != False:
        geobipy_kwargs['ylabel'] = utilities.getNameUnits(y)

    mx, _ = utilities._log(mx, geobipy_kwargs['logX'])
    my, _ = utilities._log(my, geobipy_kwargs['logY'])

    if ndim(mx) == 1 and ndim(my) == 1:
        mx, my = meshgrid(mx, my)

    ax, pm, cb = pcolormesh(X=mx, Y=my, values=values, **kwargs)

    if isinstance(ax, list):
        for a in ax:
            xlabel(a, geobipy_kwargs['xlabel'], wrap=geobipy_kwargs['wrap_xlabel'])
            ylabel(a, geobipy_kwargs['ylabel'], wrap=geobipy_kwargs['wrap_ylabel'])
    else:
        xlabel(ax, geobipy_kwargs['xlabel'], wrap=geobipy_kwargs['wrap_xlabel'])
        ylabel(ax, geobipy_kwargs['ylabel'], wrap=geobipy_kwargs['wrap_ylabel'])

    return ax, pm, cb


def pcolormesh(X, Y, values, **kwargs):

    classes = kwargs.pop('classes', None)

    if classes is None:
        kwargs.pop('axis', None)
        ax, pm, cb = _pcolormesh(X, Y, values, **kwargs)

    else:
        originalAlphaColour = kwargs.pop('alpha_colour', [1, 1, 1])
        kwargs.pop('cmap', None)

        n_classes = values.shape[kwargs['axis']]

        classes['cmaps'] = classes.get('cmaps', ['r','g','b'])
        classes['labels'] = classes.get('labels', [str(i) for i in range(n_classes)])

        assert len(classes['cmaps']) == n_classes, Exception("Number of colour maps must be {}".format(n_classes))
        assert len(classes['labels']) == n_classes, Exception("Number of labels must be {}".format(n_classes))

        split = classes.pop("split_plot", False)

        if split:
            ax, pm, cb = _pcolormesh_3d_cbar_split(X, Y, values, classes, **kwargs)
        else:
            ax, pm, cb = _pcolormesh_3d_cbar_single(X, Y, values, classes, **kwargs)



    def _pcolormesh_3d_cbar_split(X, Y, values, classes, **kwargs):

        originalAlpha = kwargs.pop('alpha', None)

        classId = classes['id']
        cmaps = classes['cmaps']
        labels = classes['labels']
        axis = classes['axis']

        # Set up the grid for plotting
        gs1 = gridspec.GridSpec(nrows=1, ncols=1, left=0.1, right=0.70, wspace=0.05)
        gs2 = gridspec.GridSpec(nrows=1, ncols=2*nClasses, left=0.71, right=0.95, wspace=1.0)

        cbAx = []
        for i in range(nClasses):
            cbAx.append(plt.subplot(gs2[:, 2*i]))
        axMain = plt.subplot(gs1[:, :])

        ax = []; pm = []; cb = []
        for i in range(nClasses):
            cn = classNumber[i]
            cmap = cmaps[i]
            cmaptmp = cmap
            if not isinstance(cmap, Colormap):
                cmaptmp = white_to_colour(cmap)
            label = labels[i]

            s = [s_[:] for i in range(values.shape[axis])]; s[axis] = i; s = tuple(s)
            # Set max transparency for pixels not belonging to the current class.
            alpha = ones_like(values[s])
            alpha[classId != cn] = 0.0

            if not originalAlpha is None:
                alpha *= originalAlpha

            a, p, c = _pcolormesh(X, Y, values[s], alpha=alpha, cmap=cmaptmp, cax=cbAx[i], **kwargs)

            c.ax.set_ylabel(label)
            ax.append(a); pm.append(p); cb.append(c)

    return ax, pm, cb

def _pcolormesh_3d_cbar_single(X, Y, values, classes, **kwargs):

        import numpy as np

        class_id = classes['id']
        assert class_id is not None, ValueError(f"must provide class id of shape {X.shape-1}")
        cmaps = classes['cmaps']
        labels = classes['labels']
        axis = kwargs.pop('axis')

        n_classes = len(cmaps)

        width_ratios = [6]
        width_ratios.extend([0.3 for i in range(n_classes)])

        # Set up the grid for plotting
        gs = gridspec.GridSpec(nrows=1, ncols=n_classes+1, width_ratios=width_ratios, wspace=0.3)

        cbAx = []
        for i in range(n_classes):
            ax = plt.subplot(gs[0, i+1])
            ax = visible_border(ax)
            ax = visible_ticks(ax)
            cbAx.append(pretty(ax))
        ax = plt.subplot(gs[0, 0])
        kwargs['ax'] = pretty(ax)

        unique_ids = unique(class_id)
        for i in range(n_classes):
            cn = unique_ids[i]
            cmap = cmaps[i]
            if not isinstance(cmap, Colormap):
                cmap = white_to_colour(cmap)
            label = labels[i]

            s = [s_[:] for i in range(values.shape[axis])]; s[axis] = i; s = tuple(s)

            subset = np.ma.masked_where(class_id != cn, values[s])

            a, p, c = _pcolormesh(X, Y, subset, cmap=cmap, cax=cbAx[i], clabel=(i==n_classes-1), **kwargs)

            c.ax.set_title(label)

        p = None

        return kwargs['ax'], p, cbAx


def _pcolormesh(X, Y, values, **kwargs):
    """Create a pseudocolour plot of a 2D array, Actually uses pcolormesh for speed.

    Create a colour plot of a 2D array.
    If the arrays x, y, and values are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    Parameters
    ----------
    values : array_like or StatArray
        A 2D array of colour values.
    X : 1D array_like or StatArray, optional
        Horizontal coordinates of the values edges.
    Y : 1D array_like or StatArray, optional
        Vertical coordinates of the values edges.

    Other Parameters
    ----------------
    alpha : scalar or array_like, optional
        If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
        If array_like, each pixel is given an individual alpha value.
    alphaColour : 'trans' or length 3 array
        If 'trans', low alpha values are mapped to transparency
        If 3 array, each entry is the RGB value of a colour to map to, e.g. white = [1, 1, 1].
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    equalize : bool, optional
        Equalize the histogram of the colourmap so that all colours have an equal amount.
    nbins : int, optional
        Number of bins to use for histogram equalization.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    grid : bool, optional
        Plot the grid
    noColorbar : bool, optional
        Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
    trim : array_like, optional
        Set the x and y limits to the first and last locations that don't equal the values in trim.
    classes : dict, optional
            A dictionary containing three entries.
            classes['id'] : array_like of same shape as self containing the class id of each element in self.
            classes['cmaps'] : list of matplotlib colourmaps.  The number of colourmaps should equal the number of classes.
            classes['labels'] : list of str.  The length should equal the number of classes.
            If classes is provided, alpha is ignored if provided.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

    """

    assert ndim(values) == 2, ValueError(f'Number of dimensions is {ndim(values)} but must be 2')

    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    kwargs['cmap'] = color_kwargs['cmap']
    kwargs['shading'] = geobipy_kwargs['shading']

    ax = geobipy_kwargs['ax']

    pretty(ax)

    # Gridlines
    if 'edgecolor' in kwargs:
        geobipy_kwargs['grid'] = True

    if geobipy_kwargs['grid']:
        kwargs['edgecolor'] = kwargs.pop('edgecolor', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)

    values = values.astype('float64')

    rang = values.max() - values.min()
    if not geobipy_kwargs['trim'] is None and rang > 0.0:
        assert isinstance(geobipy_kwargs['trim'], (float, float32, float64)), TypeError("trim must be a float")
        bounds = utilities.findFirstLastNotValue(values, geobipy_kwargs['trim'])
        X = X[bounds[0, 0]:bounds[0, 1]+2, bounds[1, 0]:bounds[1, 1]+2]
        Y = Y[bounds[0, 0]:bounds[0, 1]+2, bounds[1, 0]:bounds[1, 1]+2]
        values = values[bounds[0, 0]:bounds[0, 1]+1, bounds[1, 0]:bounds[1, 1]+1]

    values, logLabel = utilities._log(values, geobipy_kwargs['log'])

    if color_kwargs['clabel'] != False:
        color_kwargs['clabel'] = utilities.getNameUnits(values)
        if (geobipy_kwargs['log']):
            color_kwargs['clabel'] = logLabel + color_kwargs['clabel']

    if color_kwargs['equalize']:
        assert color_kwargs['nBins'] > 0, ValueError('nBins must be greater than zero')
        values, _ = utilities.histogramEqualize(values, nBins=color_kwargs['nBins'])

    if color_kwargs['clim_scaling'] is not None:
        values = utilities.trim_by_percentile(values, color_kwargs['clim_scaling'])

    if geobipy_kwargs['hillshade'] is not None:
        kw = geobipy_kwargs['hillshade'] if isinstance(geobipy_kwargs['hillshade'], dict) else {}
        values = hillshade(values, azimuth=kw.pop('azimuth', 30), altitude=kw.pop('altitude', 30))

    if npall(values.shape == X.shape) or (npall(values.shape == asarray(X.shape)-1)):
        if npall(X.shape == Y.shape[::-1]):
            Y = Y.T
    else:
        X = X.T

    if ((X[1, 0] - X[0, 0]) == 0.0) & ((Y[1, 0] - Y[0, 0]) == 0.0):
        Y = Y.T

    pm = ax.pcolormesh(X, Y, values, alpha = color_kwargs['alpha'], **kwargs)

    ax.set_xscale(geobipy_kwargs['xscale'])
    ax.set_yscale(geobipy_kwargs['yscale'])

    if geobipy_kwargs['flipX']:
        ax.invert_xaxis()

    if geobipy_kwargs['flipY']:
        ax.invert_yaxis()

    if geobipy_kwargs['xlim'] is not None:
        ax.set_xlim(geobipy_kwargs['xlim'])

    if geobipy_kwargs['ylim'] is not None:
        ax.set_ylim(geobipy_kwargs['ylim'])

    cbar = None
    if (color_kwargs['colorbar']):
        if (color_kwargs['equalize']):
            cbar = plt.colorbar(pm, extend='both', cax=color_kwargs['cax'], orientation=color_kwargs['orientation'])
        else:
            cbar = plt.colorbar(pm, cax=color_kwargs['cax'], orientation=color_kwargs['orientation'], ticks=color_kwargs['ticks'])

        clabel(cbar, color_kwargs['clabel'], wrap=color_kwargs['wrap_clabel'])

    return ax, pm, cbar

def pcolor_as_bar(X, Y, values, **kwargs):
    """Create a pseudocolour plot of a 2D array, Actually uses pcolormesh for speed.

    Create a colour plot of a 2D array.
    If the arrays x, y, and values are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    Parameters
    ----------
    values : array_like or StatArray
        A 2D array of colour values.
    X : 1D array_like or StatArray, optional
        Horizontal coordinates of the values edges.
    Y : 1D array_like or StatArray, optional
        Vertical coordinates of the values edges.

    Other Parameters
    ----------------
    alpha : scalar or array_like, optional
        If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
        If array_like, each pixel is given an individual alpha value.
    alphaColour : 'trans' or length 3 array
        If 'trans', low alpha values are mapped to transparency
        If 3 array, each entry is the RGB value of a colour to map to, e.g. white = [1, 1, 1].
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    equalize : bool, optional
        Equalize the histogram of the colourmap so that all colours have an equal amount.
    nbins : int, optional
        Number of bins to use for histogram equalization.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    grid : bool, optional
        Plot the grid
    noColorbar : bool, optional
        Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
    trim : array_like, optional
        Set the x and y limits to the first and last locations that don't equal the values in trim.
    classes : dict, optional
            A dictionary containing three entries.
            classes['id'] : array_like of same shape as self containing the class id of each element in self.
            classes['cmaps'] : list of matplotlib colourmaps.  The number of colourmaps should equal the number of classes.
            classes['labels'] : list of str.  The length should equal the number of classes.
            If classes is provided, alpha is ignored if provided.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

    """

    assert ndim(values) == 2, ValueError('Number of dimensions must be 2')

    ax = kwargs.pop('ax', None)

    pretty(ax)

    kwargs['shading'] = 'auto'

    xscale = kwargs.pop('xscale', 'linear')
    yscale = kwargs.pop('yscale', 'linear')
    flipX = kwargs.pop('flipX', False)
    flipY = kwargs.pop('flipY', False)
    transpose = kwargs.pop('transpose', False)

    xlim_u = kwargs.pop('xlim', None)
    ylim_u = kwargs.pop('ylim', None)

    # Colourbar
    equalize = kwargs.pop('equalize', False)
    clim_scaling = kwargs.pop('clim_scaling', None)
    colorbar = kwargs.pop('colorbar', False)
    cax = kwargs.pop('cax', None)
    cmap = kwargs.pop('cmap', 'viridis')
    cmapIntervals = kwargs.pop('cmapIntervals', None)
    kwargs['cmap'] = copy(colormaps.get_cmap(cmap, cmapIntervals))
    kwargs['cmap'].set_bad(color='white')
    orientation = kwargs.pop('orientation', 'vertical')
    cl = kwargs.pop('clabel', None)

    # Values
    trim = kwargs.pop('trim', None)
    log = kwargs.pop('log', False)
    alpha = kwargs.pop('alpha', 1.0)
    alphaColour = kwargs.pop('alpha_color', [1, 1, 1])

    # Gridlines
    grid = kwargs.pop('grid', False)
    if 'edgecolor' in kwargs:
        grid = True
    if grid:
        kwargs['edgecolor'] = kwargs.pop('edgecolor', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)

    values = values.astype('float64')

    if (log):
        values, logLabel = utilities._log(values, log)

    if equalize:
        nBins = kwargs.pop('nbins', 256)
        assert nBins > 0, ValueError('nBins must be greater than zero')
        values, dummy = utilities.histogramEqualize(values, nBins=nBins)

    if not clim_scaling is None:
        values = utilities.trim_by_percentile(values, clim_scaling)

    Zm = masked_invalid(values, copy=False)

    pm = ax.pcolormesh(X, Y, Zm, alpha = alpha, **kwargs)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if flipX:
        ax.invert_xaxis()

    if flipY:
        ax.invert_yaxis()

    if not xlim_u is None:
        ax.set_xlim(xlim_u)

    if not ylim_u is None:
        ax.set_ylim(ylim_u)


    cbar = None
    if (colorbar):
        if (equalize):
            cbar = plt.colorbar(pm, extend='both', cax=cax, orientation=orientation)
        else:
            cbar = plt.colorbar(pm, cax=cax, orientation=orientation)

        if cl is None:
            if (log):
                clabel(cbar, logLabel + utilities.getNameUnits(values))
            else:
                clabel(cbar, utilities.getNameUnits(values))
        else:
            clabel(cbar, cl)

    return ax, pm, cbar

def pcolor_1D(values, y=None, **kwargs):
    """Create a pseudocolour plot of an array, Actually uses pcolormesh for speed.

    Create a colour plot of an array.
    If the arrays x, y, and values are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    Parameters
    ----------
    values : array_like or StatArray
        An array of colour values.
    x : 1D array_like or StatArray
        Horizontal coordinates of the values edges.
    y : 1D array_like or StatArray, optional
        Vertical coordinates of the values edges.

    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    equalize : bool, optional
        Equalize the histogram of the colourmap so that all colours have an equal amount.
    nbins : int, optional
        Number of bins to use for histogram equalization.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'
    flipY : bool, optional
        Flip the Y axis
    clabel : str, optional
        colourbar label
    grid : bool, optional
        Show the grid lines
    transpose : bool, optional
        Transpose the image
    noColorbar : bool, optional
        Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

    """
    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    # Get the figure axis
    ax = geobipy_kwargs['ax']

    pretty(ax)

    # Set the x and y axes before meshgridding them
    if (y is None):
        mx = arange(size(values)+1)
        my = asarray([0.0, 0.1*(nanmax(mx)-nanmin(mx))])
    else:
        assert y.size == values.size+1, ValueError('y must be size '+str(values.size+1))
        mx = y
        key = 'yscale' if geobipy_kwargs['transpose'] else 'xscale'
        if geobipy_kwargs[key] == 'log':
            tmp = log10(y)
            my = asarray([0.0, 0.1*(nanmax(tmp)-nanmin(tmp))])
        else:
            my = asarray([0.0, 0.1*(nanmax(y)-nanmin(y))])

    if not geobipy_kwargs['width'] is None:
        assert geobipy_kwargs['width'] > 0.0, ValueError("width must be positive")
        my[1] = geobipy_kwargs['width']

    v = masked_invalid(values)
    if (geobipy_kwargs['log']):
        v,logLabel = utilities._log(v, geobipy_kwargs['log'])

    # Append with null values to correctly use pcolormesh
    # v = concatenate([atleast_2d(hstack([asarray(v),0])), atleast_2d(zeros(v.size+1))], axis=0)
    v = atleast_2d(v)

    if color_kwargs['equalize']:
        v, dummy = utilities.histogramEqualize(v, nBins=color_kwargs['nBins'])

    X, Y = meshgrid(mx, my)

    if geobipy_kwargs['transpose']:
        X, Y = Y, X

    pm = ax.pcolormesh(X, Y, v, color=geobipy_kwargs.get('color'), **kwargs)

    ax.set_aspect('equal')

    if geobipy_kwargs['flip']:
        ax.invert_yaxis() if geobipy_kwargs['transpose'] else ax.invert_xaxis()

    if geobipy_kwargs['xlabel'] != False:
        geobipy_kwargs['xlabel'] = utilities.getNameUnits(y)

    ax.set_xscale(geobipy_kwargs['xscale'])
    ax.set_yscale(geobipy_kwargs['yscale'])

    if geobipy_kwargs['transpose']:
        xl = ylabel
        # ax.set_ylabel(utilities.getNameUnits(y))
        ax.get_xaxis().set_ticks([])
    else:
        xl = xlabel
        # ax.set_xlabel(utilities.getNameUnits(y))
        ax.get_yaxis().set_ticks([])

    xl(ax, geobipy_kwargs['xlabel'], wrap=geobipy_kwargs['wrap_xlabel'])

    if geobipy_kwargs['xlim'] is not None:
        ax.set_xlim(geobipy_kwargs['xlim'])
    if geobipy_kwargs['ylim'] is not None:
        ax.set_ylim(geobipy_kwargs['ylim'])

    if (color_kwargs['colorbar']):
        if (color_kwargs['equalize']):
            cbar = plt.colorbar(pm, extend='both')
        else:
            cbar = plt.colorbar(pm)

        if color_kwargs['clabel'] is None:
            if (geobipy_kwargs['log']):
                clabel(cbar, logLabel + utilities.getNameUnits(values), wrap=color_kwargs['wrap_clabel'])
            else:
                clabel(cbar, utilities.getNameUnits(values), wrap=color_kwargs['wrap_clabel'])
        else:
            clabel(cbar, color_kwargs['clabel'], wrap=color_kwargs['wrap_clabel'])

    if size(color_kwargs['alpha']) > 1:
        setAlphaPerPcolormeshPixel(pm, color_kwargs['alpha'])

    return ax

def hlines(*args, **kwargs):
    """Plot y against x

    If x and y are StatArrays, the axes are automatically labelled.

    Parameters
    ----------
    x : array_like or StatArray
        The abcissa
    y : array_like or StatArray
        The ordinate, can be upto 2 dimensions.

    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'.
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    labels : bool, optional
        Plot the labels? Default is True.

    Returns
    -------
    ax
        matplotlib.Axes

    See Also
    --------
        matplotlib.pyplot.plot : For additional keyword arguments you may use.

    """
    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    ax = geobipy_kwargs['ax']

    kwargs['linestyles'] = kwargs.pop('linestyle', 'solid')

    if 'linecolor' in kwargs:
        kwargs['color'] = kwargs.pop('linecolor')

    return ax.hlines(*args, **kwargs)

def vlines(*args, **kwargs):
    """Plot y against x

    If x and y are StatArrays, the axes are automatically labelled.

    Parameters
    ----------
    x : array_like or StatArray
        The abcissa
    y : array_like or StatArray
        The ordinate, can be upto 2 dimensions.

    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'.
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    labels : bool, optional
        Plot the labels? Default is True.

    Returns
    -------
    ax
        matplotlib.Axes

    See Also
    --------
        matplotlib.pyplot.plot : For additional keyword arguments you may use.

    """
    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    ax = geobipy_kwargs['ax']

    if 'linecolor' in kwargs:
        kwargs['color'] = kwargs.pop('linecolor')

    return ax.vlines(*args, **kwargs)

def plot(x, y, **kwargs):
    """Plot y against x

    If x and y are StatArrays, the axes are automatically labelled.

    Parameters
    ----------
    x : array_like or StatArray
        The abcissa
    y : array_like or StatArray
        The ordinate, can be upto 2 dimensions.

    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'.
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    labels : bool, optional
        Plot the labels? Default is True.

    Returns
    -------
    ax
        matplotlib.Axes

    See Also
    --------
        matplotlib.pyplot.plot : For additional keyword arguments you may use.

    """
    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    ax = geobipy_kwargs['ax']

    if geobipy_kwargs['transpose']:
        y, x = x, y

    if geobipy_kwargs['reciprocateX']:
        x = 1.0 / x

    if geobipy_kwargs['xlabel'] != False:
        geobipy_kwargs['xlabel'] = utilities.getNameUnits(x)

    if geobipy_kwargs['ylabel'] != False:
        geobipy_kwargs['ylabel'] = utilities.getNameUnits(y)

    tmp = y
    if (geobipy_kwargs['log']):
        tmp, logLabel = utilities._log(y, geobipy_kwargs['log'])
        geobipy_kwargs['ylabel'] = logLabel + geobipy_kwargs['ylabel']

    pretty(ax)

    ax.plot(x, y, alpha=color_kwargs['alpha'], **kwargs)

    if geobipy_kwargs['xlim'] is not None:
        ax.set_xlim(geobipy_kwargs['xlim'])

    if geobipy_kwargs['ylim'] is not None:
        ax.set_ylim(geobipy_kwargs['ylim'])

    ax.set_xscale(geobipy_kwargs['xscale'])
    ax.set_yscale(geobipy_kwargs['yscale'])

    xlabel(ax, geobipy_kwargs['xlabel'], wrap=geobipy_kwargs['wrap_xlabel'])
    ylabel(ax, geobipy_kwargs['ylabel'], wrap=geobipy_kwargs['wrap_ylabel'])

    ax.margins(0.1, 0.1)

    if geobipy_kwargs['flipX']:
        ax.invert_xaxis()

    if geobipy_kwargs['flipY']:
        ax.invert_yaxis()

    if not geobipy_kwargs['xscale'] == 'log':
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 2))

    if not geobipy_kwargs['yscale'] == 'log':
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 2))

    return ax


def scatter2D(x, c, y=None, i=None, *args, **kwargs):
    """Create a 2D scatter plot.

    Create a 2D scatter plot, if the y values are not given, the colours are used instead.
    If the arrays x, y, and c are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. markersize etc.

    Parameters
    ----------
    x : 1D array_like or StatArray
        Horizontal locations of the points to plot
    c : 1D array_like or StatArray
        Colour values of the points
    y : 1D array_like or StatArray, optional
        Vertical locations of the points to plot, if y = None, then y = c.
    i : sequence of ints or numpy.slice, optional
        Plot a geobipy_kwargs of x, y, c, using the indices in i.

    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to base 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    equalize : bool, optional
        Equalize the histogram of the colourmap so that all colours have an equal amount.
    nbins : int, optional
        Number of bins to use for histogram equalization.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'
    flipX : bool, optional
            Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    noColorbar : bool, optional
        Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.scatter : For additional keyword arguments you may use.

    """

    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    if (i is None):
        i = s_[:]

    standardColour = isinstance(c, (str, tuple))

    xt = x[i]

    iNaN = s_[:]
    if not standardColour:
        c = c[i]
        iNaN = where(~isnan(c))[0]
        c = c[iNaN]
        xt = xt[iNaN]

    # Did the user ask for a log colour plot?
    if (geobipy_kwargs['log'] and not standardColour):
        c, logLabel = utilities._log(c, geobipy_kwargs['log'])

    # Equalize the colours?
    if color_kwargs['equalize'] and not standardColour:
      tmp, dummy = utilities.histogramEqualize(c, nBins=color_kwargs['nBins'])
      if tmp is not None:
          c = tmp

    # Get the yAxis values
    if (y is None):
        assert not standardColour, Exception("Please specify either the y coordinates or a colour array")
        yt = c
    else:
        yt = y[i]
        yt = yt[iNaN]

    ax = geobipy_kwargs['ax']
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    pretty(ax)

    f = plt.scatter(xt, yt, c = c, cmap=color_kwargs['cmap'], **kwargs)


    yl = utilities.getNameUnits(yt)

    cbar = None
    if (color_kwargs['colorbar'] and not standardColour):
        if (color_kwargs['equalize']):
            cbar = plt.colorbar(f, extend='both', cax=color_kwargs['cax'])
        else:
            cbar = plt.colorbar(f, cax=color_kwargs['cax'])

        if color_kwargs['clabel'] is None:
            if (geobipy_kwargs['log']):
                clabel(cbar, logLabel + utilities.getNameUnits(c), wrap=color_kwargs['wrap_clabel'])
                if y is None:
                    yl = logLabel + yl
            else:
                clabel(cbar, utilities.getNameUnits(c), wrap=color_kwargs['wrap_clabel'])
        else:
            cl = color_kwargs['clabel'] if isinstance(color_kwargs['clabel'], str) else utilities.getNameUnits(c)
            clabel(cbar, cl, wrap=color_kwargs['wrap_clabel'])

    if ('s' in kwargs and not geobipy_kwargs['legend_size'] is None):
        assert (not isinstance(geobipy_kwargs['legend_size'], bool)), TypeError('sizeLegend must have type int, or array_like')
        if (isinstance(geobipy_kwargs['legend_size'], int)):
            tmp0 = nanmin(kwargs['s'])
            tmp1 = nanmax(kwargs['s'])
            a = linspace(tmp0, tmp1, geobipy_kwargs['legend_size'])
        else:
            a = geobipy_kwargs['legend_size']
        sizeLegend(kwargs['s'], a)

    if geobipy_kwargs['flipX']:
        ax.invert_xaxis()
    if geobipy_kwargs['flipY']:
        ax.invert_yaxis()

    ax.set_xscale(geobipy_kwargs['xscale'])
    ax.set_yscale(geobipy_kwargs['yscale'])

    ax.set_xlabel(utilities.getNameUnits(x, 'x'))
    ax.set_ylabel(yl)
    plt.margins(0.1, 0.1)
    plt.grid(True)
    return ax, f, cbar


def setAlphaPerPcolormeshPixel(pcmesh, alphaArray):
    """Set the opacity of each pixel in a pcolormesh

    Parameters
    ----------
    pcmesh : matplotlib.collections.QuadMesh
        pcolormesh object
    alphaArray : array_like
        Values per pixel each between 0 and 1.

    """
    plt.savefig('tmp.png')
    for i, j in zip(pcmesh.get_facecolors(), alphaArray.flatten()):
        if i[3] > 0.0:
            i[3] = j  # Set the alpha value of the RGBA tuple using a

    return pcmesh


def sizeLegend(values, intervals=None, **kwargs):
    """Add a legend to a plot if the point sizes have been specified.

    If values is an StatArray, labels are generated automatically.

    Parameters
    ----------
    values : array_like or StatArray
        The array that was used as the size (s=) in a scatter function.
    intervals : array_like, optional
        The legend will have items at each value in intervals.
    **kwargs : dict
        kwargs are applied to plt.legend.

    """

    # Scatter parameters
    a = kwargs.pop('alpha', 0.3)
    kwargs.pop('color', None)
    c = kwargs.pop('c', 'k')
    # Legend paramters
    sp = kwargs.pop('scatterpoints', 1)
    f = kwargs.pop('frameon', False)
    ls = kwargs.pop('labelspacing', 1)

    for x in intervals:
        plt.scatter([], [], c=c, alpha=a, s=x, label=str(x) + ' ' + utilities.getUnits(values))
    plt.legend(scatterpoints=sp, frameon=f, labelspacing=ls, title=utilities.getName(values), **kwargs)


def stackplot2D(x, y, labels=[], colors=tatarize, **kwargs):
    """Plot a 2D array with column elements stacked on top of each other.

    Parameters
    ----------
    x : array_like or StatArray
        The abcissa.
    y : array_like or StatArray, 2D
        The cumulative sum along the columns is taken and stacked on top of each other.

    Other Parameters
    ----------------
    labels : list of str, optional
        The labels to assign to each column.
    colors : matplotlib.colors.LinearSegmentedColormap or list of colours
        The colour used for each column.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
        matplotlib.pyplot.scatterplot : For additional keyword arguments you may use.

    """

    xscale = kwargs.pop('xscale','linear')
    yscale = kwargs.pop('yscale','linear')

    ax = kwargs.pop('ax', plt.gca())

    pretty(ax)

    ax.stackplot(x, y, labels=labels, colors=colors, **kwargs)

    if (not len(labels)==0):
        ax.legend()

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(utilities.getNameUnits(x))
    ax.set_ylabel(utilities.getNameUnits(y))
    ax.margins(0.1, 0.1)

    return ax


def step(x, y, **kwargs):
    """Plots y against x as a piecewise constant (step like) function.

    Parameters
    ----------
    x : array_like

    y : array_like

    flipY : bool, optional
        Flip the y axis
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'
    yscale : str, optional
        Scale the y axis? e.g. yscale = 'linear' or 'log'
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    noLabels : bool, optional
        Do not plot the labels

    """
    # Must create a new parameter, so that the last layer is plotted

    geobipy_kwargs, kwargs = filter_plotting_kwargs(kwargs)
    color_kwargs, kwargs = filter_color_kwargs(kwargs)

    color_kwargs['color'] = color_kwargs.pop('color', wellSeparated[3])

    ax = geobipy_kwargs.pop('ax', None)

    pretty(ax)

    if geobipy_kwargs['transpose']:
        x, y = y, x

    x, _ = utilities._log(x, geobipy_kwargs['logX']); y, _ = utilities._log(y, geobipy_kwargs['logY'])

    plt.sca(ax)
    stp = plt.step(x=x, y=y, **kwargs)

    if geobipy_kwargs['flipX']:
        ax.invert_xaxis()

    if geobipy_kwargs['flipY']:
        ax.invert_yaxis()

    if (geobipy_kwargs['labels']):
        ax.set_xlabel(utilities.getNameUnits(x))
        ax.set_ylabel(utilities.getNameUnits(y))

    ax.set_xscale(geobipy_kwargs['xscale'])
    ax.set_yscale(geobipy_kwargs['yscale'])

    return ax, stp


def pause(interval):
    """Custom pause command to override matplotlib.pyplot.pause
    which keeps the figure on top of all others when using interactve mode.

    Parameters
    ----------
    interval : float
        Pause for *interval* seconds.

    """
    from matplotlib._pylab_helpers import Gcf
    from matplotlib import backends

    backend = plt.rcParams['backend']

    if backend in backends.backend_registry.list_builtin(backends.BackendFilter.INTERACTIVE):
        figManager = Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def xlabel(ax, label, length=20, wrap=False, **kwargs):
    if label is False:
        return
    if wrap:
        label = utilities.wrap_string(label, length)
    ax.set_xlabel(label, **kwargs)

def ylabel(ax, label, length=20, wrap=False, **kwargs):
    if label is False:
        return
    if wrap:
        label = utilities.wrap_string(label, length)
    ax.set_ylabel(label, **kwargs)