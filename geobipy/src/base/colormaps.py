from matplotlib import colormaps
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from numpy import linspace, ones

def make_colourmap(colours, colourmap_name):
    """Generate a Linear Segmented colourmap

    Generates a colourmap from the sequence given and registers the colourmap with matplotlib.

    Parameters
    ----------
    colours : array of hex colours.
        e.g. ['#000000','#00fcfd',...]
    colourmap_name : str
        Name of the colourmap.

    Returns
    -------
    out
        matplotlib.colors.LinearSegmentedColormap.

    """
    if colourmap_name in colormaps:
        return

    nl = len(colours)
    dl = 1.0 / (len(colours))
    l = []
    for i, item in enumerate(colours):
        l.append(mcolors.hex2color(item))
        if (i < nl - 1):
            l.append((i + 1) * dl)
    l = [(0.0,) * 3, 0.0] + list(l) + [1.0, (1.0,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(l):
        if isinstance(item, float):
            r1, g1, b1 = l[i - 1]
            r2, g2, b2 = l[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    myMap = mcolors.LinearSegmentedColormap(colourmap_name, cdict, 256)
    colormaps.register(name=colourmap_name, cmap=myMap)
    return myMap

def white_to_colour(rgba, N=256):
    rgba = mcolors.to_rgba(rgba)
    vals = ones((N, 4))
    vals[:, 0] = linspace(1, rgba[0], N)
    vals[:, 1] = linspace(1, rgba[1], N)
    vals[:, 2] = linspace(1, rgba[2], N)
    return ListedColormap(vals)

# Define our own colour maps in hex. Gets better range and nicer visuals.
wellSeparated = [
"#3F5D7D",'#881d67','#2e8bac','#ffcf4d','#1d3915',
'#1a8bff','#00fcfd','#0f061f','#fa249d','#00198f','#c7fe1c']

make_colourmap(wellSeparated, 'wellseparated')

tatarize = [
"#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
"#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]
make_colourmap(tatarize, 'tatarize')

armytage = [
"#02a53f","#b97d87","#c1d4be","#c0bcd6","#bb8477","#8e3b06","#4ae36f","#e19585",
"#e3bbb5","#b9e6af","#e0917b","#6ad33f","#3811c6","#93d58d","#c6dec7","#ead3c6",
"#f0b98d","#08ef97","#c00fcf","#9cded6","#ead5e7","#e1ebf3","#e1c4f6","#9cd4f7"]