import h5py
import numpy as np
from geobipy.src.base import plotting as cP
from geobipy.src.inversion.LineResults import LineResults
from geobipy.src.inversion.DataSetResults import DataSetResults
import matplotlib.pyplot as plt
from os.path import join
import os
import matplotlib as mpl
import argparse

Parser = argparse.ArgumentParser(description="Plotting results for individual data points.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
Parser.add_argument('datadir', default=None, help='Directory of the data')
Parser.add_argument('h5file', default=None, help='Specific h5 file to process.')
Parser.add_argument('--outputdir', default=".", help='Output directory for the images.')
Parser.add_argument('--points', nargs='+', type=int, default=None, help='Plot the results for these data point indices. If neither points or fiducials are specified, all points are plotted.')
Parser.add_argument('--fiducials', nargs='+', type=float, default=None, help='Plot the results for these data point fiducials. If neither points or fiducials are specified, all points are plotted.')
Parser.add_argument('--save', dest='save', default=True, help='Save images to png')
Parser.add_argument('--show', dest='show', default=False, help='Show images to screen')
Parser.add_argument('--dpi', dest='dpi', type=int, default=300, help='DPI of the saved images')
Parser.add_argument('--size', nargs=2, dest='size', default=None, help="Size of the figures '--size dx dy' in inches")

# Parse the command line arguments
args = Parser.parse_args()

# Get the options from the args
show = args.show
save = args.save

h5FilePath = args.h5file
h5File = os.path.split(h5FilePath)[-1]

# Set a default figure size
if args.size is None:
    figSize = (14, 10)
else:
    figSize = (int(args.size[0]), int(args.size[1]))

mpl.rcParams['figure.figsize'] = figSize[0], figSize[1]

dpi = args.dpi

# Open up the HDF5 file
LR = LineResults(args.h5file, systemFilepath=args.datadir)

fids = None
if not args.points is None:
    # points = np.asarray([np.int(x) for x in args.points])
    # print(points)
    fids = LR.iDs[args.points]

if not args.fiducials is None:
    # fiducials = np.asarray([np.float64(x) for x in args.fiducials])
    if fids is None:
        fids = args.fiducials
    else:
        fids = np.hstack([fids, args.fiducials])

# If neither indices or fiducials are specified, plot them all
if args.points is None and args.fiducials is None:
    fids = LR.iDs

for fid in fids:
    plt.figure(0, figsize=figSize, dpi=dpi)
    plt.clf()
    R = LR.getResults(fid = fid)
    R.initFigure(forcePlot=True)
    R.plot(forcePlot=True)
    outputFile = "{}//{}_dataPoint_{}.png".format(args.outputdir, h5File, fid)
    if save: plt.savefig(outputFile, dpi=dpi)
    if show: plt.show()

LR.close()
