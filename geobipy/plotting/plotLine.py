# -*- coding: utf-8 -*-

import h5py
import numpy as np
from geobipy.src.base import plotting as cP
from geobipy.src.inversion.LineResults import LineResults
from geobipy.src.inversion.DataSetResults import DataSetResults
import matplotlib.pyplot as plt
from os.path import join
import matplotlib as mpl
import argparse
import os

### Check a single line results file

Parser = argparse.ArgumentParser(description="Plotting line results",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
Parser.add_argument('datadir', default=None, help='Directory of the data')
Parser.add_argument('h5dir', default='.', help='Directory of HDF5 files')
Parser.add_argument('--outdir', default='.', help='Directory to place the images')
Parser.add_argument('--files', nargs='+', default=None, help='Specific h5 files in h5dir to process. Space separated')
Parser.add_argument('--save', dest='save', default=True, help='Save images to png')
Parser.add_argument('--show', dest='show', default=False, help='Show images to screen')
Parser.add_argument('--dpi', dest='dpi', type=int, default=300, help='DPI of the saved images')
Parser.add_argument('--size', nargs=2, dest='size', default=None, help='Size of the figures --size dx dy in inches')
Parser.add_argument('--xaxis', dest='xaxis', default='x', help='[x, y, r2d, r3d, or index] Plot the lines against this co-ordinate')

args = Parser.parse_args()

show = args.show
save = args.save
xaxis = args.xaxis

files = args.files
if files is None:
    files = [f for f in os.listdir(args.h5dir) if f.endswith('.h5')]

if args.size is None:
    figSize = (20,4)
else:
    figSize = (int(args.size[0]), int(args.size[1]))

mpl.rcParams['figure.figsize'] = figSize[0], figSize[1]

dpi = args.dpi


for file in files:

    fName = join(args.h5dir, file)
    file = os.path.join(args.outdir, file)

    LR = LineResults(fName, systemFilepath=args.datadir)

    p = 0

    points = [100, 250, 370, 550]

    p += 1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotXsection(bestModel=False, log=10, invertPar=True, vmin=1.0, vmax=np.log10(500.0), cmap='jet', xAxis=xaxis)
    LR.plotDataElevation(linewidth=0.1, xAxis=xaxis)
    #LR.plotDoi(linewidth=0.1, alpha=0.6, percent=40)
    if show: plt.show()
    if save: plt.savefig(file+'_meanModel.png',dpi=dpi)

    # p += 1
    # plt.figure(p, figsize=figSize, dpi=dpi)
    # LR.plotBestDataChannel(channel=np.s_[:], yscale='linear', linewidth=0.5, xAxis=xaxis)
    # if show: plt.show()
    # if save: plt.savefig(file+'_predictedData.png',dpi=dpi)

    # p += 1
    # plt.figure(p, figsize=figSize, dpi=dpi)
    # LR.plotObservedDataChannel(channel=np.s_[:], yscale='linear', linewidth=0.5, xAxis=xaxis)
    # if show: plt.show()
    # if save: plt.savefig(file+'_observedData.png',dpi=dpi)

    # p += 1
    # plt.figure(p, figsize=figSize, dpi=dpi)
    # LR.plotAllBestData(log=10)
    # if show: plt.show()
    # if save: plt.savefig(file+'_allBestData.png', dpi=dpi)

    p+=1
    plt.figure(p,figsize=figSize, dpi=dpi)
    LR.plotXsection(bestModel=True, log=10, invertPar=True, vmin=1.0, vmax=np.log10(500.0), cmap='jet', xAxis=xaxis)
    LR.plotDataElevation(linewidth=0.1, xAxis=xaxis)
    #LR.plotHighlightedObservationLocations(iDs=LR.iDs[points])
    if show: plt.show()
    if save: plt.savefig(file+'_bestModel.png',dpi=dpi)


    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotKlayers()
    if show: plt.show()
    if save: plt.savefig(file+'_kLayers.png',dpi=dpi)

    # p+=1
    # plt.figure(p, figsize=figSize, dpi=dpi)
    # LR.plotSuccessFail()
    # if show: plt.show()
    # if save: plt.savefig(file+'_successFail.png',dpi=dpi)

    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotAdditiveError(linestyle='none')
    if show: plt.show()
    if save: plt.savefig(file+'_additive.png',dpi=dpi)


    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotRelativeError(yscale='log', linestyle='none')
    if show: plt.show()
    if save: plt.savefig(file+'_relative.png',dpi=dpi)

#    p+=1
#    plt.figure(p, figsize=figSize, dpi=dpi)
#    c=0
#    LR.plotTotalError(channel=c, yscale='log', linestyle='none')
#    if show: plt.show()
#    if save: plt.savefig(file+'_total_channel_'+str(c)+'.png',dpi=dpi)
#
#    p+=1
#    c=5
#    plt.figure(p, figsize=figSize, dpi=dpi)
#    LR.plotTotalError(channel=c, yscale='log', linestyle='none')
#    if show: plt.show()
#    if save: plt.savefig(file+'_total_channel_'+str(c)+'.png',dpi=dpi)
#
#    p+=1
#    plt.figure(p, figsize=figSize, dpi=dpi)
#    c=9
#    LR.plotTotalError(channel=c, yscale='log', linestyle='none')
#    if show: plt.show()
#    if save: plt.savefig(file+'_total_channel_'+str(c)+'.png',dpi=dpi)


    p+=1
    plt.figure(p)
    LR.histogram(nBins=100, log=10)
    if show: plt.show()
    if save: plt.savefig(file+'_histogram.png',dpi=dpi)

    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotInterfaces(cmap='gray_r', useVariance=False, xAxis=xaxis)
    LR.plotElevation(linewidth=0.1, xAxis=xaxis)
    LR.plotDataElevation(linewidth=0.1, xAxis=xaxis)
    if show: plt.show()
    if save: plt.savefig(file+'_interfaces.png',dpi=dpi)

    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotOpacity(cmap='gray_r', xAxis=xaxis)
    LR.plotElevation(linewidth=0.1, xAxis=xaxis)
    LR.plotDataElevation(linewidth=0.1, xAxis=xaxis)
    if show: plt.show()
    if save: plt.savefig(file+'_opacity.png', dpi=dpi)

    # p+=1
    # plt.figure(p, figsize=figSize, dpi=dpi)
    # LR.plotTransparancy(cmap='gray_r', xAxis=xaxis)
    # LR.plotElevation(linewidth=0.1, xAxis=xaxis)
    # LR.plotDataElevation(linewidth=0.1, xAxis=xaxis)
    # if show: plt.show()
    # if save: plt.savefig(file+'_transparancy.png', dpi=dpi)


    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotAdditiveErrorDistributions(system=0, cmap='gray_r', xAxis=xaxis)
    if show: plt.show()
    if save: plt.savefig(file+'_addErrHist.png',dpi=dpi)

    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotRelativeErrorDistributions(system=0, cmap='gray_r', xAxis=xaxis)
    if show: plt.show()
    if save: plt.savefig(file+'_relErrHist.png',dpi=dpi)

#    p+=1
#    plt.figure(p, figsize=figSize, dpi=dpi)
#    c=0
#    LR.plotTotalErrorDistributions(channel=c, nBins=100)
#    plt.title('Channel '+str(c))
#    if show: plt.show()
#    if save: plt.savefig(file+'_totErrHist_channel_'+str(c)+'.png',dpi=dpi)
#
#    p+=1
#    plt.figure(p, figsize=figSize, dpi=dpi)
#    c=5
#    LR.plotTotalErrorDistributions(channel=c, nBins=100)
#    plt.title('Channel '+str(c))
#    if show: plt.show()
#    if save: plt.savefig(file+'_totErrHist_channel_'+str(c)+'.png',dpi=dpi)
#
#    p+=1
#    plt.figure(p, figsize=figSize, dpi=dpi)
#    c=9
#    LR.plotTotalErrorDistributions(channel=c, nBins=100)
#    plt.title('Channel '+str(c))
#    if show: plt.show()
#    if save: plt.savefig(file+'_totErrHist_channel_'+str(c)+'.png',dpi=dpi)
#
#

    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotElevationDistributions(cmap='gray_r', xAxis=xaxis)
    if show: plt.show()
    if save: plt.savefig(file+'_dataElevationHist.png',dpi=dpi)

    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.plotKlayersDistributions(cmap='gray_r', xAxis=xaxis)
    if show: plt.show()
    if save: plt.savefig(file+'_kLayersHist.png',dpi=dpi)

    p+=1
    plt.figure(p, figsize=figSize, dpi=dpi)
    LR.crossplotErrors()
    if show: plt.show()
    if save: plt.savefig(file+'_crossplotErr.png',dpi=dpi)

    plt.close('all')



    LR.close()

