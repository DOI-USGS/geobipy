"""
.. module:: ManyResults
   :platform: Unix, Windows
   :synopsis: Class to handle the many HDF5 files. Each file contains the results of a separate inversion of an EM data point.

.. moduleauthor:: Leon Foks


"""

from ..base import Error as Err
import numpy as np
from ..base.myObject import myObject
from ..base.HDF import hdfRead
from os.path import join,split
from os import listdir
import h5py


class ManyResults(myObject):
    """ Class to define many results from EMinv1D_MCMC """

    def __init__(self, root='.'):
        self.root = root
        self.fileList = None

    def getLineFiles(self, root='.'):
        """ Gets a list of all the hdf5 files in the current directory """
        self.fileList = []
        for file in [f for f in listdir(root) if f.endswith('.h5')]:
            self.fileList.append(join(root,file))
        if (len(self.fileList) == 0):
            Err.Emsg('Could not find .h5 files in current directory')
        self.nPoints = len(self.fileList)

    def setFileList(self, i0=None, i1=None, base=''):
        """ Sets the list of HDF file names
        If i0 is an integer, i1 must be present. The file names are the integer numbers
        If i0 is a list, the file names remain as that list, each name is appended to the base directory
        """
        self.fileList = []
        if isinstance(i0, list):
            for item in i0:
                self.fileList.append(join(self.root, str(item)) + ".h5")
        elif isinstance(i0, (int, np.int)):
            if i1 is None:
                Err.Emsg(__name__ + ':if i0 is an integer, so must i1')
            for i in np.arange(i0, i1 + 1):
                self.fileList.append(join(self.root,base,str(i)) + '.h5')
        # Get the First identifier in each file
        self.grpName=[]
        for file in self.fileList:
            with h5py.File(file,'r') as f:
                self.grpName.append(list(f.keys)[0])

    def checkFileList(self):
        """ Check that files have been set """
        if (self.fileList is None):
            Err.Emsg(
                'Please set the file list in order to read attributes\nUse: self.setFileList()')

    def groupResults2Line(self,sysPath = ''):
        """ Groups the individual HDF5 result files for each data point into a single line results file """
        self.checkFileList()
        with h5py.File(join(split(self.directory)[0],self.line+'.h5'),'w') as f:
            for file in self.fileList:
                print('Processing ID: ',file)
                R=Results()
                R=R.read(file,sysPath)
                R.toHdf(f,split(file)[1])

    def getAttribute(self, attribute, fileList=None):
        """ Gets an attribute from the EMinv1D MCMC results """
        if (fileList is None):
            fileList = self.fileList
        if (attribute is None):
            Err.Emsg(
                "Please specify an attribute: \n",
                self.possibleAttributes)
        low = attribute.lower()

        if (low == 'iteration #'):
            return hdfRead.read_wKey(fileList, 'i')
        if (low == '# of markov chains'):
            return hdfRead.read_wKey(fileList, 'nmc')
        if (low == 'burned in'):
            return hdfRead.read_wKey(fileList, 'burnedin')
        if (low == 'burn in #'):
            return hdfRead.read_wKey(fileList, 'iburn')
        if (low == 'data multiplier'):
            return hdfRead.read_wKey(fileList, 'multiplier')
        if (low == 'layer histogram'):
            return hdfRead.read_wKey(fileList, 'khist')
        if (low == 'elevation histogram'):
            return hdfRead.read_wKey(fileList, 'dzhist')
        if (low == 'layer depth histogram'):
            return hdfRead.read_wKey(fileList, 'mzhist')
        if (low == 'best data'):
            return hdfRead.read_wKey(fileList, 'bestd')
        if (low == 'x'):
            return hdfRead.read_wKey(fileList, 'bestd/y')
        if (low == 'y'):
            return hdfRead.read_wKey(fileList, 'bestd/x')
        if (low == 'z'):
            return hdfRead.read_wKey(fileList, 'bestd/z')
        if (low == 'elevation'):
            return hdfRead.read_wKey(fileList, 'bestd/elevation')
        if (low == '# of systems'):
            return hdfRead.read_wKey(fileList, 'nsystems')
        if (low == 'relative error'):
            return hdfRead.read_wKey(fileList, 'bestd/relerr')
        if (low == 'additive error'):
            return hdfRead.read_wKey(fileList, 'bestd/adderr')
        if (low == 'best model'):
            return hdfRead.read_wKey(fileList, 'bestmodel')
        if (low == '# layers'):
            return hdfRead.read_wKey(fileList, 'bestmodel/ncells')
        if (low == 'current data'):
            return hdfRead.read_wKey(fileList, 'currentd')
        if (low == 'hit map'):
            return hdfRead.read_wKey(fileList, 'hitmap')
        if (low == 'data misfit'):
            return hdfRead.read_wKey(fileList, 'phids')
        if (low == 'relative error histogram'):
            nSys = hdfRead.read_wKey(fileList[0], 'nsystems')[0]
            listy = [[] for i in range(nSys)]
            for i in range(nSys):
                listy[i].append(hdfRead.read_wKey(fileList, 'relerr' + str(i)))
            return listy
        if (low == 'additive error histogram'):
            nSys = hdfRead.read_wKey(fileList[0], 'nsystems')[0]
            listy = [[] for i in range(nSys)]
            for i in range(nSys):
                listy[i].append(hdfRead.read_wKey(fileList, 'adderr' + str(i)))
            return listy
        if (low == 'inversion time'):
            return hdfRead.read_wKey(fileList, 'invtime')
        if (low == 'saving time'):
            return hdfRead.read_wKey(fileList, 'savetime')

        Err.Emsg("Invalid attribute: \n", self.possibleAttributes)

    def possibleAttributes(self):
        print("Possible Attribute options to read in \n" +
              "iteration # \n" +
              "# of markov chains \n" +
              "burned in\n" +
              "burn in # \n" +
              "data multiplier \n" +
              "layer histogram \n" +
              "elevation histogram \n" +
              "layer depth histogram \n" +
              "best data \n" +
              "x\n" +
              "y\n" +
              "z\n" +
              "elevation\n" +
              "# of systems\n" +
              "relative error\n" +
              "best model \n" +
              "# layers \n" +
              "current data \n" +
              "hit map \n" +
              "data misfit \n" +
              "relative error histogram\n" +
              "additive error histogram\n" +
              "inversion time\n" +
              "saving time\n")


if __name__ == '__main__':

    Results = ManyResults('Results/10010.0/')
    Results.setFileList([9715.6, 10065.4])
    results = hdfRead.read_wKey(Results.fileList, 'EMinv1D_MCMC_Results')
    results0 = results[0]
    results0.plot(Results.fileList[0])
    results1 = results[1]
    results1.plot(Results.fileList[1])
