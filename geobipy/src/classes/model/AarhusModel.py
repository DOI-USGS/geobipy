import numpy as np
from ...base import fileIO as fio
from ..statistics.StatArray import StatArray
from ..mesh.TopoRectilinearMesh2D import TopoRectilinearMesh2D
from .Model import Model


class AarhusModel(Model):

    def __init__(self):
        """Only used to instantiate the class.

        Use self.read2D or self.read3D to fill members of the class.

        """
        self.mesh = None
        self.rho = None
        self.fid = None


    def pcolor(self, useDOI = True, **kwargs):

        if useDOI:
            alpha = np.ones(self.mesh.shape)
            cellId = self.mesh.z.cellIndex(self.doi)
            for i in range(self.mesh.x.nCells):
                alpha[cellId[i]:, i] = 0.0
            kwargs['alpha'] = alpha

        self.mesh.pcolor(self.rho, **kwargs)


    def plotDOI(self, xAxis='x', **kwargs):

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        (self.mesh.height.centres - self.doi).plot(x = xtmp, **kwargs)


    def plotElevation(self, **kwargs):
        self.mesh.plotHeight(**kwargs)


    def plotXY(self, **kwargs):
        self.mesh.plotXY(**kwargs)


    def readline_numbers(self, fileName):
        """Read in the line numbers from an inversion file.

        Parameters
        ----------
        fileName : str
            Path to the inversion file.

        """

        # Get the total number of points to pre-allocate memory.
        with open(fileName, 'r') as f:
            # Skip the top of the file until we get the column headers
            line = ''
            nHeader = 0
            while not "LINE" in line:
                line = f.readline()
                nHeader += 1

            header = line.split()[1:]

            # We now have the header line, so grab the column indices for what we need
            lineIndex = 0

            for i, head in enumerate(header):
                head = head.lower()
                if head == "line":
                    lineIndex = i
                    break

            tmp = []
            line = fio.getRealNumbersfromLine(f.readline())
            tmp.append(line[lineIndex])

            for line in f:
                l = fio.getRealNumbersfromLine(line)[lineIndex]
                if l != tmp[-1]:
                    tmp.append(l)

            return np.asarray(tmp)


    def read2D(self, fileName, line_number):
        """Read in an inversion file from the Aarhus software

        Parameters
        ----------
        fileName : str
            Path to the inversion file.
        index : int
            Index of the line to read in 0 to nLines.
        line_number : float
            The line number to read in.

        Returns
        -------
        self : TopoRectilinearMesh2D
            The mesh.
        values : geobipy.StatArray
            The values of the model.

        """

        # Get the total number of points to pre-allocate memory.
        nLines = fio.getNlines(fname=fileName)

        with open(fileName, 'r') as f:
            # Skip the top of the file until we get the column headers
            line = ''
            nLayers = 0
            nHeader = 0
            while not "LINE" in line:
                line = f.readline()
                nHeader += 1
                if "NUMLAYER" in line:
                    nLayers = np.int(f.readline().split('/')[-1])
                    nHeader += 1

            nPoints = nLines - nHeader

            header = line.split()[1:]

            # We now have the header line, so grab the column indices for what we need
            lineIndex = 0
            xIndex = 1
            yIndex = 2
            zIndex = 6
            fidIndex = 3
            rhoIndex = []
            topIndex = []
            doiIndex = None

            for i, head in enumerate(header):
                head = head.lower()
                if head == "line":
                    lineIndex = i
                elif head == "x":
                    xIndex = i
                elif head == "y":
                    yIndex = i
                elif head == "fid":
                    fidIndex = i
                elif head == "topo":
                    zIndex = i
                elif head == "doi_lower":
                    doiIndex = i

                if "rho_i" in head and not "std" in head:
                    rhoIndex.append(i)
                elif "dep_top" in head and not "std" in head:
                    topIndex.append(i)

            # Index arrays are set, pre-allocate memory
            rhoIndex = np.asarray(rhoIndex, dtype=np.int)
            topIndex = np.asarray(topIndex, dtype=np.int)

            x = StatArray(nPoints, 'Easting', 'm')
            y = StatArray(nPoints, 'Northing', 'm')
            z = StatArray(nPoints, 'Elevation', 'm')
            fid = StatArray(nPoints, 'Fiducial')
            doi = StatArray(nPoints, 'Depth of investigation', 'm')
            rho = np.zeros([nLayers, nPoints])
            depthEdges = StatArray(nLayers+1, 'Depth', 'm')

            # Skip the first data points that are not the line we need
            line = fio.getRealNumbersfromLine(f.readline())

            while line[lineIndex] != line_number:
                line = fio.getRealNumbersfromLine(f.readline())

            # Read in the data points for the requested line,
            # assumes the data points for the given line are contiguous.
            nPoints = 0
            first = True
            while line[lineIndex] == line_number:
                if first:
                     depthEdges[:-1] = line[topIndex]

                x[nPoints] = line[xIndex]
                y[nPoints] = line[yIndex]
                z[nPoints] = line[zIndex]
                fid[nPoints] = line[fidIndex]
                rho[:, nPoints] = line[rhoIndex]
                doi[nPoints] = line[doiIndex]


                nPoints += 1
                first = False
                line = fio.getRealNumbersfromLine(f.readline())

        # Assign the half space depth
        depthEdges[-1] = 1.5 * depthEdges[-2]


        self.mesh = TopoRectilinearMesh2D(x_centres=x[:nPoints], y_centres=y[:nPoints], z_edges=depthEdges, heightCentres=z[:nPoints])
        self.fid = StatArray(fid[:nPoints], 'Fiducial')
        self.rho = StatArray(rho[:, :nPoints], 'Resistivity', '$\Omega m$')
        self.doi = StatArray(doi[:nPoints], 'Depth of investigation', 'm')



    def read3D(self, fileName):

        NotImplementedError('yet')