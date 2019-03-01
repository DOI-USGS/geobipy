import numpy as np
from mpi4py import MPI
from geobipy import StatArray
from geobipy import PointCloud3D
from geobipy import Point
from geobipy import Data
from geobipy import DataPoint
from geobipy import FdemData
from geobipy import TdemData
from geobipy import FdemSystem
from geobipy import TdemSystem
from geobipy import FdemDataPoint
from geobipy import TdemDataPoint
from geobipy.src.base import MPI as myMPI

world = MPI.COMM_WORLD

rank = world.rank
master = rank == 0
size = world.size

myMPI.helloWorld(world)

x = 1
# Set up array sizes for consistency and chunk lengths per core
N = x * size+1

starts, chunks = myMPI.loadBalance_shrinkingArrays(N, size)

myMPI.rankPrint(world, "start indices: {}".format(starts))
myMPI.rankPrint(world, "chunk sizes: {}".format(chunks))

### Test base geobipy.MPI routines

# data type
# dt = None
# if master:
#     x = np.full(rank+1, rank)
#     myMPI._isendDtype(x, dest=1, world=world)
# elif rank == 1:
#     dt = myMPI._irecvDtype(source=0, world=world)

# myMPI.orderedPrint(world, dt, 'datatype send and recv')

# if not master:
#     x = None

# dt = myMPI.bcastType(x, world)

# myMPI.orderedPrint(world, dt, 'datatype bcast')

# nDim = myMPI.Bcast_1int(np.size(x), world)

# myMPI.orderedPrint(world, nDim, 'integer bcast')

# x = np.full(rank+1, rank)

# # Single Integers
# i = rank
# j = None
# if rank == 0:
#     myMPI.Isend_1int(i, 1, world)
# elif rank == 1:
#     j = myMPI.Irecv_1int(0, world)

# if rank == 1:
#     myMPI.Isend_1int(i, 2, world)
# elif rank == 2:
#     j = myMPI.Irecv_1int(1, world)

# myMPI.orderedPrint(world, j, 'integer Send and Recv')

# # Send and Recv
# y = None
# if master:
#     myMPI.Isend(x, dest=1, world=world)
# elif rank == 1:
#     y = myMPI.Irecv(source=0, world=world)

# if rank == 1:
#     myMPI.Isend(x, 2, world)
# elif rank == 2:
#     y = myMPI.Irecv(1, world)

# myMPI.orderedPrint(world, y, 'Send and Recv')

# if master:
#     for i in range(1, size):
#         myMPI.Isend(x, dest=i, world=world)
# else:
#     y = myMPI.Irecv(0, world)

# myMPI.orderedPrint(world, y, 'Send and Recv - Master to Workers')


# # Send and Recv Left
# myMPI.IsendToLeft(x, world)
# z = myMPI.IrecvFromRight(world)

# myMPI.orderedPrint(world, z, "Send Left Recv Right")

# myMPI.IsendToRight(x, world)
# z = myMPI.IrecvFromLeft(world)

# myMPI.orderedPrint(world, z, "Send Right Recv Left")


# ### Test the StatArray routines
# x = StatArray((rank+1) * np.arange(N, dtype=np.float64), "name", "units")

# # Bcast
# y = x.Bcast(world)

# myMPI.orderedPrint(world, y, "StatArray.Bcast")

# # Scatterv

# y = None
# y = x.Scatterv(starts, chunks, world)

# myMPI.orderedPrint(world, y, "StatArray.Scatterv")

# # Send and Recv
# z = None
# if master:
#     x.Isend(1, world)
# elif world.rank == 1:
#     x.Isend(2, world)
#     z = StatArray(0).Irecv(0, world)
# elif world.rank == 2:
#     z = StatArray(0).Irecv(1, world) 

# myMPI.orderedPrint(world, z, "StatArray.Isend/Irecv")

# if master:
#     for i in range(1, size):
#         x.Isend(dest=i, world=world)
# else:
#     z = StatArray(0).Irecv(0, world)

# myMPI.orderedPrint(world, z, 'Send and Recv - Master to Workers')


# # Send and Recv Left
# x.IsendToLeft(world)
# z = StatArray(0).IrecvFromRight(world)

# myMPI.orderedPrint(world, z, "StatArray.IsendToLeft/IrecvFromRight")

# x.IsendToRight(world)
# z = StatArray(0).IrecvFromLeft(world)

# myMPI.orderedPrint(world, z, "StatArray.IsendToLeft/IrecvFromRight")


# ### Test the PointCloud3D
# if master:
#     y = np.arange(N) + 10.0
#     z = np.arange(N) + 100.0
#     pc = PointCloud3D(N, x, y, z)
# else:
#     pc = PointCloud3D(0)

# # Bcast
# pc1 = pc.Bcast(world)

# myMPI.orderedPrint(world, np.asarray([pc1.x, pc1.y, pc1.z]), "PointCloud3D.Bcast")

# # Scatterv
# pc1 = pc.Scatterv(starts, chunks, world)

# myMPI.orderedPrint(world, np.asarray([pc1.x, pc1.y, pc1.z]), "PointCloud3D.Scatterv")

# # Send and Recv a point
# y = None
# if master:
#     for i in range(1, size):
#         pc.getPoint(i).Isend(dest=i, world=world)
# else:
#     y = Point().Irecv(0, world)
    
# myMPI.orderedPrint(world, str(y), "Point.Isend/Irecv")


# ## Test the Data class
# if master:
#     y = np.arange(N) + 10.0
#     z = np.arange(N) + 100.0
#     ncps = np.asarray([2,2])
#     d = np.repeat(np.arange(N)[:, np.newaxis], 4, 1) + np.asarray([1000,2000,3000,4000])
#     s = np.full([N, 4], 5.0)
#     p = np.ones([N, 4])
#     data = Data(nPoints=N, nChannelsPerSystem=ncps, x=x, y=y, z=z, data=d, std=s, predictedData=p, channelNames=['t1', 't2', 't3', 't4'])
# else:
#     data = Data(0)

# # Bcast
# data1 = data.Bcast(world)

# myMPI.orderedPrint(world, np.asarray([data1.x, data1.y, data1.z]), "Data.Bcast")
# myMPI.orderedPrint(world, data1._data)
# myMPI.orderedPrint(world, data1._std)
# myMPI.orderedPrint(world, data1._predictedData)

# # Scatterv
# data1 = data.Scatterv(starts, chunks, world)

# myMPI.orderedPrint(world, np.asarray([data1.x, data1.y, data1.z]), "Data.Scatterv")
# myMPI.orderedPrint(world, data1._data)
# myMPI.orderedPrint(world, data1._std)
# myMPI.orderedPrint(world, data1._predictedData)

# # Send and Recv a point
# y = DataPoint()
# if master:
#     for i in range(1, size):
#         data.getDataPoint(i).Isend(dest=i, world=world)
# else:
#     y = DataPoint().Irecv(0, world)

# world.barrier()

# myMPI.orderedPrint(world, y.summary(True), "DataPoint.Isend/Irecv")

# ### Test Frequency Domain Data
dataPath = "..//geobipy//documentation//notebooks//supplementary//Data//" 
# if master:
#     fd = FdemData()
#     fd.read(dataPath+"Resolve2.txt", dataPath+"FdemSystem2.stm")
# else:
#     fd = FdemData()

# # Bcast
# fd1 = fd.Bcast(world)

# myMPI.orderedPrint(world, fd1.summary(True), "FdemData.Bcast")

# # Scatterv
# starts, chunks = myMPI.loadBalance_shrinkingArrays(fd1.nPoints, size)

# fd1 = fd.Scatterv(starts, chunks, world)

# myMPI.orderedPrint(world, np.asarray([fd1.x, fd1.y, fd1.z]), "FdemData.Scatterv")
# myMPI.orderedPrint(world, fd1._data)
# myMPI.orderedPrint(world, fd1._std)
# myMPI.orderedPrint(world, fd1._predictedData)

# Point by Point read in and send 
if master:
    fd = FdemData()
    fd._initLineByLineRead(dataPath+"Resolve2.txt", dataPath+"FdemSystem2.stm")

# # Send and recieve a single datapoint from the file.
# if master:
#     for i in range(1, size):
#         fdp = fd._readSingleDatapoint()
#         fdp.Isend(dest=i, world=world)
# else:
#     fdp = FdemDataPoint().Irecv(source=0, world=world)

# myMPI.orderedPrint(world, fdp.summary(True), "FdemData.Isend/Irecv")

# Testing pre-read in system classes
sysPath = [dataPath+"FdemSystem2.stm"]
systems = []
for s in sysPath:
    systems.append(FdemSystem(systemFilename=s))

# Send and recieve a single datapoint from the file.
if master:
    for i in range(1, size):
        fdp = fd._readSingleDatapoint()
        fdp.Isend(dest=i, world=world, systems=systems)
else:
    fdp = FdemDataPoint().Irecv(source=0, world=world, systems=systems)

myMPI.orderedPrint(world, fdp.summary(True), "FdemData.Isend/Irecv")


# ### Test Time Domain Data
# if master:
#     td = TdemData()
#     td.read([dataPath+"Skytem_High.txt", dataPath+"Skytem_Low.txt"], [dataPath+"SkytemHM-SLV.stm", dataPath+"SkytemLM-SLV.stm"])
# else:
#     td = TdemData()

# # Bcast
# td1 = td.Bcast(world)


# myMPI.orderedPrint(world, np.asarray([td1.x, td1.y, td1.z]), "TdemData.Bcast")
# myMPI.orderedPrint(world, td1._data[:, td1.iActive])
# myMPI.orderedPrint(world, td1._std[:, td1.iActive])
# myMPI.orderedPrint(world, td1._predictedData[:, td1.iActive])
   
# # Scatterv
# starts, chunks = myMPI.loadBalance_shrinkingArrays(td1.nPoints, size)

# td1 = td.Scatterv(starts, chunks, world)

# myMPI.orderedPrint(world, np.asarray([td1.x, td1.y, td1.z]), "TdemData.Scatterv")
# myMPI.orderedPrint(world, td1._data[:, td1.iActive])
# myMPI.orderedPrint(world, td1._std[:, td1.iActive])
# myMPI.orderedPrint(world, td1._predictedData[:, td1.iActive])

# Point by Point read in and send 
if master:
    td = TdemData()
    td._initLineByLineRead([dataPath+"Skytem_High.txt", dataPath+"Skytem_Low.txt"], [dataPath+"SkytemHM-SLV.stm", dataPath+"SkytemLM-SLV.stm"])

# # Send and recieve a single datapoint from the file.
# if master:
#     for i in range(1, size):
#         tdp = td._readSingleDatapoint()
#         tdp.Isend(dest=i, world=world)
# else:
#     tdp = TdemDataPoint().Irecv(source=0, world=world)

# myMPI.orderedPrint(world, tdp.summary(True), "FdemData.Isend/Irecv")

# Testing pre-read in system classes
sysPath = [dataPath+"SkytemHM-SLV.stm", dataPath+"SkytemLM-SLV.stm"]
systems = []
for s in sysPath:
    systems.append(TdemSystem(s))


# Send and recieve a single datapoint from the file.
if master:
    for i in range(1, size):
        tdp = td._readSingleDatapoint()
        tdp.Isend(dest=i, world=world, systems=systems)
else:
    tdp = TdemDataPoint().Irecv(source=0, world=world, systems=systems)

myMPI.orderedPrint(world, tdp.summary(True), "FdemData.Isend/Irecv")







