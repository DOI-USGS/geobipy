import numpy as np
from mpi4py import MPI
from geobipy import StatArray
from geobipy import Point
from geobipy import Data
from geobipy import DataPoint
from geobipy import FdemData
from geobipy import TdemData
from geobipy import TempestData
from geobipy import FdemSystem
from geobipy import TdemSystem
from geobipy import FdemDataPoint
from geobipy import TdemDataPoint
from geobipy import Tempest_datapoint
from geobipy.src.base import MPI as myMPI

##############################################################################
# Init
##############################################################################
world = MPI.COMM_WORLD

rank = world.rank
master = rank == 0
size = world.size

assert size == 4, Exception("Please use 4 cores to test")

dataPath = "..//documentation_source//source//examples//supplementary//Data//"

x = 1
# Set up array sizes for consistency and chunk lengths per core
N = x * size+1

starts, chunks = myMPI.loadBalance1D_shrinkingArrays(N, size)
i0 = starts[rank]
i1 = i0 + chunks[rank]

##############################################################################
# Test base geobipy.MPI routines
##############################################################################

# data type
dt = None
if master:
    x = np.full(rank+1, rank)
    myMPI._isendDtype(x, dest=1, world=world)
elif rank == 1:
    dt = myMPI._irecvDtype(source=0, world=world)


if not master:
    x = None

dt = myMPI.bcastType(x, world)

assert isinstance(dt, type), Exception("Could not broadcast datatype. Rank {}".format(rank))

nDim = myMPI.Bcast_1int(np.size(x), world)
assert nDim == 1, Exception("Could not broadcast integer. Rank {}".format(rank))

x = np.full(rank+1, rank)

# Single Integers
i = rank
j = None
if rank == 0:
    myMPI.Isend_1int(i, 1, world)
elif rank == 1:
    j = myMPI.Irecv_1int(0, world)

if rank == 1:
    myMPI.Isend_1int(i, 2, world)
elif rank == 2:
    j = myMPI.Irecv_1int(1, world)

tst = [None, 0, 1, None]
assert j == tst[rank], Exception("a) Could not Isend/Irecv. Rank {}".format(rank))

# Send and Recv
y = None
if master:
    myMPI.Isend(x, dest=1, world=world)
elif rank == 1:
    y = myMPI.Irecv(source=0, world=world)

if rank == 1:
    myMPI.Isend(x, 2, world)
elif rank == 2:
    y = myMPI.Irecv(1, world)

tst = [None, 0, [1, 1], None]
assert np.all(y == tst[rank]), Exception("b) Could not Isend/Irecv. Rank {}".format(rank))

if master:
    for i in range(1, size):
        myMPI.Isend(x, dest=i, world=world)
else:
    y = myMPI.Irecv(0, world)

tst = [None, 0, 0, 0]
assert y == tst[rank], Exception("c) Could not Isend/Irecv. Rank {}".format(rank))

# Send and Recv Left
myMPI.IsendToLeft(x, world)
z = myMPI.IrecvFromRight(world)

tst = [[1, 1], [2, 2, 2], [3, 3, 3, 3], [0]]
assert np.all(z == tst[rank]), Exception("d) Could not Isend/Irecv Right to Left. Rank {}".format(rank))

myMPI.IsendToRight(x, world)
z = myMPI.IrecvFromLeft(world)

tst = [[3, 3, 3, 3], 0, [1, 1], [2, 2, 2]]
assert np.all(z == tst[rank]), Exception("e) Could not Isend/Irecv Left to Right. Rank {}".format(rank))

##############################################################################
# Test the StatArray routines
##############################################################################
x = StatArray((rank+1) * np.arange(N, dtype=np.float64), "name", "units")

# Bcast
y = x.Bcast(world)
assert np.all(y == [0.0, 1.0, 2.0, 3.0, 4.0]), Exception("Could not use StatArray.Bcast. Rank {}".format(rank))

# Scatterv
y = x.Scatterv(starts, chunks, world)

tst = [[0.0, 1.0], 2.0, 3.0, 4.0 ]
assert np.all(y == tst[rank]), Exception("Could not use StatArray.Scatterv. Rank {}".format(rank))

# Send and Recv
z = None
if master:
    x.Isend(1, world)
elif world.rank == 1:
    x.Isend(2, world)
    z = StatArray(0).Irecv(0, world)
elif world.rank == 2:
    z = StatArray(0).Irecv(1, world)

tst = [None, [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 4.0, 6.0, 8.0], None]
assert np.all(z == tst[rank]), Exception("Could not use StatArray.Isend/Irecv. Rank {}".format(rank))

if master:
    for i in range(1, size):
        x.Isend(dest=i, world=world)
else:
    z = StatArray(0).Irecv(0, world)

tst = [None, [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]]
assert np.all(z == tst[rank]), Exception("Could not use StatArray.Isend/Irecv. Rank {}".format(rank))


# Send and Recv Left
x.IsendToLeft(world)
z = StatArray(0).IrecvFromRight(world)

tst = [[0., 2., 4., 6., 8.], [ 0.,  3.,  6.,  9., 12.], [ 0.,  4.,  8., 12., 16.], [0., 1., 2., 3., 4.]]
assert np.all(z == tst[rank]), Exception("Could not use StatArray.IsendToLeft/IrecvFromRight. Rank {}".format(rank))

x.IsendToRight(world)
z = StatArray(0).IrecvFromLeft(world)

tst = [[ 0.,  4., 8., 12., 16.], [0., 1., 2., 3., 4.], [0., 2., 4., 6., 8.], [ 0.,  3.,  6.,  9., 12.]]
assert np.all(z == tst[rank]), Exception("Could not use StatArray.IsendToRight/IrecvFromLeft. Rank {}".format(rank))

xSave = np.arange(N)
ySave = np.arange(N) + 10.0
zSave = np.arange(N) + 100.0

dSave = np.repeat(np.arange(N)[:, np.newaxis], 4, 1) + np.asarray([1000,2000,3000,4000])
sSave = np.full([N, 4], 5.0)
pSave = np.ones([N, 4])

##############################################################################
# Test the Point
##############################################################################
if master:
    pc = Point(xSave, ySave, zSave)
else:
    pc = Point()

# Bcast
pc1 = pc.Bcast(world)
assert np.all(pc1.x == xSave) and np.all(pc1.y == ySave) and np.all(pc1.z == zSave), Exception("Could not use Point.Bcast. Rank {}".format(rank))

# Scatterv
pc1 = pc.Scatterv(starts, chunks, world)

assert np.all(pc1.x == xSave[i0:i1]) and np.all(pc1.y == ySave[i0:i1]) and np.all(pc1.z == zSave[i0:i1]), Exception("Could not use Point.Scatterv. Rank {}".format(rank))

##############################################################################
# Test the Data class
##############################################################################
if master:
    ncps = np.asarray([2,2])
    data = Data(channels_per_system=ncps, x=xSave, y=ySave, z=zSave, data=dSave, std=sSave, predictedData=pSave, channel_names=['t1', 't2', 't3', 't4'])
else:
    data = Data()

# Bcast
data1 = data.Bcast(world)

assert np.all(data1.x == xSave) and np.all(data1.y == ySave) and np.all(data1.z == zSave), Exception("Could not use Data.Bcast. Rank {}".format(rank))
assert np.all(data1.data == dSave) and np.all(data1.std == sSave) and np.all(data1.predictedData == pSave), Exception("Could not use Data.Bcast. Rank {}".format(rank))


# Scatterv
data1 = data.Scatterv(starts, chunks, world)

assert np.all(data1.x == xSave[i0:i1]) and np.all(data1.y == ySave[i0:i1]) and np.all(data1.z == zSave[i0:i1]), Exception("Could not use Data.Scatterv. Rank {}".format(rank))
assert np.all(data1.data == dSave[i0:i1, :]) and np.all(data1.std == sSave[i0:i1, :]) and np.all(data1.predictedData == pSave[i0:i1, :]), Exception("Could not use Data.Scatterv. Rank {}".format(rank))

# Send and Recv a point
y = None
if master:
    y = data.datapoint(0)
    for i in range(1, size):
        dp = data.datapoint(i)
        dp.Isend(dest=i, world=world)
else:
    y = DataPoint.Irecv(0, world)


assert np.all(y.data == dSave[rank, :]) and np.all(y.predictedData == pSave[rank, :]), Exception("Could not use Data.Isend/Irecv. Rank {}".format(rank))

##############################################################################
# Test Frequency Domain Data
##############################################################################
fdSave = FdemData.read_csv(dataPath+"Resolve_small.txt", dataPath+"FdemSystem2.stm")
if master:
    fd = fdSave
else:
    fd = FdemData(systems=dataPath+"FdemSystem2.stm")

# Bcast
fd1 = fd.Bcast(world)

assert np.allclose(fd1.data, fdSave.data, equal_nan=True), Exception("Could not use FdemData.Bcast. Rank {}".format(rank))

# Scatterv
starts, chunks = myMPI.loadBalance1D_shrinkingArrays(fd1.nPoints, size)
i0 = starts[rank]
i1 = i0 + chunks[rank]

fd1 = fd.Scatterv(starts, chunks, world)

assert np.allclose(fd1.data, fdSave.data[i0:i1, :], equal_nan=True), Exception("Could not use FdemData.Scatterv. Rank {}".format(rank))

# Point by Point read in and send
if master:
    fd = FdemData._initialize_sequential_reading(dataPath+"Resolve_small.txt", dataPath+"FdemSystem2.stm")

# Send and recieve a single datapoint from the file.
if master:
    fdp = fd._read_record()
    for i in range(1, size):
        fdp1 = fd._read_record()
        fdp1.Isend(dest=i, world=world)
else:
    fdp = FdemDataPoint.Irecv(source=0, world=world)

assert np.allclose(fdp.data, fdSave.data[rank, :], equal_nan=True), Exception("Could not use FdemData.Isend/Irecv. Rank {}".format(rank))


systems = FdemSystem.read(dataPath + "FdemSystem2.stm")

# Send and recieve a single datapoint from the file.
if master:
    fd = FdemData._initialize_sequential_reading(dataPath+"Resolve_small.txt", dataPath+"FdemSystem2.stm")

    fdp = fd._read_record()
    for i in range(1, size):
        fdp1 = fd._read_record()
        fdp1.Isend(dest=i, world=world, system=systems)
else:
    fdp = FdemDataPoint.Irecv(source=0, world=world, system=systems)

assert np.allclose(fdp.data, fdSave.data[rank, :], equal_nan=True), Exception("Could not use FdemData.Isend/Irecv, with pre-existing system class. Rank {}".format(rank))

##############################################################################
# Test TdemData
##############################################################################

tdSave = TdemData.read_csv([dataPath+"Skytem_HM_small.txt", dataPath+"Skytem_LM_small.txt"], [dataPath+"SkytemHM-SLV.stm", dataPath+"SkytemLM-SLV.stm"])
# # ### Test Time Domain Data
if master:
    td = tdSave
else:
    td = TdemData()

#Bcast
td1 = td.Bcast(world)

assert np.allclose(td1.data, tdSave.data, equal_nan=True), Exception("Could not use TdemData.Bcast. Rank {}".format(rank))

# # Scatterv
starts, chunks = myMPI.loadBalance1D_shrinkingArrays(tdSave.nPoints, size)
i0 = starts[rank]
i1 = i0 + chunks[rank]

td1 = td.Scatterv(starts, chunks, world)

assert np.allclose(td1.data, tdSave.data[i0:i1, :], equal_nan=True), Exception("Could not use TdemData.Scatterv. Rank {}".format(rank))

# Send and recieve a single datapoint from the file.
if master:
    td = TdemData._initialize_sequential_reading([dataPath+"Skytem_HM_small.txt", dataPath+"Skytem_LM_small.txt"], [dataPath+"SkytemHM-SLV.stm", dataPath+"SkytemLM-SLV.stm"])

    tdp = td._read_record()
    for i in range(1, size):
        tdp1 = td._read_record()
        tdp1.Isend(dest=i, world=world)
else:
    tdp = TdemDataPoint.Irecv(source=0, world=world)

assert np.allclose(tdp.data, tdSave.data[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))

# Testing pre-read in system classes
sysPath = [dataPath+"SkytemHM-SLV.stm", dataPath+"SkytemLM-SLV.stm"]
systems = []
for s in sysPath:
    systems.append(TdemSystem.read(s))

# Send and recieve a single datapoint from the file.
if master:
    td = TdemData._initialize_sequential_reading([dataPath+"Skytem_HM_small.txt", dataPath+"Skytem_LM_small.txt"], [dataPath+"SkytemHM-SLV.stm", dataPath+"SkytemLM-SLV.stm"])
    tdp = td._read_record()
    for i in range(1, size):
        tdp1 = td._read_record()
        tdp1.Isend(dest=i, world=world, system=systems)
else:
    tdp = TdemDataPoint.Irecv(source=0, world=world, system=systems)

assert np.allclose(tdp.data, tdSave.data[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))

##############################################################################
# Test Tempest Data
##############################################################################
tdSave = TempestData.read_netcdf(dataPath+"Tempest.nc", dataPath+"Tempest.stm", indices=np.s_[:10])
# ### Test Time Domain Data
if master:
    td = tdSave
else:
    td = TempestData()

# Bcast
td1 = td.Bcast(world)

assert np.allclose(td1.data, tdSave.data, equal_nan=True), Exception("Could not use TempestData.Bcast. Rank {}".format(rank))

# # Scatterv
starts, chunks = myMPI.loadBalance1D_shrinkingArrays(tdSave.nPoints, size)
i0 = starts[rank]
i1 = i0 + chunks[rank]

td1 = td.Scatterv(starts, chunks, world)

assert np.allclose(td1.data, tdSave.data[i0:i1, :], equal_nan=True), Exception("Could not use TempestData.Scatterv. Rank {}".format(rank))
assert np.allclose(td1.primary_field, tdSave.primary_field[i0:i1, :], equal_nan=True), Exception("Could not use TempestData.Scatterv. Rank {}".format(rank))

# Send and recieve a single datapoint from the file.
if master:
    td = TempestData._initialize_sequential_reading(dataPath+"Tempest.nc", dataPath+"Tempest.stm")

    tdp = td._read_record(0)
    for i in range(1, size):
        tdp1 = td._read_record(i)
        tdp1.Isend(dest=i, world=world)
else:
    tdp = Tempest_datapoint.Irecv(source=0, world=world)

assert np.allclose(tdp.primary_field, tdSave.primary_field[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))
assert np.allclose(tdp.secondary_field, tdSave.secondary_field[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))
assert np.allclose(tdp.data, tdSave.data[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))

# Testing pre-read in system classes
system = TdemSystem.read(dataPath+"Tempest.stm")

# Send and recieve a single datapoint from the file.
if master:
    td = TempestData._initialize_sequential_reading(dataPath+"Tempest.nc", dataPath+"Tempest.stm")
    tdp = td._read_record(0)
    for i in range(1, size):
        tdp1 = td._read_record(i)
        tdp1.Isend(dest=i, world=world, system=system)
else:
    tdp = Tempest_datapoint.Irecv(source=0, world=world, system=system)

assert np.allclose(tdp.primary_field, tdSave.primary_field[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))
assert np.allclose(tdp.secondary_field, tdSave.secondary_field[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))
assert np.allclose(tdp.data, tdSave.data[rank, :], equal_nan=True), Exception("Could not use TdemData.Isend/Irecv. Rank {}".format(rank))

print("All tests passed. Rank {}".format(rank), flush=True)
