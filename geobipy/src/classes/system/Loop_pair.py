from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..core import StatArray
from ..pointcloud.Point import Point
from ..pointcloud.PointCloud3D import PointCloud3D
from .EmLoop import EmLoop
from .EmLoops import EmLoops
from .CircularLoop import CircularLoop
from ...base.HDF.hdfRead import read_item

class Loop_pair(Point, PointCloud3D):

    def __init__(self, transmitter=None, receiver=None, **kwargs):

        super().__init__(**kwargs)
        self.x.name = 'x offset'
        self.y.name = 'y offset'
        self.z.name = 'z offset'

        if transmitter is None and receiver is None:
            return

        self.transmitter = transmitter
        self.receiver = receiver

    @property
    def addressof(self):
        msg = super().addressof
        msg += "transmitter:\n{}".format(("|   "+self.transmitter.addressof.replace("\n", "\n|   "))[:-4])
        msg += "receiver:\n{}".format(("|   "+self.receiver.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def Geometry(self):
        try:
            from gatdaem1d import Geometry
        except Exception as e:
            raise Exception("{}\n gatdaem1d is not installed. Please see instructions".format(e))

        # Generate the Brodie Geometry class
        return Geometry( self.transmitter.z.item(),
                         self.transmitter.roll.item(),
                        -self.transmitter.pitch.item(),
                        -self.transmitter.yaw.item(),
                         self.x.item(), self.y.item(), self.z.item(),
                         self.receiver.roll.item(),
                        -self.receiver.pitch.item(),
                        -self.receiver.yaw.item())

    @property
    def transmitter(self):
        return self._transmitter

    @transmitter.setter
    def transmitter(self, value):
        if value is not None:
            assert isinstance(value, (EmLoop, EmLoops)), TypeError('transmitter must be a geobipy.EmLoop')
            self._transmitter = value

    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, value):
        if value is not None:
            assert isinstance(value, (EmLoop, EmLoops)), TypeError('transmitter must be a geobipy.EmLoop')
            self._receiver = value

            self.x = self.receiver.x - self.transmitter.x
            self.y = self.receiver.y - self.transmitter.y
            self.z = self.receiver.z - self.transmitter.z

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._transmitter = deepcopy(self.transmitter)
        out._receiver = deepcopy(self.receiver)
        return out

    def perturb(self):
        super().perturb()
        self.transmitter.perturb()
        self.receiver.perturb()

    def set_priors(self, **kwargs):
        transmitter_kwargs = {k.replace('transmitter_', ''): kwargs.get(k, False) for k in kwargs.keys() if 'transmitter' in k}
        transmitter_kwargs['prng'] = kwargs.get('prng')
        self.transmitter.set_priors(**transmitter_kwargs)

        receiver_kwargs = {k.replace('receiver_', ''): kwargs.get(k, False) for k in kwargs.keys() if 'receiver' in k}
        receiver_kwargs['prng'] = kwargs.get('prng')

        position_kwargs = {k:receiver_kwargs.pop(k, False) for k in list(receiver_kwargs.keys()) if any(x in k for x in ('_x', 'x_', '_y', 'y_', '_z', 'z_'))}
        position_kwargs['prng'] = kwargs.get('prng')
        super().set_priors(**position_kwargs)
        self.receiver.set_priors(**receiver_kwargs)

    def set_proposals(self, **kwargs):
        transmitter_kwargs = {k.replace('transmitter_', ''): kwargs.get(k, False) for k in kwargs.keys() if 'transmitter' in k}
        transmitter_kwargs['prng'] = kwargs.get('prng')
        self.transmitter.set_proposals(**transmitter_kwargs)

        receiver_kwargs = {k.replace('receiver_', ''): kwargs.get(k, False) for k in kwargs.keys() if 'receiver' in k}
        receiver_kwargs['prng'] = kwargs.get('prng')

        position_kwargs = {k:receiver_kwargs.pop(k, False) for k in list(receiver_kwargs.keys()) if any(x in k for x in ('_x', 'x_', '_y', 'y_', '_z', 'z_'))}
        position_kwargs['prng'] = kwargs.get('prng')
        super().set_proposals(**position_kwargs)

        self.receiver.set_proposals(**receiver_kwargs)

    def set_posteriors(self):
        super().set_posteriors()
        self.transmitter.set_posteriors()
        self.receiver.set_posteriors()

    def reset_posteriors(self):
        super().reset_posteriors()
        self.transmitter.reset_posteriors()
        self.receiver.reset_posteriors()

    @property
    def summary(self):
        msg = "transmitter:\n{}".format("|   "+(self.transmitter.summary.replace("\n", "\n|   "))[:-4])
        msg = "offset:\n{}".format("|   "+(super().summary.replace("\n", "\n|   "))[:-4])
        msg += "receiver:\n{}".format("|   "+(self.receiver.summary.replace("\n", "\n|   "))[:-4])
        return msg

    def update_posteriors(self):
        super().update_posteriors()
        self.transmitter.update_posteriors()
        self.receiver.update_posteriors()

    def _init_posterior_plots(self, gs=None):

        if not self.transmitter.hasPosteriors and not super().hasPosteriors and not self.receiver.hasPosteriors:
            return [[], []]

        if gs is None:
            gs = plt.figure()

        if isinstance(gs, matplotlib.figure.Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        splt = gs.subgridspec(1, self.transmitter.hasPosteriors + super().hasPosteriors + self.receiver.hasPosteriors, wspace=0.3)
        ax = []
        i = 0
        if self.transmitter.hasPosteriors:
            ax.append(self.transmitter._init_posterior_plots(splt[i]))
            i += 1

        if super().hasPosteriors:
            ax.append(super()._init_posterior_plots(splt[i]))
            i += 1

        # Reciever axes
        if self.receiver.hasPosteriors:
            ax.append(self.receiver._init_posterior_plots(splt[i]))

        return ax

    def plot_posteriors(self, axes=None, **kwargs):

        n_posteriors = self.transmitter.hasPosteriors + self.hasPosteriors + self.receiver.hasPosteriors
        if n_posteriors == 0:
            return

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        n_posteriors = self.transmitter.hasPosteriors + self.hasPosteriors + self.receiver.hasPosteriors
        assert len(axes) == n_posteriors, ValueError("Length {} axes must have length {} list for the posteriors. self.init_posterior_plots can generate them".format(len(axes), n_posteriors))

        transmitter_kwargs = kwargs.pop('transmitter_kwargs', {})
        offset_kwargs = kwargs.pop('offset_kwargs', {})
        receiver_kwargs = kwargs.pop('receiver_kwargs', {})

        overlay = kwargs.get('overlay')
        if not overlay is None:
            transmitter_kwargs['overlay'] = overlay.transmitter
            offset_kwargs['overlay'] = overlay.loop_pair
            receiver_kwargs['overlay'] = overlay.receiver

        i = 0
        if self.transmitter.hasPosteriors:
            self.transmitter.plot_posteriors(axes = axes[i], **transmitter_kwargs)
            i += 1

        if super().hasPosteriors:
            super().plot_posteriors(axes = axes[i], **offset_kwargs)
            i += 1

        if self.receiver.hasPosteriors:
            self.receiver.plot_posteriors(axes = axes[i], **receiver_kwargs)

    @property
    def probability(self):
        return self.transmitter.probability + super().probability + self.receiver.probability

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        grp = super().createHdf(parent, name, withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.transmitter.createHdf(grp, 'transmitter', add_axis=add_axis, fillvalue=fillvalue)
        self.receiver.createHdf(grp, 'receiver', add_axis=add_axis, fillvalue=fillvalue)

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]
        self.transmitter.writeHdf(grp, 'transmitter', index=index)
        self.receiver.writeHdf(grp, 'receiver', index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        self = super(Loop_pair, cls).fromHdf(grp, index)

        self._transmitter = read_item(grp['transmitter'], index=index)
        self._receiver = read_item(grp['receiver'], index=index)
        return self

    def Isend(self, dest, world):
        super().Isend(dest, world)
        self.transmitter.Isend(dest, world)
        self.receiver.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world):
        self = super(Loop_pair, cls).Irecv(source, world)
        self._transmitter = CircularLoop.Irecv(source, world)
        self._receiver = CircularLoop.Irecv(source, world)
        return self