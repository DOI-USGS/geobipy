import numpy as np
from ...classes.core.myObject import myObject
from ...base import fileIO as fIO
from ...classes.core import StatArray

try:
    from gatdaem1d import TDAEMSystem

    class TdemSystem_GAAEM(TDAEMSystem):
        """ Initialize a Time domain system class

        TdemSystem(systemFileName)

        Parameters
        ----------
        systemFileName : str
            The system file to read from

        Returns
        -------
        out : TdemSystem
            A time domain system class

        """

        def __init__(self, system_filename):
            """ Nothing needed """

            # Check that the file exists, rBodies class does not handle errors
            assert fIO.fileExists(system_filename), 'Could not open file: ' + system_filename

            super().__init__(system_filename)
            self.off_time = self.windows.centre
            self.read_components(system_filename)
            self.filename = system_filename

        @property
        def isGA(self):
            return True

        @classmethod
        def read(cls, system_filename):
            # Read in the System file
            self = super(TdemSystem_GAAEM, cls).__init__(system_filename)
            assert np.min(np.diff(self.windows.centre)) > 0.0, ValueError(
                "Receiver window times must monotonically increase for system "+system_filename)

            self.read_components(system_filename)
            self.readCurrentWaveform(system_filename)
            return self

        def read_components(self, system_filename):
            self._components = []
            with open(system_filename, 'r') as f:
                for i, line in enumerate(f):
                    if 'XOutputScaling' in line:
                        value = np.float64(line.strip().split()[-1])
                        if value != 0.0:
                            self._components.append('x')
                    elif 'YOutputScaling' in line:
                        value = np.float64(line.strip().split()[-1])
                        if value != 0.0:
                            self._components.append('y')
                    elif 'ZOutputScaling' in line:
                        value = np.float64(line.strip().split()[-1])
                        if value != 0.0:
                            self._components.append('z')

        def readCurrentWaveform(self, systemFname):
            get = False
            time = []
            current = []

            with open(systemFname, 'r') as f:
                for i, line in enumerate(f):

                    if ('WaveFormCurrent End' in line):
                        self.waveform.transmitterTime = np.asarray(time[:-1])
                        self.waveform.transmitterCurrent = np.asarray(
                            current[:-1])
                        return

                    if (get):
                        x = fIO.get_real_numbers_from_line(line)
                        if len(x) > 0:
                            time.append(x[0])
                            current.append(x[1])

                    if ('WaveFormCurrent Begin' in line):
                        get = True

        @property
        def summary(self):
            msg = ("TdemSystem: \n"
                   "{}\n"
                   "{}\n").format(self.fileName, self.times.summary)
            return msg


except:
    class TdemSystem_GAAEM(object):

        def __init__(self, *args, **kwargs):
            h = ("\nCould not import the time domain forward modeller from GA_AEM. \n"
                 "Please see the package's README for instructions on how to install it \n"
                 "Check that you have loaded the compiler that was used to compile the forward modeller\n")
            print(Warning(h))
