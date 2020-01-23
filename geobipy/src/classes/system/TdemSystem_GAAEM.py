import numpy as np
from ...classes.core.myObject import myObject
from ...base import fileIO as fIO
from ...classes.core import StatArray
from . import TdemSystem

try:
    from gatdaem1d import TDAEMSystem

    class TdemSystem_GAAEM(TdemSystem.TdemSystem, TDAEMSystem):
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

        def __init__(self, systemFilename):
            """ Nothing needed """            

            # Check that the file exists, rBodies class does not handle errors
            assert fIO.fileExists(systemFilename),'Could not open file: ' + systemFilename

            TDAEMSystem.__init__(self, systemFilename)
            self.fileName = systemFilename


        @property
        def isGA(self):
            return True

        @property
        def nTimes(self):
            return self.nwindows()


        @property
        def times(self):
            """Time windows."""
            return StatArray.StatArray(self.windows.centre, name='Time', units='s')
            

        def read(self, systemFilename):
            # Read in the System file
            self.__init__(systemFilename)
            assert np.min(np.diff(self.windows.centre)) > 0.0, ValueError("Receiver window times must monotonically increase for system "+systemFilename)

            self.readCurrentWaveform(systemFilename)
        
        def readCurrentWaveform(self, systemFname):
            get = False
            time = []
            current = []
            
            with open(systemFname, 'r') as f:
                for i, line in enumerate(f):
                    
                    if ('WaveFormCurrent End' in line):
                        self.waveform.transmitterTime = np.asarray(time[:-1])
                        self.waveform.transmitterCurrent = np.asarray(current[:-1])
                        return
                    
                    if (get):
                        x = fIO.getRealNumbersfromLine(line)
                        if len(x) > 0:
                            time.append(x[0])
                            current.append(x[1])
                        
                    if ('WaveFormCurrent Begin' in line):
                        get = True

        def summary(self, out=False):
            msg = ("TdemSystem: \n"
                   "{}\n"
                   "{}\n").format(self.fileName, self.times.summary(True))
            return msg if out else print(msg)
    


except:
    class TdemSystem_GAAEM(object):
        
        def __init__(self, *args, **kwargs):
            h=("Could not import the time domain forward modeller from GA_AEM. \n"
            "Please see the package's README for instructions on how to install it \n"
            "Check that you have loaded the compiler that was used to compile the forward modeller")
            print(Warning(h))
