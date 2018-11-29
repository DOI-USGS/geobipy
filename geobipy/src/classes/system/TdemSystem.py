import numpy as np
from ...classes.core.myObject import myObject
from ...base import fileIO as fIO

try:
    from gatdaem1d import TDAEMSystem

    class TdemSystem(TDAEMSystem):
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

        def __init__(self):
            """ Nothing needed """

        def read(self, systemFilename):

            # Check that the file exists, rBodies class does not handle errors
            assert fIO.fileExists(systemFilename),'Could not open file: ' + systemFilename

            # Read in the System file
            TDAEMSystem.__init__(self, systemFilename)
            self.sysFname = systemFilename
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
                        time.append(x[0])
                        current.append(x[1])
                        
                    if ('WaveFormCurrent Begin' in line):
                        get = True  
    


except:
    h=("Could not find a Time Domain forward modeller. \n"
       "Please see the package's README for instructions on how to install one \n"
       "Check that you have loaded the compiler that was used to compile the forward modeller")
    print(Warning(h))
    class TdemSystem(object):
        "nothing in here"
