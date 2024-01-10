""" @Stopwatch_Class
Module describing a Stopwatch for timing purposes
"""
import time as tm

class Stopwatch(object):

    def __init__(self, mpi=False):
        """ Initialize a stopwatch """
        self.running = False
        self.startTime = 0.0
        self.elapsedTime = 0.0
        self.lapTime = 0.0
        self.created = tm.strftime("%Y-%m-%d %H:%M:%S", tm.gmtime())
        self.mpi = mpi

    def start(self, display=False):
        """ Start the stopwatch """
        if (not self.running):
            if (self.startTime == 0.0):
                self.startTime = tm.time()
                self.lapTime = self.startTime
            self.running = True
        return

    def stop(self):
        """ Stop the stopwatch """
        if (self.running):
            self.elapsedTime = tm.time() - self.startTime
            self.running = False

    def reset(self):
        """ Reset the stopwatch """
        self.running = False
        self.startTime = 0.0
        self.elapsedTime = 0.0
        self.lapTime = 0.0

    def restart(self):
        """ Reset and Start the stopwatch """
        self.reset()
        self.start()

    def lap(self):
        """ Print the lap time """
        if (self.running):
            tmp = tm.time()
            tmp2 = tmp - self.lapTime
            self.lapTime = tmp
        else:
            tmp2 = 0.0
        return tmp2

    def timeinSeconds(self):
        """ Gets the time in seconds """
        if (self.running):
            self.elapsedTime = tm.time() - self.startTime
        return self.elapsedTime

    def time(self):
        print(self)

    def __str__(self):
        """ Displays the time as h:m:s.ms """
        tot = self.timeinSeconds()
        s, ms = divmod(tot, 1.0)
        m, s = divmod(s, 60.0)
        h, m = divmod(m, 60.0)
        #d,h = divmod(h,24.0)
        str = "Elapsed Time: %02d:%02d:%02d.%03d (h:m:s.ms)" % (
            h, m, s, 1000 * ms)
        return str
