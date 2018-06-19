""" Custom module to handle code logs. Uses pythons built in logging functionality but tweaks it to work with an ipython console """
import logging
from . import fileIO as fIO


class myLogger(object):

  def __init__(self, aName):
    """ Initialize a custom logging object to handle debugging.
    A custom class was necessary to """
    if (aName in logging.Logger.manager.loggerDict.keys()):
      pass
    else:
      global indentLevel
      indentLevel = 0
    self.indentStyle = '  '
    self.lg= logging.getLogger(aName) # Get the logger
    # Add a stream handler, but only if the list is empty
    if (not len(self.lg.handlers)):
      self.lg.addHandler(logging.StreamHandler())

  def resetIndent(self):
    """ Set the indent level to 0 """
    global indentLevel
    indentLevel=0

  def setLevel(self, lvl):
    """ Set the write out level """
    if (lvl == '-v1'):
#      self.lvl=1
      self.lg.setLevel(logging.INFO)
    elif (lvl == '-v2'):
#      self.lvl=2
      self.lg.setLevel(logging.DEBUG)
    elif (lvl == '-v3'):
#      self.lvl=3
      self.lg.setLevel(logging.WARNING)
    elif (lvl == '-v4'):
#      self.lvl=4
      self.lg.setLevel(logging.ERROR)
    elif (lvl == '-v5'):
#      self.lvl=5
      self.lg.setLevel(logging.CRITICAL)
    else:
      self.disable()

  def addFile(self, aName):
    """  Adds a log file to the logger """
    fIO.deleteFile(aName)
    self.lg.addHandler(logging.FileHandler(aName,mode='w'))

  def getIndent(self):
    """ Gets the appropriate number of indentations """
    return indentLevel * self.indentStyle

  def info(self, msg):
    """ Write an info message """
    # print(self.lvl)
    if (not self.enabled): return
    # if (self.lvl >= 1): #print('Info: '+self.getIndent()+msg)
    self.lg.info(self.getIndent()+msg)

  def debug(self, msg):
    """ Write a debug message """
    if (not self.enabled): return
    # if (self.lvl >= 2): #print('Debug: '+self.getIndent()+msg)
    self.lg.debug(self.getIndent()+msg)

  def warning(self, msg):
    """ Write a warning message """
    if (not self.enabled): return
    # if (self.lvl >= 3):  #print('Warning: '+self.getIndent()+msg)
    self.lg.warning(self.getIndent()+msg)

  def error(self, msg):
    """ Write an error message """
    if (not self.enabled): return
    # if (self.lvl >= 4):  #print('Error: '+self.getIndent()+msg)
    self.lg.error(self.getIndent()+msg)

  def critical(self, msg):
    """ Write a critical message """
    if (not self.enabled): return
    # if (self.lvl >= 5): #print('Critical: '+self.getIndent()+msg)
    self.lg.critical(self.getIndent()+msg)

  def enable(self):
    """ Turn on the logger """
    self.lg.disabled=False

  def enabled(self):
    """ Logical whether enabled """
    return (not self.lg.disabled)

  def disable(self):
    """ Turn off the logger """
    self.lg.disabled=True

  def indent(self):
    """ Increase the indent """
    global indentLevel
    indentLevel += 1

  def dedent(self):
    """ Decrease the indent """
    global indentLevel
    indentLevel -= 1
    if (indentLevel < 0): indentLevel=0
