""" @fileIO
Module with custom file handling operations
"""
import os
from subprocess import Popen, PIPE, STDOUT
from numpy import asarray, atleast_1d, float64, zeros
import re
from pandas import read_csv


def filesExist(fNames):
    """Check if all files in fNames exist on disk

    Parameters
    ----------
    fNames : list of str

    Returns
    -------
    out : bool
        Whether all files exist or not

    """
    return all([fileExists(f) for f in fNames])


def fileExists(fname):
    """Check if a single file exists on disk

    Parameters
    ----------
    fname : str
        A file name

    Returns
    -------
    out : bool
        Whether the file exists or not

    """
    return os.path.isfile(fname)


def dirExists(dirPath):
    """Check if a directory exists on disk

    Parameters
    ----------
    dirPath : str
        A directory path

    Returns
    -------
    out : bool
        Whether the directory path exists

    """
    return os.path.isdir(dirPath)


def deleteFile(fname):
    """Deletes a file if it exists

    Parameters
    ----------
    fName : str
        A path and/or file name

    """
    try:
        os.remove(fname)
    except:
        pass


def isFileExtension(fname, ext):
    """Check that the filename is of the given extension

    Parameters
    ----------
    fname : str
        A path and/or file name
    ext : str
        The extension you want to check the file name against

    Returns
    -------
    out : bool
        Whether the extension of fname matches ext

    """
    return getFileExtension(fname) == ext


def getFileExtension(fname):
    """Gets the extension of the filename

    Parameters
    ----------
    fname : str
        A path and/or file name

    Returns
    -------
    out : str
        The extension of the filename

    """
    tmp, ext = os.path.splitext(fname)
    return ext[1:]

def bytes2readable(nBytes):
    """Convert bytes to KB, MB, ..., PB

    Parameters
    ----------
    nBytes : float
        The number of bytes

    Returns
    -------
    out : str
        The number of KB, MB, etc.

    """
    assert nBytes >= 0, ValueError("")
    tmp = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']
    for x in tmp:
        if nBytes < 1024.0:
            return "%3.1f %s" % (nBytes, x)
        nBytes /= 1024.0
    return "%3.1f %s" % (nBytes, 'PB')

def getFileSize(fName):
    """Get the size of a file on disk

    Parameters
    ----------
    fName : str
        A path and/or file name

    Returns
    -------
    out : str
        The file size in KB, MB, etc.

    """
    if os.path.isfile(fName):
        return bytes2readable(os.stat(fName).st_size)

def wccount(filename):
    """Count the number of lines in a file using a wc system call

    Parameters
    ----------
    fName : str
        A path and/or file name

    Returns
    -------
    out : int
        The number of lines in the file

    """
    with open(filename, 'rb') as f:
        return sum(1 for line in f)

def getNlines(fname, nHeaders=0):
    """Gets the number of lines in a file after taking into account the number of header lines

    Parameters
    ----------
    fName : str
        A path and/or file name
    nHeaders : int, optional
        Subtract the number of header lines from the total number of lines in the file

    Returns
    -------
    out : int, optional
        Number of lines without the headers

    """
    assert fileExists(fname), 'Cannot find file '+fname
    nLines = wccount(fname)
    return nLines - nHeaders
#    with open(fname) as f: # Open the file
#        skipLines(f,nHeaders) # Skip the header lines
#        return sum(1 for line in f) # Sum up the end of line flags


def getNcolumns(fName, nHeaders=0):
    """Gets the number of columns in a file using the line after nHeaders

    Parameters
    ----------
    fName : str
        A path and/or file name
    nHeaders : int, optional
        Number of header lines to skip before obtaining the number of columns

    Returns
    -------
    out : int
        The number of columns

    """
    assert fileExists(fName), 'Cannot find file '+fName
    with open(fName) as f:  # Open the file
        skipLines(f, nHeaders)  # Skip the header lines
        line = f.readline()
        line = parseString(line)
        return len(line)


# def skipLines(f, nLines=0):
#     """Skip N lines in an open file object

#     Parameters
#     ----------
#     f : _io.TextIOWrapper
#         A file handle generated with open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None).
#     nLines : int
#         The number of lines to skip.

#     """
#     for _ in range(nLines):
#         next(f)


# def read_columns(fName, indices=None, nHeaders=0, nLines=0):
#     """Reads specified columns from a file

#     Parameters
#     ----------
#     fName : str
#         A path and/or file name.
#     indices : in or list of ints, optional
#         The indices of the columns to read in from the file.  By default, all columns are read in.

#     nHeaders : int, optional
#         The number of header lines to skip in the file.
#     nLines : int, optional
#         The number of lines to read in.  By default, all lines are read in after the header lines.

#     Returns
#     -------
#     out : numpy.ndarray
#         2D array containing the lines in the file.

#     """

#     assert fileExists(fName), 'Cannot find file '+fName

#     indices = atleast_1d(indices)
#     if (nLines == 0):
#         # Get the number of lines in the file
#         nLines = getNlines(fName, nHeaders)

#     nCols = getNcolumns(fName, nHeaders) if indices is None else indices.size

#     values = zeros([nLines, nCols], dtype='float64', order='F')  # Initialize output

#     with open(fName) as f:  # Open the file
#         skipLines(f, nHeaders)  # Skip header lines
#         for j, line in enumerate(f):  # For each line in the file
#             # try:
#                 values[j, ] = getRealNumbersfromLine(line, indices)  # grab the requested entries
#             # except:
#             #     assert False, Exception("Could not read numbers from line {} in file {} \n\n {}".format(j+nHeaders, fName, line))
#     return values


def get_real_numbers_from_line(line, indices=None, delimiters=','):
    """Reads strictly the numbers from a string

    Parameters
    ----------
    line : str
        A string, or a line from a file.
    i : in or list of ints, optional
        The indices of the entries to read from the string.  By default, all entries are read in.
    delimiters : str
        Splits the line based on these delimiters.

    Returns
    -------
    out : numpy.ndarray
        The values read in from the string.

    """

    line = parseString(line, delimiters)

    if not indices is None:
        indices = atleast_1d(indices)
        values = asarray([i for i in indices], float64)
    else:
        values = asarray(line, float64)

    return values


def parseString(this, delimiters=','):
    """Parse a string into its entries

    Parameters
    ----------
    this : str
        The string to parse.
    delimiters : str
        Patterns to split against, e.g. ',' splits at every comma.

    Returns
    -------
    out : list of str
        A list of the parsed entries

    """

    line = this.replace("*", "NaN")
    # Replace multiple spaces with a single space
    line = ' '.join(line.split())
    delims = ' |, |,' + '|' + delimiters
    # Split the entries based on these delimiters
    line = re.split(delims, line)
    return list(filter(None, line))


def get_column_name(filename):
    """Read headers names from a line in a file

    Parameters
    ----------
    fName : str
        A path and/or file name.
    i : in or list of ints, optional
        The indices of the entries to read from the string.  By default, all entries are read in.
    nHeaders : int, optional
        The number of header lines to skip in the file.

    Returns
    -------
    out : list of str
        A list of the parsed header entries.

    """
    assert fileExists(filename), 'Cannot find file {}'.format(filename)

    channels = read_csv(filename, nrows=0).columns

    if len(channels) == 1:
        channels = read_csv(filename, nrows=0, sep=r"\s+").columns

    channels = [c.strip() for c in channels]

    return channels


def int2str(i, N):
    """Converts an integer to a string with leading zeros in order to maintain the correct order in the file system

    Parameters
    ----------
    i : int
        The integer to convert to a string.
    N : int
        The maximum number of digits you wish to have in the integer. e.g. int2str(3,4)='0003'.

    Returns
    -------
    out : str
        The integer padded with zeroes on the front.

    """
    return ('{{0:0{0:d}d}}').format(N).format(i)
