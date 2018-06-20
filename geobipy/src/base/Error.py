#class Error(Exception):
#    """Base class for exceptions in this module."""
#    pass
#
#
#class Emsg(Error):
#    """ Generic exception with a message """
#
#    def __init__(self, msg, extraFunc=None):
#        self.msg = msg
#        print("Error:" + msg)
#        if (not extraFunc is None):
#            extraFunc()
#        raise SystemExit()
#
#
#class InputError(Error):
#    """Exception raised for errors in the input.
#
#    Attributes:
#        expr -- input expression in which the error occurred
#        msg  -- explanation of the error
#    """
#
#    def __init__(self, expr, msg):
#        self.expr = expr
#        self.msg = msg
#
#
#class TransitionError(Error):
#    """Raised when an operation attempts a state transition that's not
#    allowed.
#
#    Attributes:
#        prev -- state at beginning of transition
#        next -- attempted new state
#        msg  -- explanation of why the specific transition is not allowed
#    """
#
#    def __init__(self, prev, next, msg):
#        self.prev = prev
#        self.next = next
#        self.msg = msg


def isIpython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
