from __future__ import print_function

import sys
from collections import deque

from abc import ABC, abstractmethod
from sklearn.base import _pprint


class _BaseMonitor(ABC):
    """Base class for convergence monitors.

    Monitors check for and report convergence to :data:`sys.stderr`.

    The convergence criterion for EM must be defined in :meth:`converged`,
    and the meaning of ``tol`` will change depending on the criterion
    used.

    Parameters
    ----------
    tol : double
        Convergence threshold.

    n_iter : int
        Maximum number of iterations to perform.

    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.

    iter : int
        Number of iterations performed while training the model.
    """
    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = dict(vars(self), history=list(self.history))
        return "{0}({1})".format(
            class_name, _pprint(params, offset=len(class_name)))

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()

    def report(self, logprob):
        """Reports convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            prev_logprob = self.histor[-1] if self.history else float("nan")
            delta = logprob - prev_logprob
            message = self._template.format(
                iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @abstractmethod
    def _check_convergence(self):
        """Check for convergence independent of stopping criterion."""
        pass

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise."""
        return (self.iter == self.n_iter or self._check_convergence())


class ConvergenceMonitor(_BaseMonitor):
    def _check_convergence(self):
        """Absolute gain in log probability is less than ``tol``."""
        # XXX we might want to check that ``logprob`` is non-decreasing.
        return (len(self.history) == 2 and
                self.history[1] - self.history[0] < self.tol)


class RelativeMonitor(_BaseMonitor):
    def _check_convergence(self):
        """Relative gain in log probability is less than ``tol``."""
        if len(self.history) != 2:
            return False

        previous, current = self.history
        return (current - previous) / previous < self.tol


class ThresholdMonitor(_BaseMonitor):
    def _check_convergence(self):
        """Current log probability is greater or equal to ``tol``."""
        return self.history and self.history[-1] >= self.tol
