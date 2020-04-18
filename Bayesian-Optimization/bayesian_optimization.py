# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:16:34 2018

@author: HL
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from helpers import (UtilityFunction, PrintLog, acq_max, ensure_rng, unique_rows)

def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))

class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added
    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.observe_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.
        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """

        self.random_state = ensure_rng(random_state)

        # Some function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self.keys = list(pbounds.keys())
        # Create an array with parameters bounds
        self.bounds = np.array(list(pbounds.values()), dtype=np.float)
        # Find number of parameters
        self.dim = len(self.keys)

        # preallocated memory for X and Y points
        self._Xarr = None
        self._Yarr = None

        # Number of observations
        self._length = 0

        # Views of the preallocated arrays showing only populated data
        self._Xview = None
        self._Yview = None

        self._cache = {}  # keep track of unique points we have seen so far

    @property
    def X(self):
        return self._Xview

    @property
    def Y(self):
        return self._Yview

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        return self._length

    def _dict_to_points(self, points_dict):
        """
        Example:
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> points_dict = {'p1': [0, .5, 1], 'p2': [0, 1, 2]}
        >>> space._dict_to_points(points_dict)
        [[0, 0], [1, 0.5], [2, 1]]
        """
        # Consistency check
        param_tup_lens = []

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))

        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points '
                             'must be entered for every parameter.')

        # Turn into list of lists
        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])

        # Take transpose of list
        points = list(map(list, zip(*all_points)))
        return points

    def observe_point(self, x):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.
        Notes
        -----
        If x has been previously seen returns a cached value of y.
        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        Returns
        -------
        y : float
            target function value.
        """
        x = np.asarray(x).ravel()
        assert x.size == self.dim, 'x must have the same dimensions'

        if x in self:
            # Lookup previously seen point
            y = self._cache[_hashable(x)]
        else:
            # measure the target function
            params = dict(zip(self.keys, x))
            y = self.target_func(**params)
            self.add_observation(x, y)
        return y

    def add_observation(self, x, y):
        """
        Append a point and its target value to the known data.
        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        y : float
            target function value
        Raises
        ------
        KeyError:
            if the point is not unique
        Notes
        -----
        runs in ammortized constant time
        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        if self._length >= self._n_alloc_rows:
            self._allocate((self._length + 1) * 2)

        x = np.asarray(x).ravel()

        # Insert data into unique dictionary
        self._cache[_hashable(x)] = y

        # Insert data into preallocated arrays
        self._Xarr[self._length] = x
        self._Yarr[self._length] = y
        # Expand views to encompass the new data point
        self._length += 1

        # Create views of the data
        self._Xview = self._Xarr[:self._length]
        self._Yview = self._Yarr[:self._length]

    def _allocate(self, num):
        """
        Allocate enough memory to store `num` points
        """
        if num <= self._n_alloc_rows:
            raise ValueError('num must be larger than current array length')

        self._assert_internal_invariants()

        # Allocate new memory
        _Xnew = np.empty((num, self.bounds.shape[0]))
        _Ynew = np.empty(num)

        # Copy the old data into the new
        if self._Xarr is not None:
            _Xnew[:self._length] = self._Xarr[:self._length]
            _Ynew[:self._length] = self._Yarr[:self._length]
        self._Xarr = _Xnew
        self._Yarr = _Ynew

        # Create views of the data
        self._Xview = self._Xarr[:self._length]
        self._Yview = self._Yarr[:self._length]

    @property
    def _n_alloc_rows(self):
        """ Number of allocated rows """
        return 0 if self._Xarr is None else self._Xarr.shape[0]

    def random_points(self, num):
        """
        Creates random points within the bounds of the space
        Parameters
        ----------
        num : int
            Number of random points to create
        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self.keys`
        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(3)
        array([[ 55.33253689,   0.54488318],
               [ 71.80374727,   0.4236548 ],
               [ 60.67357423,   0.64589411]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((num, self.dim))
        for col, (lower, upper) in enumerate(self.bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=num)
        return data

    def max_point(self):
        """
        Return the current parameters that best maximize target function with
        that maximum value.
        """
        return {'max_val': self.Y.max(),
                'max_params': dict(zip(self.keys,
                                       self.X[self.Y.argmax()]))}

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds
        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self.bounds[row] = new_bounds[key]

    def _assert_internal_invariants(self, fast=True):
        """
        Run internal consistency checks to ensure that data structure
        assumptions have not been violated.
        """
        if self._Xarr is None:
            assert self._Yarr is None
            assert self._Xview is None
            assert self._Yview is None
        else:
            assert self._Yarr is not None
            assert self._Xview is not None
            assert self._Yview is not None
            assert len(self._Xview) == self._length
            assert len(self._Yview) == self._length
            assert len(self._Xarr) == len(self._Yarr)

            if not fast:
                # run slower checks
                assert np.all(unique_rows(self.X))
                # assert np.may_share_memory(self._Xview, self._Xarr)
                # assert np.may_share_memory(self._Yview, self._Yarr)


class BayesianOptimization(object):

    def __init__(self, f, pbounds, random_state=None, verbose=1):
        """
        :param f:
            Function to be maximized.
        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.
        :param verbose:
            Whether or not to print progress.
        """
        # Store the original dictionary
        self.pbounds = pbounds

        self.random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self.space = TargetSpace(f, pbounds, random_state)

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Counter of iterations
        self.i = 0

        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=self.random_state
        )

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.space.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # non-public config for maximizing the aquisition function
        # (used to speedup tests, but generally leave these as is)
        self._acqkw = {'n_warmup': 100000, 'n_iter': 250}

        # Verbose
        self.verbose = verbose

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.
        :param init_points:
            Number of random points to probe.
        """
        # Concatenate new random points to possible existing
        # points from self.explore method.
        rand_points = self.space.random_points(init_points)
        self.init_points.extend(rand_points)

        # Evaluate target function at all initialization points
        for x in self.init_points:
            y = self._observe_point(x)

        # Add the points from `self.initialize` to the observations
        if self.x_init:
            x_init = np.vstack(self.x_init)
            y_init = np.hstack(self.y_init)
            for x, y in zip(x_init, y_init):
                self.space.add_observation(x, y)
                if self.verbose:
                    self.plog.print_step(x, y)

        # Updates the flag
        self.initialized = True

    def _observe_point(self, x):
        y = self.space.observe_point(x)
        if self.verbose:
            self.plog.print_step(x, y)
        return y

    def explore(self, points_dict, eager=False):
        """Method to explore user defined points.
        :param points_dict:
        :param eager: if True, these points are evaulated immediately
        """
        if eager:
            self.plog.reset_timer()
            if self.verbose:
                self.plog.print_header(initialization=True)

            points = self.space._dict_to_points(points_dict)
            for x in points:
                self._observe_point(x)
        else:
            points = self.space._dict_to_points(points_dict)
            self.init_points = points

    def initialize(self, points_dict):
        """
        Method to introduce points for which the target function value is known
        :param points_dict:
            dictionary with self.keys and 'target' as keys, and list of
            corresponding values as values.
        ex:
            {
                'target': [-1166.19102, -1142.71370, -1138.68293],
                'alpha': [7.0034, 6.6186, 6.0798],
                'colsample_bytree': [0.6849, 0.7314, 0.9540],
                'gamma': [8.3673, 3.5455, 2.3281],
            }
        :return:
        """

        self.y_init.extend(points_dict['target'])
        for i in range(len(points_dict['target'])):
            all_points = []
            for key in self.space.keys:
                all_points.append(points_dict[key][i])
            self.x_init.append(all_points)

    def initialize_df(self, points_df):
        """
        Method to introduce point for which the target function
        value is known from pandas dataframe file
        :param points_df:
            pandas dataframe with columns (target, {list of columns matching
            self.keys})
        ex:
              target        alpha      colsample_bytree        gamma
        -1166.19102       7.0034                0.6849       8.3673
        -1142.71370       6.6186                0.7314       3.5455
        -1138.68293       6.0798                0.9540       2.3281
        -1146.65974       2.4566                0.9290       0.3456
        -1160.32854       1.9821                0.5298       8.7863
        :return:
        """

        for i in points_df.index:
            self.y_init.append(points_df.loc[i, 'target'])

            all_points = []
            for key in self.space.keys:
                all_points.append(points_df.loc[i, key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds
        :param new_bounds:
            A dictionary with the parameter name and its new bounds
        """
        # Update the internal object stored dict
        self.pbounds.update(new_bounds)
        self.space.set_bounds(new_bounds)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """
        Main optimization method.
        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.
        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.
        :param acq:
            Acquisition function to be used, defaults to Upper Confidence Bound.
        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object
        Returns
        -------
        :return: Nothing
        Example:
        >>> xs = np.linspace(-2, 10, 10000)
        >>> f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2/10) + 1/ (xs**2 + 1)
        >>> bo = BayesianOptimization(f=lambda x: f[int(x)],
        >>>                           pbounds={"x": (0, len(f)-1)})
        >>> bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=1)
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.space.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        self.gp.fit(self.space.X, self.space.Y)

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.space.bounds,
                        random_state=self.random_state,
                        **self._acqkw)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            while x_max in self.space:
                x_max = self.space.random_points(1)[0]
                pwarning = True

            # Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            if self.verbose:
                self.plog.print_step(x_max, y, pwarning)

            # Updating the GP.
            self.gp.fit(self.space.X, self.space.Y)

            # Update the best params seen so far
            self.res['max'] = self.space.max_point()
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x_max)))

            # Update maximum value to search for next probe point.
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.space.bounds,
                            random_state=self.random_state,
                            **self._acqkw)

            # Keep track of total number of iterations
            self.i += 1

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

    def points_to_csv(self, file_name):
        """
        After training all points for which we know target variable
        (both from initialization and optimization) are saved
        :param file_name: name of the file where points will be saved in the csv
            format
        :return: None
        """

        points = np.hstack((self.space.X, np.expand_dims(self.space.Y, axis=1)))
        header = ','.join(self.space.keys + ['target'])
        np.savetxt(file_name, points, header=header, delimiter=',', comments='')


        
    @property
    def X(self):
        return self.space.X

    @property
    def Y(self):
        return self.space.Y

    @property
    def keys(self):
        return self.space.keys

    @property
    def f(self):
        return self.space.target_func

    @property
    def bounds(self):
        return self.space.bounds

    @property
    def dim(self):
        return self.space.dim
        