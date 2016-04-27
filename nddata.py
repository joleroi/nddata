"""Scratch for NDDataArray class"""

import numpy as np
from astropy.units import Quantity, Unit
from astropy.table import Table, Column
import IPython


class NDDataArray(object):
    """ND Data Array

    Follows numpy convention for arrays
    """

    def __init__(self):
        self._axes = list()
        self._data = None

    def add_axis(self, axis):
        default_names = {0 : 'x', 1 : 'y', 2: 'z'}
        if axis.name is None:
            axis.name = default_names[self.dim]
        self._axes = [axis] + self._axes
        
        # Quick solution: delete data to prevent unwanted behaviour
        self._data = None

    @property
    def axes(self):
        return self._axes

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data)
        d = len(data.shape)
        if d != self.dim:
            raise ValueError('Overall dimensions to not match. Data: {}, Hist: {}'.format(d, self.dim))

        for dim in np.arange(self.dim):
            if self.axes[dim].nbins != data.shape[dim]:
                a = self.axes[dim]
                raise ValueError('Data shape does not match in dimension {d}\n'
                                 'Axis "{n}": {sa}, Data {sd}'.format(
                                     d = dim, n = a.name, sa = a.nbins, sd = data.shape[dim]))

        self._data = data


    @property
    def axis_names(self):
        """Currently set axis names"""
        return [a.name for a in self.axes]
    
    def get_axis_index(self, name):
        """Return axis index by it name"""
        for a in self.axes:
            if a.name == name:
                return self.axes.index(a)
        raise ValueError("No axis with name {}".format(name))


    def get_axis(self, name):
        """Return axis by it name"""
        idx = self.get_axis_index(name)
        return self.axes[idx]

    @property
    def dim(self):
        return len(self.axes)

    def to_table(self):
        """Convert to astropy.Table"""
        cols = [Column(data=[a.value], unit=a.unit) for a in self.axes]
        cols.append(Column(data=[self.data], name='data'))
        t = Table(cols)
        return t

    def __str__(self):
        return str(self.to_table())

    def find_node(self, **kwargs):
        """Find nearest node

        Parameters
        ----------
        kwargs : dict
            Search values
        """
        for key in kwargs.keys():
            if key not in self.axis_names:
                raise ValueError('No axis for key {}'.format(key))

        for name, val in zip(self.axis_names, self.axes):
            kwargs.setdefault(name, val.nodes)

        nodes = list()
        for a in self.axes:
            value = kwargs[a.name]
            nodes.append(a.find_node(value))

        return nodes

    def evaluate(self, **kwargs):
        """Evaluate NDData Array

        No interpolation
        """
        idx = self.find_node(**kwargs)
        data = self.data
        for i in np.arange(self.dim):
            data = np.take(data, idx[i], axis = i)

        return data
        
    def plot_image(self, ax=None, plot_kwargs = {}, **kwargs):
        """Plot image

        Only avalable for 2D (after slicing)
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        data = self.evaluate(**kwargs)
        if len(data.squeeze().shape) != 2:
            raise ValueError('Data has shape {} after slicing. '
                             'Need 2d data for image plot'.format(data.shape))
    
        
        ax.set_xlabel('{} [{}]'.format(self.axes[0].name, self.axes[0].unit))
        ax.set_ylabel('{} [{}]'.format(self.axes[1].name, self.axes[1].unit))
        ax.imshow(data.transpose(), origin='lower', **plot_kwargs)

    def plot_profile(self, axis, ax=None, **kwargs):
        """Show data as function of one axis"""

        raise NotImplementedError

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        ax_ind = self.get_axis_index(axis)
        kwargs.setdefault(axis, self.axes[ax_ind])

        x = kwargs.pop(axis)
        

        y = self.evaluate(**kwargs)

        


class DataAxis(Quantity):
    def __new__(cls, energy, unit=None, dtype=None, copy=True, name=None):
        self = super(DataAxis, cls).__new__(cls, energy, unit,
                                            dtype=dtype, copy=copy)

        self.name = name
        return self

    def __array_finalize__(self, obj):
        super(DataAxis, self).__array_finalize__(obj)

    def find_node(self, val):
        """Find next node"""
        val = Quantity(val)

        if not val.unit.is_equivalent(self.unit):
            raise ValueError('Units {} and {} do not match'.format(
                val.unit, self.unit))

        val = val.to(self.unit)
        val = np.atleast_1d(val)
        x1 = np.array([val] * self.nbins).transpose()
        x2 = np.array([self.nodes] * len(val))
        temp = np.abs(x1 - x2)
        idx = np.argmin(temp, axis=1)
        return idx

    @property
    def nbins(self):
        return self.size

    @property
    def nodes(self):
        return self


class BinnedDataAxis(DataAxis):
    @classmethod
    def linspace(cls, min, max, nbins, unit=None):
        if unit is None:
            raise NotImplementedError

        data = np.linspace(min, max, nbins+1)
        unit = Unit(unit)
        return cls(data, unit)

    @property
    def nbins(self):
        return self.size - 1

    @property
    def nodes(self):
        return self.lin_center()

    def lin_center(self):
        """Linear bin centers"""
        return DataAxis(self[:-1] + self[1:]) / 2

