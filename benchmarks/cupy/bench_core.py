from benchmarks import BenchmarkBase
from benchmarks.utils import sync
from benchmarks.utils import parameterize

import cupy
import numpy


@sync
@parameterize([
    ('dtype', ['float32', 'complex128']),
    ('ndim', [0, 1, 2, 5, 8]),
    ('in_order', ['C', 'F']),
    ('out_order', ['C', 'F']),
])
class Array(BenchmarkBase):
    def setup(self, dtype, ndim, in_order, out_order):
        self.x = numpy.zeros((10,) * ndim, dtype=dtype, order=in_order)

    def time_array_from_numpy(self, dtype, ndim, in_order, out_order):
        cupy.array(self.x, order=out_order)
