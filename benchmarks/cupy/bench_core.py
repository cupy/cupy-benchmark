from benchmarks.numpy.common import Benchmark
from benchmarks.utils import sync
from benchmarks.utils import parameterize

import cupy
import numpy


@sync
@parameterize([
    ('ndim', [1, 2, 5, 8]),
    ('in_order', ['C', 'F']),
    ('out_order', ['C', 'F']),
])
class Array(Benchmark):
    def setup(self, ndim, in_order, out_order):
        self.x = numpy.zeros((10,) * ndim, order=in_order)

    def time_array_from_numpy(self, ndim, in_order, out_order):
        cupy.array(self.x, order=out_order)
