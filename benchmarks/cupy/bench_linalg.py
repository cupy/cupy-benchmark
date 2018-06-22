from benchmarks import BenchmarkBase
from benchmarks.utils import sync

import cupy


@sync
class Eindot(BenchmarkBase):
    def setup(self):
        self.b = cupy.arange(240000.0).reshape(400, 600)
        self.sq1 = cupy.arange(160000).reshape(400, 400)
        self.sq2 = cupy.arange(360000).reshape(600, 600)

    def time_einsum_ii(self):
        cupy.einsum('ii', self.sq2)

    def time_einsum_ii_jj_ij_ij(self):
        cupy.einsum('ii,jj,ij->ij', self.sq1, self.sq2, self.b)
