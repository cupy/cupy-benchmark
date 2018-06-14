import functools
import inspect

from benchmarks.numpy.common import Benchmark
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

import cupy


def fuse_shapes(*shape_args, **shapes_dict):
    def decorator(func):
        fused_func = cupy.fuse(func)
        @functools.wraps(func)
        def wrapper(self, fusion_mode):
            if fusion_mode == 'enabled':
                fused_func(*args)
            elif fusion_mode == 'disabled':
                func(*args)
            elif fusion_mode == 'compile':
                raise NotImplementedError
                # TODO(imanishi)
                # fused_func._memo = {}
                # fused_func.compile(*args)
            else:
                raise ValueError('Invalid parameter')

        uniform = cupy.random.uniform
        args = []
        if shape_args:
            assert len(shapes_dict) == 0
            for name in inspect.getargspec(func).args:
                args.append(uniform(0, 200, shape_args, cupy.float32))
        else:
            for name in inspect.getargspec(func).args:
                shape = shapes_dict[name]
                args.append(uniform(0, 200, shape, cupy.float32))
        # TODO (imanishi)
        # fused_func.compile(*args)
        fused_func(*args)
        return wrapper
    return decorator


@sync
@parameterize([('fusion_mode', ['enabled', 'disabled'])])
class Fusion(Benchmark):
    def setup(self, fusion_mode):
        pass

    @fuse_shapes(1000, 1000)
    def time_fusion_add_10_times(x):
        return x + x + x + x + x + x + x + x + x + x

    @fuse_shapes(1000, 1000)
    def time_fusion_loops(x):
        res = 0
        for i in range(300):
            res += x
        return res

    @fuse_shapes(1000, 1000)
    def time_fusion_lstm_grad_grad(
            c_prev, a, i, f, o, c, gc, gh, ggc_prev, gga, ggi, ggf, ggo,
            gc_prev, ga, gi, gf, go, gc_next, ggc, ggh):

        def _cupy_sigmoid(x):
            half = x.dtype.type(0.5)
            return cupy.tanh(x * half) * half + half

        def _grad_grad_sigmoid(x):
            return x * (1 - x) * (1 - 2 * x)

        def _grad_sigmoid(x):
            return x * (1 - x)

        def _grad_tanh(x):
            return 1 - x * x

        def _grad_grad_tanh(x, gx):
            return -2 * x * gx
        
        sig_o = _cupy_sigmoid(o)
        gsig_o = _grad_sigmoid(sig_o)
        ggsig_o = _grad_grad_sigmoid(sig_o)
        sig_i = _cupy_sigmoid(i)
        gsig_i = _grad_sigmoid(sig_i)
        ggsig_i = _grad_grad_sigmoid(sig_i)
        sig_f = _cupy_sigmoid(f)
        gsig_f = _grad_sigmoid(sig_f)
        ggsig_f = _grad_grad_sigmoid(sig_f)
        tanh_a = cupy.tanh(a)
        gtanh_a = _grad_tanh(tanh_a)
        ggtanh_a = _grad_grad_tanh(tanh_a, gtanh_a)
        tanh_c = cupy.tanh(c)
        gtanh_c = _grad_tanh(tanh_c)
        ggtanh_c = _grad_grad_tanh(tanh_c, gtanh_c)

        gc_bar = gh * sig_o * gtanh_c + gc

        gc_prev[:] = ggf * gc_bar * gsig_f
        ga[:] = (gga * sig_i * ggtanh_a + ggi * gtanh_a * gsig_i) * gc_bar
        gi[:] = (gga * gtanh_a * gsig_i + ggi * tanh_a * ggsig_i) * gc_bar
        gf[:] = (ggc_prev * (gh * sig_o * gtanh_c + gc) * gsig_f +
                 ggf * gc_bar * c_prev * ggsig_f)

        ggc[:] = (ggc_prev * sig_f +
                  gga * sig_i * gtanh_a +
                  ggi * tanh_a * gsig_i +
                  ggf * c_prev * gsig_f)

        dgc_do = gh * gsig_o * gtanh_c
        go[:] = ggc * dgc_do + ggo * gh * tanh_c * ggsig_o
        dgc_dc = gh * sig_o * ggtanh_c
        gc_next[:] = ggc * dgc_dc + ggo * gh * gtanh_c * gsig_o
        ggh[:] = ggc * sig_o * gtanh_c + ggo * tanh_c * gsig_o

        return gc_prev, ga, gi, gf, go, gc_next, ggc, ggh
