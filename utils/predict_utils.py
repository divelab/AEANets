from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter


from .basic_utils import normalize_mi_ma, axes_dict
import warnings
import numpy as np


def _raise(e):
    raise e

    
class PercentileNormalizer(object):

    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=np.float32, **kwargs):

        (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100) or _raise(ValueError())
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs

    def before(self, img, axes):

        len(axes) == img.ndim or _raise(ValueError())
        channel = axes_dict(axes)['C']
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img,self.pmin,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(img,self.pmax,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        return normalize_mi_ma(img, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def after(self, img):

        self.do_after or _raise(ValueError())
        alpha = self.ma - self.mi
        beta  = self.mi
        return ( alpha*img+beta ).astype(self.dtype,copy=False)

    def do_after(self):
        
        return self._do_after
    
    
    
class PadAndCropResizer(object):

    def __init__(self, mode='reflect', **kwargs):

        self.mode = mode
        self.kwargs = kwargs
        
    def _normalize_exclude(self, exclude, n_dim):
        """Return normalized list of excluded axes."""
        if exclude is None:
            return []
        exclude_list = [exclude] if np.isscalar(exclude) else list(exclude)
        exclude_list = [d%n_dim for d in exclude_list]
        len(exclude_list) == len(np.unique(exclude_list)) or _raise(ValueError())
        all(( isinstance(d,int) and 0<=d<n_dim for d in exclude_list )) or _raise(ValueError())
        return exclude_list

    def before(self, x, div_n, exclude):

        def _split(v):
            a = v // 2
            return a, v-a
        exclude = self._normalize_exclude(exclude, x.ndim)
        self.pad = [_split((div_n-s%div_n)%div_n) if (i not in exclude) else (0,0) for i,s in enumerate(x.shape)]
        x_pad = np.pad(x, self.pad, mode=self.mode, **self.kwargs)
        for i in exclude:
            del self.pad[i]
        return x_pad

    def after(self, x, exclude):

        pads = self.pad[:len(x.shape)]
        crop = [slice(p[0], -p[1] if p[1]>0 else None) for p in self.pad]
        for i in self._normalize_exclude(exclude, x.ndim):
            crop.insert(i,slice(None))
        len(crop) == x.ndim or _raise(ValueError())
        return x[tuple(crop)]

    
# def tile_iterator(x,axis,n_tiles,block_size,n_block_overlap):

#     n = x.shape[axis]

#     n % block_size == 0 or _raise(ValueError("'x' must be evenly divisible by 'block_size' along 'axis'"))
#     n_blocks = n // block_size

#     n_tiles_valid = int(np.clip(n_tiles,1,n_blocks))
#     if n_tiles != n_tiles_valid:
#         warnings.warn("invalid value (%d) for 'n_tiles', changing to %d" % (n_tiles,n_tiles_valid))
#         n_tiles = n_tiles_valid

#     s = n_blocks // n_tiles # tile size
#     r = n_blocks %  n_tiles # blocks remainder
#     assert n_tiles * s + r == n_blocks

#     # list of sizes for each tile
#     tile_sizes = s*np.ones(n_tiles,int)
#     # distribute remaning blocks to tiles at beginning and end
#     if r > 0:
#         tile_sizes[:r//2]      += 1
#         tile_sizes[-(r-r//2):] += 1

#     off = [(n_block_overlap if i > 0 else 0, n_block_overlap if i < n_tiles-1 else 0) for i in range(n_tiles)]

#     def to_slice(t):
#         sl = [slice(None) for _ in x.shape]
#         sl[axis] = slice(
#             t[0]*block_size,
#             t[1]*block_size if t[1]!=0 else None)
#         return tuple(sl)

#     start = 0
#     for i in range(n_tiles):
#         off_pre, off_post = off[i]

#         if start-off_pre < 0:
#             off_pre = start

#         if start+tile_sizes[i]+off_post > n_blocks:
#             off_post = n_blocks-start-tile_sizes[i]

#         tile_in   = (start-off_pre,start+tile_sizes[i]+off_post)  # src in input image     / tile
#         tile_out  = (start,start+tile_sizes[i])                   # dst in output image    / s_dst
#         tile_crop = (off_pre,-off_post)                           # crop of src for output / s_src

#         yield x[to_slice(tile_in)], to_slice(tile_crop), to_slice(tile_out)
#         start += tile_sizes[i]

def tile_iterator(x, axis, n_tiles, block_size, overlap, enqueue=False):
    L = x.shape[axis] // block_size
    tile_size = int(np.ceil((L + (n_tiles - 1)*overlap)/n_tiles))
    
    def to_slice(t):
        sl = [slice(None) for _ in x.shape]
        sl[axis] = slice(
            t[0]*block_size,
            t[1]*block_size if t[1]!=0 else None)
        return tuple(sl)
    
    start_x= 0
    for i in range(n_tiles):
        if start_x == 0:
            start_tile = 0
        else:
            start_tile =  overlap - (overlap//2)

        end_x = start_x + tile_size

        
        if end_x >= L:
            start_x -= end_x - L
            start_tile += end_x - L
            end_x = L
            end_tile = 0
        else:
            end_tile = - (overlap//2)
        
        if enqueue:
            tile_in = (start_x, end_x)
            yield x[to_slice(tile_in)]
        else:
            tile_in = (start_x, end_x)
            tile_out = (start_x+start_tile, end_x+end_tile)
            tile_crop = (start_tile, end_tile)
            yield (x[to_slice(tile_in)], to_slice(tile_crop), to_slice(tile_out))
        
        start_x += tile_size - overlap