  #https://github.com/jbwang1997/OBBDetection/blob/4fd26a74f5e4a5aec9395b39a78b0faaaff7c07c/mmdet/ops/convex/convex_wrapper.py#L20

  
  from torch.autograd import Function
from . import convex_ext


class ConvexSortFunction(Function):

    @staticmethod
    def forward(ctx, pts, masks, circular):
        idx = convex_ext.convex_sort(pts, masks, circular)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_output):
        return ()

convex_sort_func = ConvexSortFunction.apply


def convex_sort(pts, masks, circular=True):
    return convex_sort_func(pts, masks, circular)
