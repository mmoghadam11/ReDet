from .rbbox_base_sampler  import RbboxBaseSampler
from ..assign_sampling import build_sampler


class CombinedRbboxSampler(RbboxBaseSampler):

    def __init__(self, pos_sampler, neg_sampler, **kwargs):
        super(CombinedRbboxSampler, self).__init__(**kwargs)
        self.pos_sampler = build_sampler(pos_sampler, **kwargs)
        self.neg_sampler = build_sampler(neg_sampler, **kwargs)

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError
