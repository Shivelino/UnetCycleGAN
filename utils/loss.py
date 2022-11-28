from torch.nn import MSELoss
from torch.nn import L1Loss


class GanLoss(MSELoss):
    def __init__(self):
        super(GanLoss).__init__(self)


class CycleLoss(L1Loss):
    def __init__(self):
        super(CycleLoss).__init__(self)


class IdentityLoss(L1Loss):
    def __init__(self):
        super(IdentityLoss).__init__(self)
