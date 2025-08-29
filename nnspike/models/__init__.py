from .customized import SimpleNetClassification25
from .loss import MultiTaskLoss
from .nvidia import NvidiaModelMultiTask, NvidiaModelRegression

__all__ = ["SimpleNetClassification25", "NvidiaModelMultiTask", "NvidiaModelRegression", "MultiTaskLoss"]
