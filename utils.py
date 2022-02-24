import torch
import torch.nn as nn


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()


class SingleChannelModel():
    """ reshapes images to rgb before classification
        i.e. [N, 1, H, W x 3] -> [N, 3, H, W]
    """
    def __init__(self, model):
        if isinstance(model, nn.Module):
            assert not model.training
        self.model = model

    def __call__(self, x):
        return self.model(x.view(x.shape[0], 3, x.shape[2], x.shape[3] // 3))

