import torch.nn as nn

from .utils import BasicEncoder, UpdateModule


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(out_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(out_dim=256, norm_fn='none')
        self.update = UpdateModule()

    @staticmethod
    def _apply_module(module, inputs):
        inputs_b = tuple(x.unsqueze(0) for x in inputs)
        outputs_b = module(*inputs_b)

        if isinstance(outputs_b, tuple):
            return tuple(o.squeeze(0) for o in outputs_b)
        return outputs_b.squeeze(0)
    

    def apply_fnet(self, *inputs):
        return DroidNet._apply_module(
            self.fnet, inputs
        )


    def apply_cnet(self, *inputs):
        return DroidNet._apply_module(
            self.cnet, inputs
        )

    def apply_update(self, *inputs):
        return DroidNet._apply_module(
            self.update, inputs
        )
    