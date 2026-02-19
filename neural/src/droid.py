import torch
import torch.nn as nn
from collections import OrderedDict

from .utils import BasicEncoder, UpdateModule


class DroidNet(nn.Module):
    def __init__(self, cfg: dict):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(out_dim=128, norm_fn="instance")
        self.cnet = BasicEncoder(out_dim=256, norm_fn="none")
        self.update = UpdateModule().half()

        self._load_weights(cfg["weights"]["droid"])

    def _load_weights(self, fp: str):
        state_dict = OrderedDict(
            [(k.replace("module.", ""), v) for (k, v) in torch.load(fp).items()]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.load_state_dict(state_dict)

    @staticmethod
    def _apply_module(module, inputs):
        inputs_b = tuple(x.unsqueeze(0) for x in inputs)
        outputs_b = module(*inputs_b)

        if isinstance(outputs_b, tuple):
            return tuple(o.squeeze(0) for o in outputs_b)
        return outputs_b.squeeze(0)

    def apply_fnet(self, *inputs):
        return DroidNet._apply_module(self.fnet, inputs)

    def apply_cnet(self, *inputs):
        x = DroidNet._apply_module(self.cnet, inputs)
        return x.split([128, 128], dim=1)  # net, inp

    def apply_update(self, *inputs):
        outputs_b = DroidNet._apply_module(self.update, tuple(x.half() for x in inputs))
        if isinstance(outputs_b, tuple):
            return tuple(o.float() for o in outputs_b)
        return outputs_b.float()
