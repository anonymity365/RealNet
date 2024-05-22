import torch
from .feature_transformer import get_TRR
def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Classifier(torch.nn.Module):
    def __init__(self, in_channelss, n_layers=2, hidden=None):
        super(Classifier, self).__init__()

        _hidden = in_channelss if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_channelss if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Adaptor(torch.nn.Module):

    def __init__(self, in_channels, out_channels=None, n_layers=1):
        super(Adaptor, self).__init__()

        if out_channels is None:
            out_channels = out_channels
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_channels if i == 0 else _out
            _out = out_channels
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
        self.apply(init_weight)

    def forward(self, x):

        x = self.layers(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=1):
        super(Discriminator, self).__init__()
        self.adapter = Adaptor(in_channels, out_channels, n_layers)
        self.classifier = Classifier(out_channels)

    def forward(self, x, device):
        x = get_TRR(x, device).detach()
        x = self.adapter(x)
        return self.classifier(x).sigmoid()

    def load(self, ckpt_path):
        best = torch.load(ckpt_path)
        self.adapter.load_state_dict(best['project'])
        self.classifier.load_state_dict(best['discriminator'])