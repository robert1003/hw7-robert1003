import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentNet(nn.Module):
    
    def _cnn_block(self, in_channels, out_channels, kernek_size, groups=1, have_pool=True, break_down=True):
        if break_down:
            layers = [
                nn.Conv2d(in_channels, in_channels, kernek_size, 1, 1, groups=groups),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(),
                nn.Conv2d(in_channels, out_channels, 1)
            ]
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernek_size, 1, 1, groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6()
            ]
        if have_pool:
            layers.append(nn.MaxPool2d(2, 2, 0))
        return nn.Sequential(*layers)

    def __init__(self, base=16, width_mult=1):
        super().__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        bandwidth = [base * m for m in multiplier]

        for i in range(3, len(bandwidth)):
            bandwidth[i] = int(bandwidth[i] * width_mult)
        self.cnn = nn.Sequential(
            self._cnn_block(3, bandwidth[0], 3, break_down=False),
            self._cnn_block(bandwidth[0], bandwidth[1], 3, groups=bandwidth[0]),
            self._cnn_block(bandwidth[1], bandwidth[2], 3, groups=bandwidth[1]),
            self._cnn_block(bandwidth[2], bandwidth[3], 3, groups=bandwidth[2]),
            self._cnn_block(bandwidth[3], bandwidth[4], 3, groups=bandwidth[3], have_pool=False),
            self._cnn_block(bandwidth[4], bandwidth[5], 3, groups=bandwidth[4], have_pool=False),
            self._cnn_block(bandwidth[5], bandwidth[6], 3, groups=bandwidth[5], have_pool=False),
            self._cnn_block(bandwidth[6], bandwidth[7], 3, groups=bandwidth[6], have_pool=False),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(bandwidth[7] * 4, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
