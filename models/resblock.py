import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, nb_channels):
        super(ResBlock, self).__init__()

        self.network = nn.ModuleList([nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1, bias=False), \
                            nn.ReLU(inplace=True), nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1, bias=False)])
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        return x + self.network(x)
    
class ResBlocks(nn.Module):
    def __init__(self, depth, nb_channels):
        super(ResBlocks, self).__init__()

        self.network = nn.ModuleList([ResBlock(nb_channels) for _ in range(depth)])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x