import torch
from torch import nn


class UBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inner_channels: int,
                 submodule: nn.Module = None, innermost: bool = False, outermost: bool = False) -> None:
        super().__init__()

        self.innermost = innermost
        self.outermost = outermost

        if in_channels is None:
            in_channels = out_channels

        if innermost:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 4,
                          stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_channels, out_channels, 4,
                                   stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True),
            )
        elif outermost:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 4,
                          stride=2, padding=1),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(True),
                submodule,
                nn.ConvTranspose2d(inner_channels * 2, out_channels, 4,
                                   stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 4,
                          stride=2, padding=1, bias=True),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(True),
                submodule,
                nn.ConvTranspose2d(inner_channels * 2, out_channels, 4,
                                   stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True),
            )

    def forward(self, x: torch.tensor):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat((x, self.model(x)), 1)


class AutoSteganographer(nn.Module):
    merger: nn.Sequential
    revealer: nn.Sequential

    def __init__(self, channels: int = 16) -> None:
        super().__init__()

        ublock = UBlock(None, channels * 8, channels * 8, innermost=True)
        ublock = UBlock(None, channels * 4, channels * 8, submodule=ublock)
        ublock = UBlock(None, channels * 2, channels * 4, submodule=ublock)
        ublock = UBlock(None, channels, channels * 2, submodule=ublock)
        ublock = UBlock(6, 3, channels, submodule=ublock, outermost=True)

        self.merger = ublock

        self.revealer = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(True),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.ReLU(True),
            nn.Conv2d(channels * 4, channels * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.ReLU(True),
            nn.Conv2d(channels * 4, channels * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(True),
            nn.Conv2d(channels * 2, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 3, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def merge(self, original: torch.tensor, hidden: torch.tensor):
        x = torch.cat((original, hidden), 1)
        x = self.merger(x)
        return x

    def forward(self, original: torch.tensor, hidden: torch.tensor):
        x = self.merge(original, hidden)
        y = self.revealer(x)
        return x, y
