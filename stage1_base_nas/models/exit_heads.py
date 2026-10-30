import torch
import torch.nn as nn

class ExitHead_A1(nn.Module):
    """A1: GAP(1x1) -> FC(C, 10)"""
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ExitHead_A2(nn.Module):
    """A2: AvgPool(2x2) -> Flatten -> FC(4C, 10)"""
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(in_channels * 4, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ExitHead_A3(nn.Module):
    """A3: AvgPool(4x4) -> Flatten -> FC(16C, 256) -> ReLU -> FC(256, 10)"""
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(in_channels * 16, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ExitHead_A4(nn.Module):
    """A4: Conv1x1(C->C/4) + BN + ReLU -> AvgPool(2x2) -> FC(C, 128) -> ReLU -> FC(128, 10)"""
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        out_c = max(16, in_channels // 4)
        self.conv = nn.Conv2d(in_channels, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(out_c * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ExitHead_A5(nn.Module):
    """A5: DWConv3x3 + PWConv(C->C/2) + BN + ReLU -> GAP -> FC(C/2, 10)"""
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        out_c = max(16, in_channels // 2)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pwconv = nn.Conv2d(in_channels, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_c, num_classes)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.relu(self.bn(self.pwconv(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


HEAD_BUILDERS = {
    "A1": ExitHead_A1,
    "A2": ExitHead_A2,
    "A3": ExitHead_A3,
    "A4": ExitHead_A4,
    "A5": ExitHead_A5,
}
