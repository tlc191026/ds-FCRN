import torch
import torch.nn as nn

class convBlock1(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(outplace)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class convBlock2(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplace, outplace, kernel_size=kernel_size, bias=False)
        self.bn1 = nn.BatchNorm3d(outplace)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class SFCN(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        channel = [32, 64, 128, 256, 256, 256]

        self.channel = channel
        self.conv1 = convBlock1(inplace, channel[0])
        self.conv2 = convBlock1(channel[0], channel[1])
        self.conv3 = convBlock1(channel[1], channel[2])
        self.conv4 = convBlock1(channel[2], channel[3])
        self.conv5 = convBlock1(channel[3], channel[4])
        self.conv6 = convBlock2(channel[4], channel[5])
        self.mx = nn.MaxPool3d(kernel_size=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.mx(x)
        x = self.conv2(x)
        x = self.mx(x)
        x = self.conv3(x)
        x = self.mx(x)
        x = self.conv4(x)
        x = self.mx(x)
        x = self.conv5(x)
        x = self.mx(x)
        x = self.conv6(x)

        return x

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class VGG8(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        channel = [32, 64, 128, 256]

        self.channel = channel

        self.maxp = nn.MaxPool3d(2)

        self.conv11 = convBlock1(inplace, channel[0])
        self.conv12 = convBlock1(channel[0], channel[0])

        self.conv21 = convBlock1(channel[0], channel[1])
        self.conv22 = convBlock1(channel[1], channel[1])

        self.conv31 = convBlock1(channel[1], channel[2])
        self.conv32 = convBlock1(channel[2], channel[2])

        self.conv41 = convBlock1(channel[2], channel[3])
        self.conv42 = convBlock1(channel[3], channel[3])

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp(x)

        return x

class VGG16(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        channel = [32, 64, 128, 256, 256]

        self.channel = channel

        self.maxp = nn.MaxPool3d(2)

        self.conv11 = convBlock1(inplace, channel[0])
        self.conv12 = convBlock1(channel[0], channel[0])

        self.conv21 = convBlock1(channel[0], channel[1])
        self.conv22 = convBlock1(channel[1], channel[1])

        self.conv31 = convBlock1(channel[1], channel[2])
        self.conv32 = convBlock1(channel[2], channel[2])
        self.conv33 = convBlock1(channel[2], channel[2])

        self.conv41 = convBlock1(channel[2], channel[3])
        self.conv42 = convBlock1(channel[3], channel[3])
        self.conv43 = convBlock1(channel[3], channel[3])

        self.conv51 = convBlock1(channel[3], channel[3])
        self.conv52 = convBlock1(channel[3], channel[3])
        self.conv53 = convBlock1(channel[3], channel[3])

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.maxp(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        return x

class FCRN(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        channel = [32, 64, 128, 256]

        self.channel = channel

        self.maxp = nn.MaxPool3d(2)

        self.conv11 = convBlock1(inplace, channel[0])
        self.conv12 = convBlock1(channel[0], channel[0])

        self.conv21 = convBlock1(channel[0], channel[1])
        self.conv22 = convBlock1(channel[1], channel[1])

        self.conv31 = convBlock1(channel[1], channel[2])
        self.conv32 = convBlock1(channel[2], channel[2])

        self.conv41 = convBlock1(channel[2], channel[3])
        self.conv42 = convBlock1(channel[3], channel[3])

        self.conv5 = convBlock2(channel[3], channel[3])

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv12(x1)
        x = torch.add(x1, x2)
        x = self.maxp(x)

        x1 = self.conv21(x)
        x2 = self.conv22(x1)
        x = torch.add(x1, x2)
        x = self.maxp(x)

        x1 = self.conv31(x)
        x2 = self.conv32(x1)
        x = torch.add(x1, x2)
        x = self.maxp(x)

        x1 = self.conv41(x)
        x2 = self.conv42(x1)
        x = torch.add(x1, x2)
        x = self.maxp(x)

        x = self.conv5(x)

        return x