import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, 
                 kernel_size:int, stride:int=1, padding:int=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()

        # Conv 7x7x64-s-2 + Maxpool
        self.conv1 = ConvBlock(3, 64, 7, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv 3x3x192 + Maxpool
        self.conv2 = ConvBlock(64, 192, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv Layers (128, 256, 256, 512) + Maxpool
        self.conv3 = ConvBlock(192, 128, 1, stride=1, padding=1)
        self.conv4 = ConvBlock(128, 256, 3, stride=1, padding=1)
        self.conv5 = ConvBlock(256, 256, 1, stride=1, padding=1)
        self.conv6 = ConvBlock(256, 512, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Conv Layers x4 (256, 512) + (512, 1024) + Maxpool
        self.conv7_14 = nn.ModuleList([
                        nn.Sequential(
                            ConvBlock(512, 256, stride=1, padding = 1),
                            ConvBlock(256, 512, stride=1, padding = 1))
                         for _ in range(4)               
                        ])
        
        self.conv15 = ConvBlock(512, 512, 1, stride=1, padding=1)
        self.conv16 = ConvBlock(512, 1024, 3, stride=1, padding=1)
        self.pool4  = nn.MaxPool2d(2, 2)

        # Conv Layers x2 (512, 1024) + (1024, 1024)
        self.conv17_20 = nn.ModuleList([
                         nn.Sequential(
                            ConvBlock(1024, 512, 1, stride=1, padding=1),
                            ConvBlock(512, 1024, 3, stride=1, padding=1))
                         for _ in range(2)
                        ])
        self.conv21 = ConvBlock(1024, 1024, 3, stride=1, padding=1)
        self.conv22 = ConvBlock(1024, 1024, 3, stride=2, padding=1)

        # Conv Layers finales
        self.conv23 = ConvBlock(1024, 1024, 3, stride=1, padding=1)
        self.conv24 = ConvBlock(1024, 1024, 3, stride=1, padding=1)

        # Fully Connected
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 7 * 7 * 30)  # S*S*(B*5+C) = 7*7*30

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(self.conv6(x))

        
        x = self.conv7_14(x)
        x = self.conv15(x)
        x = self.pool4(self.conv16(x))

        x = self.conv17_20(x)
        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels, 3, stride=stride,padding = 1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, stride=1,padding = 1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.lineal_projection = stride != 1 or in_channels != out_channels
        if self.lineal_projection:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,stride=stride,bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self,x, fmap_dict = None, prefix=''):

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.lineal_projection else x
        out_add = out + shortcut

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        out = torch.relu(out_add)

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out
        
        return out


class ResNet34(nn.Module):

    def __init__(self, num_classes = 2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(3, stride=2, padding=1))

        self.layer1 = nn.ModuleList([ResidualBlock(64, 64, stride=1) for i in range(3)])
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i==0 else 128, 128, stride=2 if i==0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5) # Probabilidad de que un elemento se haga cero
        self.fc = nn.Linear(512,num_classes)