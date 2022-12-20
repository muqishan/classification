from torch.nn import Module
from torch import nn

class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.features = self._make_layers()
        self.fc = nn.Linear(256,2)

    def _make_layers(self):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32,eps=0.00001,momentum=0.99),
                        nn.ELU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        layers += [nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64,eps=0.00001,momentum=0.99),
                        nn.ELU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [nn.Conv2d(64, 16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16,eps=0.00001,momentum=0.99),
                        nn.ELU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.features(x)
        out = out.view(x.shape[0],-1) #展平
        out = self.fc(out)
        return out