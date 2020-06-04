import torch
from torch import nn
from torch.nn import functional as F


# Convolutional network with dropout
class ConvDropoutNet(nn.Module):
    def __init__(self):
        super(ConvDropoutNet, self).__init__()
        self.conv1_l = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2_l = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_l_drop = nn.Dropout2d()

        self.conv1_r = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2_r = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_r_drop = nn.Dropout2d()

        self.fc1_class_l = nn.Linear(128, 64)
        self.fc2_class_l = nn.Linear(64, 10)

        self.fc1_class_r = nn.Linear(128, 64)
        self.fc2_class_r = nn.Linear(64, 10)

        self.fc1_target = nn.Linear(256, 64)
        self.fc2_target = nn.Linear(64, 32)
        self.fc3_target = nn.Linear(32 + 2 * 10, 32)
        self.fc4_target = nn.Linear(32, 2)

    def forward(self, x, weight_sharing):
        x1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        x2 = x[:, 1, :, :].view(-1, 1, 14, 14)

        x1 = F.leaky_relu(F.max_pool2d(self.conv1_l(x1), 2))
        x1 = F.leaky_relu(F.max_pool2d(self.conv2_l_drop(self.conv2_l(x1)), 2))
        if weight_sharing:
            x2 = F.leaky_relu(F.max_pool2d(self.conv1_l(x2), 2))
            x2 = F.leaky_relu(F.max_pool2d(self.conv2_l_drop(self.conv2_l(x2)), 2))
        else:
            x2 = F.leaky_relu(F.max_pool2d(self.conv1_r(x2), 2))
            x2 = F.leaky_relu(F.max_pool2d(self.conv2_r_drop(self.conv2_r(x2)), 2))

        x = torch.cat((x1, x2), 1)

        x = x.view(x.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x1 = F.leaky_relu(self.fc1_class_l(x1))
        x1 = F.dropout(x1, training=self.training)
        x1 = self.fc2_class_l(x1)
        if weight_sharing:
            x2 = F.leaky_relu(self.fc1_class_l(x2))
            x2 = F.dropout(x2, training=self.training)
            x2 = self.fc2_class_l(x2)
        else:
            x2 = F.leaky_relu(self.fc1_class_r(x2))
            x2 = F.dropout(x2, training=self.training)
            x2 = self.fc2_class_r(x2)

        x = F.leaky_relu(self.fc1_target(x))
        x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.fc2_target(x))
        x = F.dropout(x, training=self.training)

        x = torch.cat((x, x1, x2), 1)
        x = F.leaky_relu(self.fc3_target(x))
        x = F.dropout(x, training=self.training)
        x = self.fc4_target(x)

        return x, x1, x2
