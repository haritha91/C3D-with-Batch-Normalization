import torch.nn as nn

class C3D_BN(nn.Module):
    """
    The C3D network as described in [1]
    Batch Normalization as described in [2]

    """

    def __init__(self):
        super(C3D_BN, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1_bn = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2_bn = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3a_bn = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b_bn = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4a_bn = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b_bn = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5a_bn = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b_bn = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 8)
        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.conv1_bn(self.conv1(x)))
        h = self.pool1(h)

        h = self.relu(self.conv2_bn(self.conv2(h)))
        h = self.pool2(h)

        h = self.relu(self.conv3a_bn(self.conv3a(h)))
        h = self.relu(self.conv3b_bn(self.conv3b(h)))
        h = self.pool3(h)

        h = self.relu(self.conv4a_bn(self.conv4a(h)))
        h = self.relu(self.conv4b_bn(self.conv4b(h)))
        h = self.pool4(h)

        h = self.relu(self.conv5a_bn(self.conv5a(h)))
        h = self.relu(self.conv5b_bn(self.conv5b(h)))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.relu(self.fc7(h))
        h = self.fc8(h)
        return h

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
[2] Ioffe, Surgey, et al. "Batch Normalization: Accelerating deep network training 
by reducing internal covariate shift."
arXiv:1502.03167v2 [cs.LG] 13 Feb 2015
"""