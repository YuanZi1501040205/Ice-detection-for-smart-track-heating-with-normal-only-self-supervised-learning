"""models_zoo_alpha.py: File to stores Neural Network architectures for AGT_FWI_PROJECT2020"""

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "3.0.0"


def models(name_model):
    """ This class defines neural networks modeling
        Set the necessary parameters for different models
        input:
            name_model is a string that specifies the neural network modeling approach
        Return:
            model: pytorch
            loss_function
            optimizer
    """
    import torch
    import torch.nn as nn

# for viking
    # Frequency domain
    if name_model == "CNN11_ResNet3":
        from Functions import residual_block_
        class resnet(nn.Module):
            def __init__(self, in_channel, out_channel, verbose=False):
                super(resnet, self).__init__()
                self.verbose = verbose

                self.block1 = nn.Sequential(
                    nn.Conv2d(in_channel, 32, 3, 1),
                    nn.PReLU(),
                    nn.Conv2d(32, 64, 3, 1),
                )

                self.block2 = nn.Sequential(

                    residual_block_(64, 64),
                    residual_block_(64, 64)
                )

                self.block3 = nn.Sequential(
                    residual_block_(64, 128, False),
                    residual_block_(128, 128)
                )

                self.block4 = nn.Sequential(
                    residual_block_(128, 128),
                    residual_block_(128, 128)
                )

                self.block5 = nn.Sequential(
                    residual_block_(128, 256, False),
                    residual_block_(256, 256)
                )

                self.block6 = nn.Sequential(
                    residual_block_(256, 256),
                    residual_block_(256, 256)
                )

                self.block7 = nn.Sequential(
                    residual_block_(256, 512, False),
                    residual_block_(512, 512)
                )

                self.block8 = nn.Sequential(
                    residual_block_(512, 512),
                    residual_block_(512, 512)
                )

                self.block9 = nn.Sequential(
                    residual_block_(512, 1024, False),
                    residual_block_(1024, 1024),
                )

                self.output = nn.Linear(30720, out_channel)

            def forward(self, x):
                x = self.block1(x)
                if self.verbose:
                    print('block 1 output: {}'.format(x.shape))
                x = self.block2(x)
                if self.verbose:
                    print('block 2 output: {}'.format(x.shape))
                x = self.block3(x)
                if self.verbose:
                    print('block 3 output: {}'.format(x.shape))
                x = self.block4(x)
                if self.verbose:
                    print('block 4 output: {}'.format(x.shape))
                x = self.block5(x)
                if self.verbose:
                    print('block 5 output: {}'.format(x.shape))
                x = self.block6(x)
                if self.verbose:
                    print('block 6 output: {}'.format(x.shape))
                x = self.block7(x)
                if self.verbose:
                    print('block 7 output: {}'.format(x.shape))
                x = self.block8(x)
                if self.verbose:
                    print('block 8 output: {}'.format(x.shape))
                x = self.block9(x)
                if self.verbose:
                    print('block 9 output: {}'.format(x.shape))
                x = x.view(x.shape[0], -1)
                if self.verbose:
                    print('reshape output: {}'.format(x.shape))
                x = self.output(x)
                return x

        model = resnet(1, 110)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.01, lr=0.0001, momentum=0.9)

    # Time Domain
    elif name_model == "CNN19_ResUNet1":  # !!! THis is for Time Domain Model
        #  model build
        from Functions import DoubleConv
        from Functions import ResDown
        from Functions import Up
        from Functions import OutConv


        class UNet(nn.Module):
            def __init__(self, input_channels,  bilinear=True):
                super(UNet, self).__init__()
                self.input_channels = input_channels
                self.bilinear = bilinear

                # self.in_attention = SALayer()
                self.inc = DoubleConv(input_channels, 64)
                self.down1 = ResDown(64, 128)
                self.down2 = ResDown(128, 256)
                self.down3 = ResDown(256, 512)
                factor = 2 if bilinear else 1
                self.down4 = ResDown(512, 1024 // factor)
                self.up1 = Up(1024, 512 // factor, bilinear)
                self.up2 = Up(512, 256, bilinear)
                self.up3 = Up(384, 128, bilinear)
                self.up4 = Up(192, 64, bilinear)
                self.outc = OutConv(64, 1)


            def forward(self, x):
                # atten_x = self.in_attention(x)
                #print('x1: ', x.shape)
                x1 = self.inc(x)
                #print('x1: ', x1.shape)
                x2 = self.down1(x1)
                #print('x2: ', x2.shape)
                x3 = self.down2(x2)
                #print('x3: ', x3.shape)
                x4 = self.down3(x3)
                #print('x4: ', x4.shape)
                x5 = self.down4(x4)
                x = self.up1(x5, x4)
                #print('up1: ', x.shape)
                x = self.up2(x, x3)
                #print('up2: ', x.shape)
                x = self.up3(x, x2)
                #print('up3: ', x.shape)
                x = self.up4(x, x1)
                #print('up4: ', x.shape)
                x = self.outc(x)
                #print('outc: ', x.shape)
                return x

        model = UNet(input_channels=1, bilinear=True)
        # case we will use self define Mean Squared Error (MSE) as  ur loss function.
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0001, lr=0.02, momentum=0.9)

    elif name_model == "CNN20_AttenResUNet":  # !!! THis is for Time Domain Model
        #  model build
        import torch.nn.functional as F
        from Functions import DoubleConv
        from Functions import Down
        from Functions import ResDown
        from Functions import Up
        from Functions import OutConv
        from Functions import SpatialAttention
        from Functions import AttenResDown


        class UNet(nn.Module):
            def __init__(self, input_channels, bilinear=True):
                super(UNet, self).__init__()
                self.input_channels = input_channels
                self.bilinear = bilinear

                # self.in_attention = SALayer()
                self.inc = DoubleConv(input_channels, 64)
                self.down1 = AttenResDown(64, 128)
                self.down2 = ResDown(128, 256)
                self.down3 = ResDown(256, 512)
                self.down4 = ResDown(512, 1024)
                self.down5 = ResDown(1024, 2048)
                factor = 2 if bilinear else 1
                self.down6 = ResDown(2048, 4096 // factor)
                self.up1 = Up(4096, 2048 // factor, bilinear)
                self.up2 = Up(2048, 1024 // factor, bilinear)
                self.up3 = Up(1024, 512 // factor, bilinear)
                self.up4 = Up(512, 256, bilinear)
                self.up5 = Up(256, 128, bilinear)
                self.up6 = Up(128, 64, bilinear)
                self.outc = OutConv(64, 1)

            def forward(self, x):
                # atten_x = self.in_attention(x)
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                x6 = self.down5(x5)
                x7 = self.down6(x6)
                x = self.up1(x7, x6)
                x = self.up2(x, x5)
                x = self.up3(x, x4)
                x = self.up4(x, x3)
                x = self.up5(x, x2)
                x = self.up6(x, x1)
                x = self.outc(x)
                x = x.view(x.shape[0], -1)
                return x

        model = UNet(input_channels=1, bilinear=True)
        # case we will use self define Mean Squared Error (MSE) as  ur loss function.
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0001, lr=0.01, momentum=0.9)

    return model, loss_func, optimizer
