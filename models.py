import torch
from torch import nn


def crop(image, new_shape):
    """Function for cropping an image tensor: given an image tensor  and the new shape

    Args:
        image (tensor): image tensor of shape (batch_size, num_channels, height, width)
        new_shape (tensor.Size): the new shape
    """
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    start_height = middle_height - round(new_shape[2] / 2)
    end_height = start_height + new_shape[2]
    start_width = middle_width - round(new_shape[3] / 2)
    end_width = start_width + new_shape[3]
    cropped_image = image[:, :, start_height:end_height, start_width:end_width]
    return cropped_image


class ContractingBlock(nn.Module):
    """ContractingBlock class
        Performs Two convolutions followed by a maxpool operation
    Args:
        input_channels (int): the number of channels to expect from a given input
    """
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_bn = use_bn
        if self.use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()

    def forward(self, x):
        """Method for completing a forward pass of ContractingBlock

        Args:
            x (tensor): Image tensor
        """
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x) 
        x = self.maxpool(x) 
        return x      


class ExpandingBlock(nn.Module):
    """ExpandingBlock Class
        Performs an upsampling, a convolution, a concat of its two inputs, followed by two more convolutions with optional droput
    Args:
        input_channels (int): the number of input channels to expect from a given input
    """
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        self.activation = nn.ReLU()
        self.use_bn = use_bn
        if self.use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()

    def forward(self, x, skip_con_x):
        """Method for completing a forward pass of the ExpandingBlock:

        Args:
            x (tensor): iamge tensor of shape (batch_size, num_channels, height, width)
            skip_con_x (tensor): image tensor from the contracting path (from the ooposing block of x)
        """
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x
    

class FeatureMapBlock(nn.Module):
    """FeatureMapBlock Class 
        maps each pixel to a pixel with the corresponding number of output dimensions using a 1x1 conv

    Args:
        input_channels (int): the number of input channels to expect from a given input
        output_channels (int): the number of output channels to expect from a given output
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        """Method for completing a forward pass of FeatureMapBlock
            Given an image tensor, returns it mapped to the desired number of channels
        Args:
            x (tensor): tensor image of shape (batch_size, num_channels, height, width)
        """
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet  Class 
        A series of 4 contracting blocks followed by 4 expanding blocks to transform an input image 
        into the corresponding paired image, with an upfeature layter at the start, and a downfeature layer at the end

    Args:
        input_channels (int): the number of channels channels to expect from a given input
        output_channels (int): the number of channels to expect from a given output
    """
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super().__init__()

        print(f"The input channels parameter to UNet is: {input_channels}")
        print(f"The output channels parameter to UNet is: {output_channels}")

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.contract5 = ContractingBlock(hidden_channels * 16)
        self.contract6 = ContractingBlock(hidden_channels * 32)
        self.expand1 = ExpandingBlock(hidden_channels * 64)
        self.expand2 = ExpandingBlock(hidden_channels * 32)
        self.expand3 = ExpandingBlock(hidden_channels * 16)
        self.expand4 = ExpandingBlock(hidden_channels * 8)
        self.expand5 = ExpandingBlock(hidden_channels * 4)
        self.expand6 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Method for completing a forward pass of the UNet model:
            Given an image tensor, passes it through UNet and returns the output

        Args:
            x (tensor): The image tensor of shape (batch_size, num_channels, height, width)
        """
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand1(x6, x5)
        x8 = self.expand2(x7, x4)
        x9 = self.expand3(x8, x3)
        x10 = self.expand4(x9, x2)
        x11 = self.expand5(x10, x1)
        x12 = self.expand6(x11, x0)
        xn = self.downfeature(x12)
        return self.sigmoid(xn)


class Discriminator(nn.Module):
    """Discriminator Class
        Structured like the contracting path of the U-Net, the discriminator will output a matrix
        of values classifying corresponding portions of the image as real or fake
    Args:
        input_channels (int): The number of input channels
        hidden_channels (int): The inital number of discriminator convolutional filters
    """
    def __init__(self, input_channels, hidden_channels=8):
        super().__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    """Returns the loss of the generator given inputs.

    Args:
        gen (UNet): the generator, takes the condition and returns the potential images
        disc (Discriminator): takes images and the condition and returns real/fake predictions
        real (tensor): the real imags (maps), to be used to evalueate the reconstruction
        condition (tensor): the source images (e.g. satellite imagery) which is used to produce the real iamges
        adv_criterion (float): the adversarial loss fucntion. takes the discrminator predictions and the true labels
                            and returns the adversarial loss (aim to minimize)
        recon_criterion (float): the reconstruction loss fucntion; takes the generator outputs and the real images and 
                            returns a reconstruction loss (aim to minimize)
        lambda_recon (float): the degree to which the reconstuction loss should be weighted in the total loss sum
    """
    fake = gen(condition)
    disc_fake_pred = disc(fake,  condition)
    adv_loss = adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    recon_loss = recon_criterion(real, fake)
    gen_loss = adv_loss + (recon_loss * lambda_recon)
    return gen_loss
