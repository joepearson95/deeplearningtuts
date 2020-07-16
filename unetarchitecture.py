import torch
import torch.nn as nn


# double convolution with two 3x3 convolution layers
# followed by a ReLU activation function
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


def crop_img(tensor, target_tensor):
    # square images size
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size

    # division that rounds down to nearest whole number
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


# U-Net architecture
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 5 double convs on the way down, matching the design
        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        # essentially starts making it into the same size as the original
        # this is done with transposing a matrix
        self.up_trans1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_conv1 = double_conv(1024, 512)

        self.up_trans2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_conv2 = double_conv(512, 256)

        self.up_trans3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_conv3 = double_conv(256, 128)

        self.up_trans4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.up_conv4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,  # increase for more segmented obj
            kernel_size=1
        )

    def forward(self, image):
        # batchsize, channnel, h, w
        # encoder
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)

        # decoder
        x = self.up_trans1(x9)
        y = crop_img(x7, x)
        x = self.up_conv1(torch.cat([x, y], 1))

        x = self.up_trans2(x)
        y = crop_img(x5, x)
        x = self.up_conv2(torch.cat([x, y], 1))

        x = self.up_trans3(x)
        y = crop_img(x3, x)
        x = self.up_conv3(torch.cat([x, y], 1))

        x = self.up_trans4(x)
        y = crop_img(x1, x)
        x = self.up_conv4(torch.cat([x, y], 1))

        x = self.out(x)
        print(x.size())
        return x


if __name__ == "__main__":
    # random image of 572*572 based on input size in paper
    image = torch.rand((1, 1, 572, 572))
    model = unet()
    print(model(image))
