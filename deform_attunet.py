import torch.nn as nn
import torch


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class AttU_Net_deform(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, start_channel=64):
        super(AttU_Net_deform, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=start_channel)
        self.Conv2 = conv_block(ch_in=start_channel, ch_out=start_channel*2)
        self.Conv3 = conv_block(ch_in=start_channel*2, ch_out=start_channel*4)
        self.Conv4 = conv_block(ch_in=start_channel*4, ch_out=start_channel*8)
        self.Conv5 = conv_block(ch_in=start_channel*8, ch_out=start_channel*16)

        self.Up5 = up_conv(ch_in=start_channel*16, ch_out=start_channel*8)
        self.Att5 = Attention_block(F_g=start_channel*8, F_l=start_channel*8, F_int=start_channel*4)
        self.Up_conv5 = conv_block(ch_in=start_channel*16, ch_out=start_channel*8)

        self.Up4 = up_conv(ch_in=start_channel*8, ch_out=start_channel*4)
        self.Att4 = Attention_block(F_g=start_channel*4, F_l=start_channel*4, F_int=start_channel*2)
        self.Up_conv4 = conv_block(ch_in=start_channel*8, ch_out=start_channel*4)

        self.Up3 = up_conv(ch_in=start_channel*4, ch_out=start_channel*2)
        self.Att3 = Attention_block(F_g=start_channel*2, F_l=start_channel*2, F_int=start_channel)
        self.Up_conv3 = conv_block(ch_in=start_channel*4, ch_out=start_channel*2)

        self.Up2 = up_conv(ch_in=start_channel*2, ch_out=start_channel)
        self.Att2 = Attention_block(F_g=start_channel, F_l=start_channel, F_int=int(start_channel/2))
        self.Up_conv2 = conv_block(ch_in=start_channel*2, ch_out=start_channel)

        self.Conv_1x1 = nn.Conv2d(start_channel, output_ch, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

        self.averp = nn.AvgPool2d((15, 15), stride=(1, 1), padding=(int((15 - 1) / 2), int((15 - 1) / 2)))

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d3 = self.Conv_1x1(d2)
        d3 = self.averp(d3)
        d3 = self.tanh(d3)
        return d3