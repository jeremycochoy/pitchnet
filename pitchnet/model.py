import torch.nn as nn
import torch

__all__ = ['ResBlock', 'BottleneckBlock', 'DilationResBlock',
           'ChannelMaxPooling', 'FreqMaxPooling', 'Head', 'TCS2CTS',
           'build_monophonic_resnet', 'build_model']


class ResBlock(nn.Module):
    """
    Resnet like block which stack two layers and forward the signal
    """

    def __init__(self, channels, kernel=(3, 3), bias=False, groups=4):
        super().__init__()

        if kernel[0] % 2 == 0:
            raise (Exception("First kernel dimension should be odd!"))
        if kernel[1] % 2 == 0:
            raise (Exception("Second kernel dimension should be odd!"))
        paddx = int((kernel[0] - 1) / 2)
        paddy = int((kernel[1] - 1) / 2)

        self.c = channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.c, self.c, kernel_size=kernel, stride=(1, 1), padding=(paddx, paddy), bias=bias, groups=groups),
            nn.CELU(inplace=True),
            nn.BatchNorm2d(self.c),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.c, self.c, kernel_size=kernel, stride=(1, 1), padding=(paddx, paddy), bias=bias, groups=groups),
            nn.CELU(inplace=True),
            nn.BatchNorm2d(self.c),
        )

    def forward(self, x):
        return self.layer2(self.layer1(x)) + x


class BottleneckBlock(nn.Module):
    """
    A convolutional layer that reduce the dimension of the input
    """
    def __init__(self, channels_in, channels_out, kernel=(3, 3), bias=False, groups=4):
        super().__init__()

        if kernel[0] % 2 == 0:
            raise (Exception("First kernel dimension should be odd!"))
        if kernel[1] % 2 == 0:
            raise (Exception("Second kernel dimension should be odd!"))
        paddx = int((kernel[0] - 1) / 2)
        paddy = int((kernel[1] - 1) / 2)

        self.c_in = channels_in
        self.c_out = channels_out
        self.layer = nn.Sequential(
            nn.Conv2d(self.c_in, self.c_out, kernel_size=kernel, stride=(1, 2), padding=(paddx, paddy), bias=bias,
                      groups=groups),
            nn.CELU(inplace=True),
            nn.BatchNorm2d(self.c_out),
        )

    def forward(self, x):
        return self.layer(x)


class DilationResBlock(nn.Module):
    """
    Resnet like block which stack two layers and forward the signal.
    First layer is a normal 3x3 conv. Second layer is a 3 dilated 3x3 kernel.
    """

    def __init__(self, channels, bias=False, groups=4, dilation_power=1):
        super().__init__()

        assert dilation_power >= 1
        self.c = channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.c, self.c, kernel_size=3, stride=1, padding=3, dilation=3, bias=bias, groups=groups),
            # nn.LeakyReLU(inplace=True),
            nn.CELU(inplace=True),
            nn.BatchNorm2d(self.c),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.c, self.c, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias, groups=groups),
            # nn.LeakyReLU(inplace=True),
            nn.CELU(inplace=True),
            nn.BatchNorm2d(self.c),
        )

    def forward(self, x):
        return self.layer2(self.layer1(x)) + x



class DilationResBlock1D(nn.Module):
    """
    Resnet like block which stack two layers and forward the signal.
    First layer is a normal 3x3 conv. Second layer is a 3 dilated 3x3 kernel.
    """

    def __init__(self, channels, bias=False, groups=4, dilation_power=1):
        super().__init__()

        assert dilation_power >= 1
        self.c = channels
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.c, self.c, kernel_size=3, stride=1, padding=2**dilation_power, dilation=2**dilation_power, bias=bias, groups=groups),
            nn.CELU(inplace=True),
            nn.BatchNorm1d(self.c),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.c, self.c, kernel_size=3, stride=1, padding=2**dilation_power, dilation=2**dilation_power, bias=bias, groups=groups),
            nn.CELU(inplace=True),
            nn.BatchNorm1d(self.c),
        )

    def forward(self, x):
        return self.layer2(self.layer1(x)) + x


class ChannelMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, _ = torch.max(x, 1, keepdim=True)
        return x


class FreqMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, _ = torch.max(x, 3, keepdim=True)
        return x


class GruBlock(nn.Module):
    """
    Transform an input of shape NxTx1 to NxTxE
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.fx = nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):  # NxCxT
        x = x.transpose(1, 2)  # NxTxC
        if self.bidirectional:
            hidden_shape = (self.num_layers * 2, x.shape[0], self.hidden_size)
        else:
            hidden_shape = (self.num_layers, x.shape[0], self.hidden_size)
        h = torch.zeros(hidden_shape).to(x.device)
        x, _ = self.fx(x, h)  # NxTxC
        x = x.transpose(1, 2)  # NxCxT
        return x


class ResnetStackBlock(nn.Module):
    def __init__(self, input_size, hidden_size, width=4):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=1, padding=0, groups=1, bias=True),
            DilationResBlock1D(hidden_size, bias=True, groups=1, dilation_power=2),
            DilationResBlock1D(hidden_size, bias=True, groups=1, dilation_power=5),
            DilationResBlock1D(hidden_size, bias=True, groups=1, dilation_power=6),
            DilationResBlock1D(hidden_size, bias=True, groups=1, dilation_power=7),
            DilationResBlock1D(hidden_size, bias=True, groups=1, dilation_power=8),
            DilationResBlock1D(hidden_size, bias=True, groups=1, dilation_power=9),
            nn.Conv1d(hidden_size, 1, kernel_size=1, padding=0, groups=1, bias=True),
        )

    def forward(self, x):
        return self.layer(x)

class GRUStackBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=1, padding=0, groups=1, bias=True),
            nn.GRU(hidden_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True),
            nn.Conv1d(hidden_size, 1, kernel_size=1, padding=0, groups=1, bias=True),
        )

    def forward(self, x):
        return self.layer(x)


class Head(nn.Module):
    def __init__(self, width=4):
        super().__init__()

        self.enable_width = False
        self.enable_confidence = False
        self.enable_offset = False
        self.enable_presence = False

        self.pitch = nn.Conv1d(256 // 4 * width, 128, kernel_size=3, padding=1, groups=1, bias=True)
        self.relu1 = nn.CELU(inplace=True)
        self.soft1 = nn.LogSoftmax(dim=1)

        if self.enable_presence:
        # self.presence = nn.Conv1d(256 // 4 * width, 1, kernel_size=1, padding=0, groups=1, bias=True)
            self.presence = ResnetStackBlock(256 // 4 * width, 256 // 4 * width)
        else:
            self.presence = nn.Conv1d(256 // 4 * width, 1, kernel_size=1, padding=0, groups=1, bias=True)

        if self.enable_width:
            self.width = GRUStackBlock(256 // 4 * width, 16)
        if self.enable_offset:
            # We have two variant of the offset
            self.offset = nn.Conv1d(256 // 4 * width, 1, kernel_size=1, padding=0, groups=1, bias=True)
            # self.offset = ResnetStackBlock(256 // 4 * width, 256 // 4 * width)
        if self.enable_confidence:
            self.confidence = GRUStackBlock(256 // 4 * width, 256 // 4 * width)
        
    def forward(self, x):
        """
        Expect input of shape NxCxTx1 and
        output a shape

        N: Batch size
        C: Channels (3 frames)
        T: Time scale

        :param x: A resnet processed batch from the model's body.
        :return: An analysed batch of shape NxTx(128+2)
        """
        # Input is of shape NxCxTx1

        # Reshape to NxCxT
        x = x.view(x.shape[0], x.shape[1], x.shape[2])

        # Compute extracted features
        x_pitch = self.soft1(self.relu1(self.pitch(x)))

        if self.enable_presence:
            x_seg_presence = torch.sigmoid(self.presence(x))
        else:
            x_seg_presence = torch.zeros_like(x_pitch[:, -1:, :])

        if self.enable_offset:
            x_seg_offset = torch.tanh(self.offset(x))
        else:
            x_seg_offset = torch.zeros_like(x_pitch[:, -1:, :])

        if self.enable_width:
            x_seg_width = torch.sigmoid(self.width(x)) * 7  # Normalize by sequence length
        else:
            x_seg_width = torch.zeros_like(x_pitch[:, -1:, :])

        if self.enable_confidence:
            x_seg_confidence = torch.sigmoid(self.confidence(x))
        else:
            x_seg_confidence = torch.zeros_like(x_pitch[:, -1:, :])

        # Concatenate everything in a single tensor
        x = torch.cat([x_seg_width, x_seg_offset, x_seg_confidence, x_seg_presence, x_pitch], dim=1)

        return x.transpose(1, 2)


class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.nb_midi_numbers = 128
        self.nb_coordinates = 4

        self.cnn2 = nn.Conv1d(1024, self.nb_midi_numbers * self.nb_coordinates, kernel_size=1, padding=0, groups=1, bias=True)
        self.relu2 = nn.ELU(inplace=True)
        self.soft2 = nn.LogSoftmax(dim=1)

        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 1024, kernel_size=3, padding=1, groups=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024)
        )

        self.cnn1 = self.layer1[1]

    def forward(self, x):
        """
        Expect input of shape NxCxTx1

        The last dimension of the output is of the form
        (pitch, objectness, width, offset)

        N: Batch size
        C: Channels (3 frames)
        T: Time scale

        :param x: A resnet processed batch from the model's body.
        :return: An analysed batch of shape NxTx128x4
        """
        # Input is of shape NxCxTx1

        # Reshape to NxCxT
        N = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]
        x = x.view(N, C, T)

        # Apply CNN for Pitch
        x_features = self.layer1(x)

        # Produce segmentation estimation
        x_raw_box = self.cnn2(x_features)
        x_raw_box = x_raw_box.transpose(1, 2).reshape(N, T, self.nb_midi_numbers, self.nb_coordinates)

        # log(sigmoid) saturate vertically, so logsoftmax :(
        pitch = self.soft2(self.relu2(x_raw_box[:, :, :, 0]).transpose(1, 2)).transpose(1, 2)

        objectness = torch.sigmoid(x_raw_box[:, :, :, 1])
        width = torch.exp(x_raw_box[:, :, :, 2]) * 50 # 0.5 s is anchor box
        offset = torch.sigmoid(x_raw_box[:, :, :, 3]) * width

        x_box = torch.stack([pitch, objectness, width, offset], dim=3)
        #x_box = torch.stack([pitch, pitch, pitch, pitch], dim=3)
        return x_box


class TCS2CTS(nn.Module):
    """
    Reshape an input of format NxTxCxS to NxCxTxS
    """
    def __init__(self, ablation_autocorrelation=False):
        super().__init__()
        self.ablation_autocorrelation = ablation_autocorrelation

    def forward(self, x):
        """
        Reshape the input of neural network to match dataset format.

        :param x: A vector shaped NxTxCxS
        :return: The same vector shaped NxCxTxS
        """

        if self.ablation_autocorrelation:
            # Zero the autocorrelation channel
            x[:, :, 2, :] = x[:, :, 2, :] * 0
        return x.transpose(1, 2)


class NormalizeCTS(nn.Module):
    """
    Normalize an input of shape NxCxTxS
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Normalize neural network input along the time and freqency axis.

        :param x: Frames from preprocessing
        :return: Frames normalized for network
        """
        # Mean along time and freq axis
        x -= x.mean(dim=(2, 3))[:, :, None, None]
        std = x.std(dim=(2, 3))[:, :, None, None]
        std += (std == 0).double()  # Replace 0 values by 1
        x /= std
        return x


def build_monophonic_resnet(width = 4):
    return nn.Sequential(
        BottleneckBlock(4, 128 // 4 * width, kernel=(7, 7), groups=1),

        BottleneckBlock(128 // 4 * width, 128 // 4 * width, kernel=(3, 3), groups=2),
        DilationResBlock(128 // 4 * width),

        BottleneckBlock(128 // 4 * width, 128 // 4 * width, kernel=(3, 3)),
        DilationResBlock(128 // 4 * width),

        BottleneckBlock(128 // 4 * width, 256 // 4 * width, kernel=(3, 3)),
        DilationResBlock(256 // 4 * width),

        BottleneckBlock(256 // 4 * width, 256 // 4 * width, kernel=(3, 3)),
        DilationResBlock(256 // 4 * width),

        BottleneckBlock(256 // 4 * width, 256 // 4 * width, kernel=(1, 3), groups=1),
        BottleneckBlock(256 // 4 * width, 256 // 4 * width, kernel=(1, 3), groups=1),

        nn.Conv2d(256 // 4 * width, 256 // 4 * width, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0)),


        nn.CELU(inplace=True),
    )


def build_model(ablation_autocorrelation=False):
    return nn.Sequential(TCS2CTS(ablation_autocorrelation), NormalizeCTS(), build_monophonic_resnet(width=2), Head(width=2))


def build_segmentation_model(model):
    """
    Build a segmentation model using an existing model.

    :param model: A model create with build_model
    :return: A new model with previous layer and a new head
    """

    return nn.Sequential(model[0], model[1], model[2], SegmentationHead())
