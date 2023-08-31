#!/usr/bin/env python3

# Look for pitchnet in current directory and parent directory
import sys
sys.path = ['.', '..'] + sys.path

from pitchnet.model import *

import torch
import torch.nn as nn
import torch.onnx

__all__ = ['ResBlock', 'BottleneckBlock', 'DilationResBlock',
           'ChannelMaxPooling', 'FreqMaxPooling', 'Head', 'TCS2CTS',
           'build_monophonic_resnet', 'build_model']


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
        #std = std + (std == 0).double()  # Replace 0 values by 1
        #std = torch.sqrt((x*x).sum(dim=(2, 3)) / x.numel())[:, :, None, None]
        std = torch.clamp_min(std, 1e-3)
        x = x / std
        return x


def build_monophonic_resnet():
    return nn.Sequential(
        BottleneckBlock(4, 32, kernel=(7, 7), groups=1),

        # BottleneckBlock(64, 64, kernel=(3, 3), groups=1),
        # DilationResBlock(64),
        # DilationResBlock(64),
        # DilationResBlock(64),
        #
        # BottleneckBlock(64, 128, kernel=(3, 3)),
        # DilationResBlock(128),
        # DilationResBlock(128),
        # DilationResBlock(128),
        #
        # BottleneckBlock(128, 256, kernel=(3, 3)),
        # DilationResBlock(256),
        # DilationResBlock(256),
        # DilationResBlock(256),
        #
        # BottleneckBlock(256, 256, kernel=(3, 3)),
        # DilationResBlock(256),

        #DEBUG
        BottleneckBlock(32, 32, kernel=(3, 3), groups=1),
        DilationResBlock(32),
#        DilationResBlock(64),
#        DilationResBlock(64),

        BottleneckBlock(32, 64, kernel=(3, 3)),
        DilationResBlock(64),
#        DilationResBlock(128),
#        DilationResBlock(128),

        BottleneckBlock(64, 128, kernel=(3, 3)),
        DilationResBlock(128),
#        DilationResBlock(256),
#        DilationResBlock(256),

        BottleneckBlock(128, 256, kernel=(3, 3)),
        # DilationResBlock(256),

        # Kernel is 256x1x65 for dims CxTxS
        nn.Conv2d(256, 256, kernel_size=(1, 65), stride=(1, 1), padding=(0, 0)),
        nn.ReLU(inplace=True),
    )


def build_model():
    #return nn.Sequential(TCS2CTS(), build_monophonic_resnet(), Head())
    return nn.Sequential(TCS2CTS(), NormalizeCTS(), build_monophonic_resnet(), Head())


def onnx_export(model: torch.nn.Module, input_shape, filename: str,
                input_names=["network_input"],
                output_names=["network_output"]) -> ():
    # Create dummy input
    device = model.parameters().__next__().device
    dummy_input = torch.rand(input_shape).to(device)

    # Convert the PyTorch model to ONNX
    torch.onnx.export(model,
                      dummy_input,
                      filename,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)


def main():
    # NxTxCxS
    shape = (1, 100, 4, 2049)
    model = build_model()

    # Write model
    onnx_export(model, shape, "SimpleModel.onnx")

    # Convert to coreml
    import os
    os.system("python3 models/onnx-to-coreml.py SimpleModel.onnx -o SimpleModel.mlmodel "
              "--minimum_ios_deployment_target=13")


if __name__ == "__main__":
    # execute only if run as a script
    main()
