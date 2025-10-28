import torch
import torch.nn as nn
from collections import OrderedDict

class Scale(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, x):
        return x * self.value

class ConvertedModel(nn.Module):
    def __init__(self):
        super().__init__()


        layers = []

        layers.append(('_mulscalar0', Scale(0.0078125)))

        layers.append(('convolution0', nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm0', nn.BatchNorm2d(32)))

        layers.append(('activation0', nn.ReLU(inplace=True)))

        layers.append(('convolution1', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm1', nn.BatchNorm2d(32)))

        layers.append(('activation1', nn.ReLU(inplace=True)))

        layers.append(('convolution2', nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm2', nn.BatchNorm2d(64)))

        layers.append(('activation2', nn.ReLU(inplace=True)))

        layers.append(('convolution3', nn.Conv2d(
            in_channels=64,
            out_channels=80,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm3', nn.BatchNorm2d(80)))

        layers.append(('activation3', nn.ReLU(inplace=True)))

        layers.append(('convolution4', nn.Conv2d(
            in_channels=80,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm4', nn.BatchNorm2d(192)))

        layers.append(('activation4', nn.ReLU(inplace=True)))

        layers.append(('pooling1', nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0)
        )))

        layers.append(('convolution5', nn.Conv2d(
            in_channels=192,
            out_channels=96,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm5', nn.BatchNorm2d(96)))

        layers.append(('activation5', nn.ReLU(inplace=True)))

        layers.append(('convolution6', nn.Conv2d(
            in_channels=192,
            out_channels=48,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm6', nn.BatchNorm2d(48)))

        layers.append(('activation6', nn.ReLU(inplace=True)))

        layers.append(('convolution7', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=True
        )))

        layers.append(('batchnorm7', nn.BatchNorm2d(64)))

        layers.append(('activation7', nn.ReLU(inplace=True)))

        layers.append(('convolution8', nn.Conv2d(
            in_channels=192,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm8', nn.BatchNorm2d(64)))

        layers.append(('activation8', nn.ReLU(inplace=True)))

        layers.append(('convolution9', nn.Conv2d(
            in_channels=64,
            out_channels=96,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm9', nn.BatchNorm2d(96)))

        layers.append(('activation9', nn.ReLU(inplace=True)))

        layers.append(('convolution10', nn.Conv2d(
            in_channels=96,
            out_channels=96,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm10', nn.BatchNorm2d(96)))

        layers.append(('activation10', nn.ReLU(inplace=True)))

        layers.append(('pooling2', nn.AvgPool2d(
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )))

        layers.append(('convolution11', nn.Conv2d(
            in_channels=192,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm11', nn.BatchNorm2d(64)))

        layers.append(('activation11', nn.ReLU(inplace=True)))

        layers.append(('convolution12', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm12', nn.BatchNorm2d(32)))

        layers.append(('activation12', nn.ReLU(inplace=True)))

        layers.append(('convolution13', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm13', nn.BatchNorm2d(32)))

        layers.append(('activation13', nn.ReLU(inplace=True)))

        layers.append(('convolution14', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm14', nn.BatchNorm2d(32)))

        layers.append(('activation14', nn.ReLU(inplace=True)))

        layers.append(('convolution15', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm15', nn.BatchNorm2d(32)))

        layers.append(('activation15', nn.ReLU(inplace=True)))

        layers.append(('convolution16', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm16', nn.BatchNorm2d(48)))

        layers.append(('activation16', nn.ReLU(inplace=True)))

        layers.append(('convolution17', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm17', nn.BatchNorm2d(64)))

        layers.append(('activation17', nn.ReLU(inplace=True)))

        layers.append(('convolution18', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm18', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar1', Scale(0.17)))

        layers.append(('activation18', nn.ReLU(inplace=True)))

        layers.append(('convolution19', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm19', nn.BatchNorm2d(32)))

        layers.append(('activation19', nn.ReLU(inplace=True)))

        layers.append(('convolution20', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm20', nn.BatchNorm2d(32)))

        layers.append(('activation20', nn.ReLU(inplace=True)))

        layers.append(('convolution21', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm21', nn.BatchNorm2d(32)))

        layers.append(('activation21', nn.ReLU(inplace=True)))

        layers.append(('convolution22', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm22', nn.BatchNorm2d(32)))

        layers.append(('activation22', nn.ReLU(inplace=True)))

        layers.append(('convolution23', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm23', nn.BatchNorm2d(48)))

        layers.append(('activation23', nn.ReLU(inplace=True)))

        layers.append(('convolution24', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm24', nn.BatchNorm2d(64)))

        layers.append(('activation24', nn.ReLU(inplace=True)))

        layers.append(('convolution25', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm25', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar2', Scale(0.17)))

        layers.append(('activation25', nn.ReLU(inplace=True)))

        layers.append(('convolution26', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm26', nn.BatchNorm2d(32)))

        layers.append(('activation26', nn.ReLU(inplace=True)))

        layers.append(('convolution27', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm27', nn.BatchNorm2d(32)))

        layers.append(('activation27', nn.ReLU(inplace=True)))

        layers.append(('convolution28', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm28', nn.BatchNorm2d(32)))

        layers.append(('activation28', nn.ReLU(inplace=True)))

        layers.append(('convolution29', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm29', nn.BatchNorm2d(32)))

        layers.append(('activation29', nn.ReLU(inplace=True)))

        layers.append(('convolution30', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm30', nn.BatchNorm2d(48)))

        layers.append(('activation30', nn.ReLU(inplace=True)))

        layers.append(('convolution31', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm31', nn.BatchNorm2d(64)))

        layers.append(('activation31', nn.ReLU(inplace=True)))

        layers.append(('convolution32', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm32', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar3', Scale(0.17)))

        layers.append(('activation32', nn.ReLU(inplace=True)))

        layers.append(('convolution33', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm33', nn.BatchNorm2d(32)))

        layers.append(('activation33', nn.ReLU(inplace=True)))

        layers.append(('convolution34', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm34', nn.BatchNorm2d(32)))

        layers.append(('activation34', nn.ReLU(inplace=True)))

        layers.append(('convolution35', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm35', nn.BatchNorm2d(32)))

        layers.append(('activation35', nn.ReLU(inplace=True)))

        layers.append(('convolution36', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm36', nn.BatchNorm2d(32)))

        layers.append(('activation36', nn.ReLU(inplace=True)))

        layers.append(('convolution37', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm37', nn.BatchNorm2d(48)))

        layers.append(('activation37', nn.ReLU(inplace=True)))

        layers.append(('convolution38', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm38', nn.BatchNorm2d(64)))

        layers.append(('activation38', nn.ReLU(inplace=True)))

        layers.append(('convolution39', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm39', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar4', Scale(0.17)))

        layers.append(('activation39', nn.ReLU(inplace=True)))

        layers.append(('convolution40', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm40', nn.BatchNorm2d(32)))

        layers.append(('activation40', nn.ReLU(inplace=True)))

        layers.append(('convolution41', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm41', nn.BatchNorm2d(32)))

        layers.append(('activation41', nn.ReLU(inplace=True)))

        layers.append(('convolution42', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm42', nn.BatchNorm2d(32)))

        layers.append(('activation42', nn.ReLU(inplace=True)))

        layers.append(('convolution43', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm43', nn.BatchNorm2d(32)))

        layers.append(('activation43', nn.ReLU(inplace=True)))

        layers.append(('convolution44', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm44', nn.BatchNorm2d(48)))

        layers.append(('activation44', nn.ReLU(inplace=True)))

        layers.append(('convolution45', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm45', nn.BatchNorm2d(64)))

        layers.append(('activation45', nn.ReLU(inplace=True)))

        layers.append(('convolution46', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm46', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar5', Scale(0.17)))

        layers.append(('activation46', nn.ReLU(inplace=True)))

        layers.append(('convolution47', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm47', nn.BatchNorm2d(32)))

        layers.append(('activation47', nn.ReLU(inplace=True)))

        layers.append(('convolution48', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm48', nn.BatchNorm2d(32)))

        layers.append(('activation48', nn.ReLU(inplace=True)))

        layers.append(('convolution49', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm49', nn.BatchNorm2d(32)))

        layers.append(('activation49', nn.ReLU(inplace=True)))

        layers.append(('convolution50', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm50', nn.BatchNorm2d(32)))

        layers.append(('activation50', nn.ReLU(inplace=True)))

        layers.append(('convolution51', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm51', nn.BatchNorm2d(48)))

        layers.append(('activation51', nn.ReLU(inplace=True)))

        layers.append(('convolution52', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm52', nn.BatchNorm2d(64)))

        layers.append(('activation52', nn.ReLU(inplace=True)))

        layers.append(('convolution53', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm53', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar6', Scale(0.17)))

        layers.append(('activation53', nn.ReLU(inplace=True)))

        layers.append(('convolution54', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm54', nn.BatchNorm2d(32)))

        layers.append(('activation54', nn.ReLU(inplace=True)))

        layers.append(('convolution55', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm55', nn.BatchNorm2d(32)))

        layers.append(('activation55', nn.ReLU(inplace=True)))

        layers.append(('convolution56', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm56', nn.BatchNorm2d(32)))

        layers.append(('activation56', nn.ReLU(inplace=True)))

        layers.append(('convolution57', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm57', nn.BatchNorm2d(32)))

        layers.append(('activation57', nn.ReLU(inplace=True)))

        layers.append(('convolution58', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm58', nn.BatchNorm2d(48)))

        layers.append(('activation58', nn.ReLU(inplace=True)))

        layers.append(('convolution59', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm59', nn.BatchNorm2d(64)))

        layers.append(('activation59', nn.ReLU(inplace=True)))

        layers.append(('convolution60', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm60', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar7', Scale(0.17)))

        layers.append(('activation60', nn.ReLU(inplace=True)))

        layers.append(('convolution61', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm61', nn.BatchNorm2d(32)))

        layers.append(('activation61', nn.ReLU(inplace=True)))

        layers.append(('convolution62', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm62', nn.BatchNorm2d(32)))

        layers.append(('activation62', nn.ReLU(inplace=True)))

        layers.append(('convolution63', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm63', nn.BatchNorm2d(32)))

        layers.append(('activation63', nn.ReLU(inplace=True)))

        layers.append(('convolution64', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm64', nn.BatchNorm2d(32)))

        layers.append(('activation64', nn.ReLU(inplace=True)))

        layers.append(('convolution65', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm65', nn.BatchNorm2d(48)))

        layers.append(('activation65', nn.ReLU(inplace=True)))

        layers.append(('convolution66', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm66', nn.BatchNorm2d(64)))

        layers.append(('activation66', nn.ReLU(inplace=True)))

        layers.append(('convolution67', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm67', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar8', Scale(0.17)))

        layers.append(('activation67', nn.ReLU(inplace=True)))

        layers.append(('convolution68', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm68', nn.BatchNorm2d(32)))

        layers.append(('activation68', nn.ReLU(inplace=True)))

        layers.append(('convolution69', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm69', nn.BatchNorm2d(32)))

        layers.append(('activation69', nn.ReLU(inplace=True)))

        layers.append(('convolution70', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm70', nn.BatchNorm2d(32)))

        layers.append(('activation70', nn.ReLU(inplace=True)))

        layers.append(('convolution71', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm71', nn.BatchNorm2d(32)))

        layers.append(('activation71', nn.ReLU(inplace=True)))

        layers.append(('convolution72', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm72', nn.BatchNorm2d(48)))

        layers.append(('activation72', nn.ReLU(inplace=True)))

        layers.append(('convolution73', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm73', nn.BatchNorm2d(64)))

        layers.append(('activation73', nn.ReLU(inplace=True)))

        layers.append(('convolution74', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm74', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar9', Scale(0.17)))

        layers.append(('activation74', nn.ReLU(inplace=True)))

        layers.append(('convolution75', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm75', nn.BatchNorm2d(32)))

        layers.append(('activation75', nn.ReLU(inplace=True)))

        layers.append(('convolution76', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm76', nn.BatchNorm2d(32)))

        layers.append(('activation76', nn.ReLU(inplace=True)))

        layers.append(('convolution77', nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm77', nn.BatchNorm2d(32)))

        layers.append(('activation77', nn.ReLU(inplace=True)))

        layers.append(('convolution78', nn.Conv2d(
            in_channels=320,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm78', nn.BatchNorm2d(32)))

        layers.append(('activation78', nn.ReLU(inplace=True)))

        layers.append(('convolution79', nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm79', nn.BatchNorm2d(48)))

        layers.append(('activation79', nn.ReLU(inplace=True)))

        layers.append(('convolution80', nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm80', nn.BatchNorm2d(64)))

        layers.append(('activation80', nn.ReLU(inplace=True)))

        layers.append(('convolution81', nn.Conv2d(
            in_channels=128,
            out_channels=320,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm81', nn.BatchNorm2d(320)))

        layers.append(('_mulscalar10', Scale(0.17)))

        layers.append(('activation81', nn.ReLU(inplace=True)))

        layers.append(('convolution82', nn.Conv2d(
            in_channels=320,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm82', nn.BatchNorm2d(384)))

        layers.append(('activation82', nn.ReLU(inplace=True)))

        layers.append(('convolution83', nn.Conv2d(
            in_channels=320,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm83', nn.BatchNorm2d(256)))

        layers.append(('activation83', nn.ReLU(inplace=True)))

        layers.append(('convolution84', nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm84', nn.BatchNorm2d(256)))

        layers.append(('activation84', nn.ReLU(inplace=True)))

        layers.append(('convolution85', nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm85', nn.BatchNorm2d(384)))

        layers.append(('activation85', nn.ReLU(inplace=True)))

        layers.append(('pooling3', nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0)
        )))

        layers.append(('convolution86', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm86', nn.BatchNorm2d(192)))

        layers.append(('activation86', nn.ReLU(inplace=True)))

        layers.append(('convolution87', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm87', nn.BatchNorm2d(129)))

        layers.append(('activation87', nn.ReLU(inplace=True)))

        layers.append(('convolution88', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm88', nn.BatchNorm2d(160)))

        layers.append(('activation88', nn.ReLU(inplace=True)))

        layers.append(('convolution89', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm89', nn.BatchNorm2d(192)))

        layers.append(('activation89', nn.ReLU(inplace=True)))

        layers.append(('convolution90', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm90', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar11', Scale(0.1)))

        layers.append(('activation90', nn.ReLU(inplace=True)))

        layers.append(('convolution91', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm91', nn.BatchNorm2d(192)))

        layers.append(('activation91', nn.ReLU(inplace=True)))

        layers.append(('convolution92', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm92', nn.BatchNorm2d(129)))

        layers.append(('activation92', nn.ReLU(inplace=True)))

        layers.append(('convolution93', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm93', nn.BatchNorm2d(160)))

        layers.append(('activation93', nn.ReLU(inplace=True)))

        layers.append(('convolution94', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm94', nn.BatchNorm2d(192)))

        layers.append(('activation94', nn.ReLU(inplace=True)))

        layers.append(('convolution95', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm95', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar12', Scale(0.1)))

        layers.append(('activation95', nn.ReLU(inplace=True)))

        layers.append(('convolution96', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm96', nn.BatchNorm2d(192)))

        layers.append(('activation96', nn.ReLU(inplace=True)))

        layers.append(('convolution97', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm97', nn.BatchNorm2d(129)))

        layers.append(('activation97', nn.ReLU(inplace=True)))

        layers.append(('convolution98', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm98', nn.BatchNorm2d(160)))

        layers.append(('activation98', nn.ReLU(inplace=True)))

        layers.append(('convolution99', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm99', nn.BatchNorm2d(192)))

        layers.append(('activation99', nn.ReLU(inplace=True)))

        layers.append(('convolution100', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm100', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar13', Scale(0.1)))

        layers.append(('activation100', nn.ReLU(inplace=True)))

        layers.append(('convolution101', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm101', nn.BatchNorm2d(192)))

        layers.append(('activation101', nn.ReLU(inplace=True)))

        layers.append(('convolution102', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm102', nn.BatchNorm2d(129)))

        layers.append(('activation102', nn.ReLU(inplace=True)))

        layers.append(('convolution103', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm103', nn.BatchNorm2d(160)))

        layers.append(('activation103', nn.ReLU(inplace=True)))

        layers.append(('convolution104', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm104', nn.BatchNorm2d(192)))

        layers.append(('activation104', nn.ReLU(inplace=True)))

        layers.append(('convolution105', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm105', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar14', Scale(0.1)))

        layers.append(('activation105', nn.ReLU(inplace=True)))

        layers.append(('convolution106', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm106', nn.BatchNorm2d(192)))

        layers.append(('activation106', nn.ReLU(inplace=True)))

        layers.append(('convolution107', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm107', nn.BatchNorm2d(129)))

        layers.append(('activation107', nn.ReLU(inplace=True)))

        layers.append(('convolution108', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm108', nn.BatchNorm2d(160)))

        layers.append(('activation108', nn.ReLU(inplace=True)))

        layers.append(('convolution109', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm109', nn.BatchNorm2d(192)))

        layers.append(('activation109', nn.ReLU(inplace=True)))

        layers.append(('convolution110', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm110', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar15', Scale(0.1)))

        layers.append(('activation110', nn.ReLU(inplace=True)))

        layers.append(('convolution111', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm111', nn.BatchNorm2d(192)))

        layers.append(('activation111', nn.ReLU(inplace=True)))

        layers.append(('convolution112', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm112', nn.BatchNorm2d(129)))

        layers.append(('activation112', nn.ReLU(inplace=True)))

        layers.append(('convolution113', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm113', nn.BatchNorm2d(160)))

        layers.append(('activation113', nn.ReLU(inplace=True)))

        layers.append(('convolution114', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm114', nn.BatchNorm2d(192)))

        layers.append(('activation114', nn.ReLU(inplace=True)))

        layers.append(('convolution115', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm115', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar16', Scale(0.1)))

        layers.append(('activation115', nn.ReLU(inplace=True)))

        layers.append(('convolution116', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm116', nn.BatchNorm2d(192)))

        layers.append(('activation116', nn.ReLU(inplace=True)))

        layers.append(('convolution117', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm117', nn.BatchNorm2d(129)))

        layers.append(('activation117', nn.ReLU(inplace=True)))

        layers.append(('convolution118', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm118', nn.BatchNorm2d(160)))

        layers.append(('activation118', nn.ReLU(inplace=True)))

        layers.append(('convolution119', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm119', nn.BatchNorm2d(192)))

        layers.append(('activation119', nn.ReLU(inplace=True)))

        layers.append(('convolution120', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm120', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar17', Scale(0.1)))

        layers.append(('activation120', nn.ReLU(inplace=True)))

        layers.append(('convolution121', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm121', nn.BatchNorm2d(192)))

        layers.append(('activation121', nn.ReLU(inplace=True)))

        layers.append(('convolution122', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm122', nn.BatchNorm2d(129)))

        layers.append(('activation122', nn.ReLU(inplace=True)))

        layers.append(('convolution123', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm123', nn.BatchNorm2d(160)))

        layers.append(('activation123', nn.ReLU(inplace=True)))

        layers.append(('convolution124', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm124', nn.BatchNorm2d(192)))

        layers.append(('activation124', nn.ReLU(inplace=True)))

        layers.append(('convolution125', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm125', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar18', Scale(0.1)))

        layers.append(('activation125', nn.ReLU(inplace=True)))

        layers.append(('convolution126', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm126', nn.BatchNorm2d(192)))

        layers.append(('activation126', nn.ReLU(inplace=True)))

        layers.append(('convolution127', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm127', nn.BatchNorm2d(129)))

        layers.append(('activation127', nn.ReLU(inplace=True)))

        layers.append(('convolution128', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm128', nn.BatchNorm2d(160)))

        layers.append(('activation128', nn.ReLU(inplace=True)))

        layers.append(('convolution129', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm129', nn.BatchNorm2d(192)))

        layers.append(('activation129', nn.ReLU(inplace=True)))

        layers.append(('convolution130', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm130', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar19', Scale(0.1)))

        layers.append(('activation130', nn.ReLU(inplace=True)))

        layers.append(('convolution131', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm131', nn.BatchNorm2d(192)))

        layers.append(('activation131', nn.ReLU(inplace=True)))

        layers.append(('convolution132', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm132', nn.BatchNorm2d(129)))

        layers.append(('activation132', nn.ReLU(inplace=True)))

        layers.append(('convolution133', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm133', nn.BatchNorm2d(160)))

        layers.append(('activation133', nn.ReLU(inplace=True)))

        layers.append(('convolution134', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm134', nn.BatchNorm2d(192)))

        layers.append(('activation134', nn.ReLU(inplace=True)))

        layers.append(('convolution135', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm135', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar20', Scale(0.1)))

        layers.append(('activation135', nn.ReLU(inplace=True)))

        layers.append(('convolution136', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm136', nn.BatchNorm2d(192)))

        layers.append(('activation136', nn.ReLU(inplace=True)))

        layers.append(('convolution137', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm137', nn.BatchNorm2d(129)))

        layers.append(('activation137', nn.ReLU(inplace=True)))

        layers.append(('convolution138', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm138', nn.BatchNorm2d(160)))

        layers.append(('activation138', nn.ReLU(inplace=True)))

        layers.append(('convolution139', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm139', nn.BatchNorm2d(192)))

        layers.append(('activation139', nn.ReLU(inplace=True)))

        layers.append(('convolution140', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm140', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar21', Scale(0.1)))

        layers.append(('activation140', nn.ReLU(inplace=True)))

        layers.append(('convolution141', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm141', nn.BatchNorm2d(192)))

        layers.append(('activation141', nn.ReLU(inplace=True)))

        layers.append(('convolution142', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm142', nn.BatchNorm2d(129)))

        layers.append(('activation142', nn.ReLU(inplace=True)))

        layers.append(('convolution143', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm143', nn.BatchNorm2d(160)))

        layers.append(('activation143', nn.ReLU(inplace=True)))

        layers.append(('convolution144', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm144', nn.BatchNorm2d(192)))

        layers.append(('activation144', nn.ReLU(inplace=True)))

        layers.append(('convolution145', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm145', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar22', Scale(0.1)))

        layers.append(('activation145', nn.ReLU(inplace=True)))

        layers.append(('convolution146', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm146', nn.BatchNorm2d(192)))

        layers.append(('activation146', nn.ReLU(inplace=True)))

        layers.append(('convolution147', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm147', nn.BatchNorm2d(129)))

        layers.append(('activation147', nn.ReLU(inplace=True)))

        layers.append(('convolution148', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm148', nn.BatchNorm2d(160)))

        layers.append(('activation148', nn.ReLU(inplace=True)))

        layers.append(('convolution149', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm149', nn.BatchNorm2d(192)))

        layers.append(('activation149', nn.ReLU(inplace=True)))

        layers.append(('convolution150', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm150', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar23', Scale(0.1)))

        layers.append(('activation150', nn.ReLU(inplace=True)))

        layers.append(('convolution151', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm151', nn.BatchNorm2d(192)))

        layers.append(('activation151', nn.ReLU(inplace=True)))

        layers.append(('convolution152', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm152', nn.BatchNorm2d(129)))

        layers.append(('activation152', nn.ReLU(inplace=True)))

        layers.append(('convolution153', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm153', nn.BatchNorm2d(160)))

        layers.append(('activation153', nn.ReLU(inplace=True)))

        layers.append(('convolution154', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm154', nn.BatchNorm2d(192)))

        layers.append(('activation154', nn.ReLU(inplace=True)))

        layers.append(('convolution155', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm155', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar24', Scale(0.1)))

        layers.append(('activation155', nn.ReLU(inplace=True)))

        layers.append(('convolution156', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm156', nn.BatchNorm2d(192)))

        layers.append(('activation156', nn.ReLU(inplace=True)))

        layers.append(('convolution157', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm157', nn.BatchNorm2d(129)))

        layers.append(('activation157', nn.ReLU(inplace=True)))

        layers.append(('convolution158', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm158', nn.BatchNorm2d(160)))

        layers.append(('activation158', nn.ReLU(inplace=True)))

        layers.append(('convolution159', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm159', nn.BatchNorm2d(192)))

        layers.append(('activation159', nn.ReLU(inplace=True)))

        layers.append(('convolution160', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm160', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar25', Scale(0.1)))

        layers.append(('activation160', nn.ReLU(inplace=True)))

        layers.append(('convolution161', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm161', nn.BatchNorm2d(192)))

        layers.append(('activation161', nn.ReLU(inplace=True)))

        layers.append(('convolution162', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm162', nn.BatchNorm2d(129)))

        layers.append(('activation162', nn.ReLU(inplace=True)))

        layers.append(('convolution163', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm163', nn.BatchNorm2d(160)))

        layers.append(('activation163', nn.ReLU(inplace=True)))

        layers.append(('convolution164', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm164', nn.BatchNorm2d(192)))

        layers.append(('activation164', nn.ReLU(inplace=True)))

        layers.append(('convolution165', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm165', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar26', Scale(0.1)))

        layers.append(('activation165', nn.ReLU(inplace=True)))

        layers.append(('convolution166', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm166', nn.BatchNorm2d(192)))

        layers.append(('activation166', nn.ReLU(inplace=True)))

        layers.append(('convolution167', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm167', nn.BatchNorm2d(129)))

        layers.append(('activation167', nn.ReLU(inplace=True)))

        layers.append(('convolution168', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm168', nn.BatchNorm2d(160)))

        layers.append(('activation168', nn.ReLU(inplace=True)))

        layers.append(('convolution169', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm169', nn.BatchNorm2d(192)))

        layers.append(('activation169', nn.ReLU(inplace=True)))

        layers.append(('convolution170', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm170', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar27', Scale(0.1)))

        layers.append(('activation170', nn.ReLU(inplace=True)))

        layers.append(('convolution171', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm171', nn.BatchNorm2d(192)))

        layers.append(('activation171', nn.ReLU(inplace=True)))

        layers.append(('convolution172', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm172', nn.BatchNorm2d(129)))

        layers.append(('activation172', nn.ReLU(inplace=True)))

        layers.append(('convolution173', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm173', nn.BatchNorm2d(160)))

        layers.append(('activation173', nn.ReLU(inplace=True)))

        layers.append(('convolution174', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm174', nn.BatchNorm2d(192)))

        layers.append(('activation174', nn.ReLU(inplace=True)))

        layers.append(('convolution175', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm175', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar28', Scale(0.1)))

        layers.append(('activation175', nn.ReLU(inplace=True)))

        layers.append(('convolution176', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm176', nn.BatchNorm2d(192)))

        layers.append(('activation176', nn.ReLU(inplace=True)))

        layers.append(('convolution177', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm177', nn.BatchNorm2d(129)))

        layers.append(('activation177', nn.ReLU(inplace=True)))

        layers.append(('convolution178', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm178', nn.BatchNorm2d(160)))

        layers.append(('activation178', nn.ReLU(inplace=True)))

        layers.append(('convolution179', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm179', nn.BatchNorm2d(192)))

        layers.append(('activation179', nn.ReLU(inplace=True)))

        layers.append(('convolution180', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm180', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar29', Scale(0.1)))

        layers.append(('activation180', nn.ReLU(inplace=True)))

        layers.append(('convolution181', nn.Conv2d(
            in_channels=1088,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm181', nn.BatchNorm2d(192)))

        layers.append(('activation181', nn.ReLU(inplace=True)))

        layers.append(('convolution182', nn.Conv2d(
            in_channels=1088,
            out_channels=129,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm182', nn.BatchNorm2d(129)))

        layers.append(('activation182', nn.ReLU(inplace=True)))

        layers.append(('convolution183', nn.Conv2d(
            in_channels=129,
            out_channels=160,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(1, 2),
            bias=True
        )))

        layers.append(('batchnorm183', nn.BatchNorm2d(160)))

        layers.append(('activation183', nn.ReLU(inplace=True)))

        layers.append(('convolution184', nn.Conv2d(
            in_channels=160,
            out_channels=192,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(2, 1),
            bias=True
        )))

        layers.append(('batchnorm184', nn.BatchNorm2d(192)))

        layers.append(('activation184', nn.ReLU(inplace=True)))

        layers.append(('convolution185', nn.Conv2d(
            in_channels=384,
            out_channels=1088,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm185', nn.BatchNorm2d(1088)))

        layers.append(('_mulscalar30', Scale(0.1)))

        layers.append(('activation185', nn.ReLU(inplace=True)))

        layers.append(('convolution186', nn.Conv2d(
            in_channels=1088,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm186', nn.BatchNorm2d(256)))

        layers.append(('activation186', nn.ReLU(inplace=True)))

        layers.append(('convolution187', nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm187', nn.BatchNorm2d(384)))

        layers.append(('activation187', nn.ReLU(inplace=True)))

        layers.append(('convolution188', nn.Conv2d(
            in_channels=1088,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm188', nn.BatchNorm2d(256)))

        layers.append(('activation188', nn.ReLU(inplace=True)))

        layers.append(('convolution189', nn.Conv2d(
            in_channels=256,
            out_channels=288,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm189', nn.BatchNorm2d(288)))

        layers.append(('activation189', nn.ReLU(inplace=True)))

        layers.append(('convolution190', nn.Conv2d(
            in_channels=1088,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm190', nn.BatchNorm2d(256)))

        layers.append(('activation190', nn.ReLU(inplace=True)))

        layers.append(('convolution191', nn.Conv2d(
            in_channels=256,
            out_channels=288,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True
        )))

        layers.append(('batchnorm191', nn.BatchNorm2d(288)))

        layers.append(('activation191', nn.ReLU(inplace=True)))

        layers.append(('convolution192', nn.Conv2d(
            in_channels=288,
            out_channels=320,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm192', nn.BatchNorm2d(320)))

        layers.append(('activation192', nn.ReLU(inplace=True)))

        layers.append(('pooling4', nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0)
        )))

        layers.append(('convolution193', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm193', nn.BatchNorm2d(192)))

        layers.append(('activation193', nn.ReLU(inplace=True)))

        layers.append(('convolution194', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm194', nn.BatchNorm2d(192)))

        layers.append(('activation194', nn.ReLU(inplace=True)))

        layers.append(('convolution195', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm195', nn.BatchNorm2d(224)))

        layers.append(('activation195', nn.ReLU(inplace=True)))

        layers.append(('convolution196', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm196', nn.BatchNorm2d(256)))

        layers.append(('activation196', nn.ReLU(inplace=True)))

        layers.append(('convolution197', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm197', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar31', Scale(0.2)))

        layers.append(('activation197', nn.ReLU(inplace=True)))

        layers.append(('convolution198', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm198', nn.BatchNorm2d(192)))

        layers.append(('activation198', nn.ReLU(inplace=True)))

        layers.append(('convolution199', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm199', nn.BatchNorm2d(192)))

        layers.append(('activation199', nn.ReLU(inplace=True)))

        layers.append(('convolution200', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm200', nn.BatchNorm2d(224)))

        layers.append(('activation200', nn.ReLU(inplace=True)))

        layers.append(('convolution201', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm201', nn.BatchNorm2d(256)))

        layers.append(('activation201', nn.ReLU(inplace=True)))

        layers.append(('convolution202', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm202', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar32', Scale(0.2)))

        layers.append(('activation202', nn.ReLU(inplace=True)))

        layers.append(('convolution203', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm203', nn.BatchNorm2d(192)))

        layers.append(('activation203', nn.ReLU(inplace=True)))

        layers.append(('convolution204', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm204', nn.BatchNorm2d(192)))

        layers.append(('activation204', nn.ReLU(inplace=True)))

        layers.append(('convolution205', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm205', nn.BatchNorm2d(224)))

        layers.append(('activation205', nn.ReLU(inplace=True)))

        layers.append(('convolution206', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm206', nn.BatchNorm2d(256)))

        layers.append(('activation206', nn.ReLU(inplace=True)))

        layers.append(('convolution207', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm207', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar33', Scale(0.2)))

        layers.append(('activation207', nn.ReLU(inplace=True)))

        layers.append(('convolution208', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm208', nn.BatchNorm2d(192)))

        layers.append(('activation208', nn.ReLU(inplace=True)))

        layers.append(('convolution209', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm209', nn.BatchNorm2d(192)))

        layers.append(('activation209', nn.ReLU(inplace=True)))

        layers.append(('convolution210', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm210', nn.BatchNorm2d(224)))

        layers.append(('activation210', nn.ReLU(inplace=True)))

        layers.append(('convolution211', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm211', nn.BatchNorm2d(256)))

        layers.append(('activation211', nn.ReLU(inplace=True)))

        layers.append(('convolution212', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm212', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar34', Scale(0.2)))

        layers.append(('activation212', nn.ReLU(inplace=True)))

        layers.append(('convolution213', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm213', nn.BatchNorm2d(192)))

        layers.append(('activation213', nn.ReLU(inplace=True)))

        layers.append(('convolution214', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm214', nn.BatchNorm2d(192)))

        layers.append(('activation214', nn.ReLU(inplace=True)))

        layers.append(('convolution215', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm215', nn.BatchNorm2d(224)))

        layers.append(('activation215', nn.ReLU(inplace=True)))

        layers.append(('convolution216', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm216', nn.BatchNorm2d(256)))

        layers.append(('activation216', nn.ReLU(inplace=True)))

        layers.append(('convolution217', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm217', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar35', Scale(0.2)))

        layers.append(('activation217', nn.ReLU(inplace=True)))

        layers.append(('convolution218', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm218', nn.BatchNorm2d(192)))

        layers.append(('activation218', nn.ReLU(inplace=True)))

        layers.append(('convolution219', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm219', nn.BatchNorm2d(192)))

        layers.append(('activation219', nn.ReLU(inplace=True)))

        layers.append(('convolution220', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm220', nn.BatchNorm2d(224)))

        layers.append(('activation220', nn.ReLU(inplace=True)))

        layers.append(('convolution221', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm221', nn.BatchNorm2d(256)))

        layers.append(('activation221', nn.ReLU(inplace=True)))

        layers.append(('convolution222', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm222', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar36', Scale(0.2)))

        layers.append(('activation222', nn.ReLU(inplace=True)))

        layers.append(('convolution223', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm223', nn.BatchNorm2d(192)))

        layers.append(('activation223', nn.ReLU(inplace=True)))

        layers.append(('convolution224', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm224', nn.BatchNorm2d(192)))

        layers.append(('activation224', nn.ReLU(inplace=True)))

        layers.append(('convolution225', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm225', nn.BatchNorm2d(224)))

        layers.append(('activation225', nn.ReLU(inplace=True)))

        layers.append(('convolution226', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm226', nn.BatchNorm2d(256)))

        layers.append(('activation226', nn.ReLU(inplace=True)))

        layers.append(('convolution227', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm227', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar37', Scale(0.2)))

        layers.append(('activation227', nn.ReLU(inplace=True)))

        layers.append(('convolution228', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm228', nn.BatchNorm2d(192)))

        layers.append(('activation228', nn.ReLU(inplace=True)))

        layers.append(('convolution229', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm229', nn.BatchNorm2d(192)))

        layers.append(('activation229', nn.ReLU(inplace=True)))

        layers.append(('convolution230', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm230', nn.BatchNorm2d(224)))

        layers.append(('activation230', nn.ReLU(inplace=True)))

        layers.append(('convolution231', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm231', nn.BatchNorm2d(256)))

        layers.append(('activation231', nn.ReLU(inplace=True)))

        layers.append(('convolution232', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm232', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar38', Scale(0.2)))

        layers.append(('activation232', nn.ReLU(inplace=True)))

        layers.append(('convolution233', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm233', nn.BatchNorm2d(192)))

        layers.append(('activation233', nn.ReLU(inplace=True)))

        layers.append(('convolution234', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm234', nn.BatchNorm2d(192)))

        layers.append(('activation234', nn.ReLU(inplace=True)))

        layers.append(('convolution235', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm235', nn.BatchNorm2d(224)))

        layers.append(('activation235', nn.ReLU(inplace=True)))

        layers.append(('convolution236', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm236', nn.BatchNorm2d(256)))

        layers.append(('activation236', nn.ReLU(inplace=True)))

        layers.append(('convolution237', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm237', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar39', Scale(0.2)))

        layers.append(('activation237', nn.ReLU(inplace=True)))

        layers.append(('convolution238', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm238', nn.BatchNorm2d(192)))

        layers.append(('activation238', nn.ReLU(inplace=True)))

        layers.append(('convolution239', nn.Conv2d(
            in_channels=2080,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm239', nn.BatchNorm2d(192)))

        layers.append(('activation239', nn.ReLU(inplace=True)))

        layers.append(('convolution240', nn.Conv2d(
            in_channels=192,
            out_channels=224,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )))

        layers.append(('batchnorm240', nn.BatchNorm2d(224)))

        layers.append(('activation240', nn.ReLU(inplace=True)))

        layers.append(('convolution241', nn.Conv2d(
            in_channels=224,
            out_channels=256,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=True
        )))

        layers.append(('batchnorm241', nn.BatchNorm2d(256)))

        layers.append(('activation241', nn.ReLU(inplace=True)))

        layers.append(('convolution242', nn.Conv2d(
            in_channels=448,
            out_channels=2080,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm242', nn.BatchNorm2d(2080)))

        layers.append(('_mulscalar40', Scale(1.0)))

        layers.append(('convolution243', nn.Conv2d(
            in_channels=2080,
            out_channels=1536,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True
        )))

        layers.append(('batchnorm243', nn.BatchNorm2d(1536)))

        layers.append(('activation242', nn.ReLU(inplace=True)))

        layers.append(('bn1', nn.BatchNorm2d(1536)))

        layers.append(('pre_fc1', nn.Linear(
            in_features=221184,
            out_features=512,
            bias=True
        )))

        layers.append(('fc1', nn.BatchNorm2d(512)))

        layers.append(('fc7', nn.Linear(
            in_features=512,
            out_features=10572,
            bias=True
        )))

        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)

def load_model(weights_path):
    model = ConvertedModel()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model
