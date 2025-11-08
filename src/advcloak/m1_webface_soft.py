import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class M1WebfaceSoft(nn.Module):

    
    def __init__(self, weight_file):
        super(M1WebfaceSoft, self).__init__()
        global _weights_dict
        self.to(torch.device("cuda"))
        _weights_dict = load_weights(weight_file)

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _old_from_numpy = torch.from_numpy
        def _from_numpy_and_move(x):
            return _old_from_numpy(x).to(_device)

        torch.from_numpy = _from_numpy_and_move

        self.conv_1_conv2d = self.__conv(2, name='conv_1_conv2d', in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv_1_batchnorm = self.__batch_normalization(2, 'conv_1_batchnorm', num_features=32, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_2_dw_conv2d = self.__conv(2, name='conv_2_dw_conv2d', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=32, bias=False)
        self.conv_2_dw_batchnorm = self.__batch_normalization(2, 'conv_2_dw_batchnorm', num_features=32, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_2_conv2d = self.__conv(2, name='conv_2_conv2d', in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_2_batchnorm = self.__batch_normalization(2, 'conv_2_batchnorm', num_features=64, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_3_dw_conv2d = self.__conv(2, name='conv_3_dw_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=64, bias=False)
        self.conv_3_dw_batchnorm = self.__batch_normalization(2, 'conv_3_dw_batchnorm', num_features=64, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_3_conv2d = self.__conv(2, name='conv_3_conv2d', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_3_batchnorm = self.__batch_normalization(2, 'conv_3_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_4_dw_conv2d = self.__conv(2, name='conv_4_dw_conv2d', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=128, bias=False)
        self.conv_4_dw_batchnorm = self.__batch_normalization(2, 'conv_4_dw_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_4_conv2d = self.__conv(2, name='conv_4_conv2d', in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_4_batchnorm = self.__batch_normalization(2, 'conv_4_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_5_dw_conv2d = self.__conv(2, name='conv_5_dw_conv2d', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=128, bias=False)
        self.conv_5_dw_batchnorm = self.__batch_normalization(2, 'conv_5_dw_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_5_conv2d = self.__conv(2, name='conv_5_conv2d', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_5_batchnorm = self.__batch_normalization(2, 'conv_5_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_6_dw_conv2d = self.__conv(2, name='conv_6_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv_6_dw_batchnorm = self.__batch_normalization(2, 'conv_6_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_6_conv2d = self.__conv(2, name='conv_6_conv2d', in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_6_batchnorm = self.__batch_normalization(2, 'conv_6_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_7_dw_conv2d = self.__conv(2, name='conv_7_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=256, bias=False)
        self.conv_7_dw_batchnorm = self.__batch_normalization(2, 'conv_7_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_7_conv2d = self.__conv(2, name='conv_7_conv2d', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_7_batchnorm = self.__batch_normalization(2, 'conv_7_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_8_dw_conv2d = self.__conv(2, name='conv_8_dw_conv2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=512, bias=False)
        self.conv_8_dw_batchnorm = self.__batch_normalization(2, 'conv_8_dw_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_8_conv2d = self.__conv(2, name='conv_8_conv2d', in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_8_batchnorm = self.__batch_normalization(2, 'conv_8_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_9_dw_conv2d = self.__conv(2, name='conv_9_dw_conv2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=512, bias=False)
        self.conv_9_dw_batchnorm = self.__batch_normalization(2, 'conv_9_dw_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_9_conv2d = self.__conv(2, name='conv_9_conv2d', in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_9_batchnorm = self.__batch_normalization(2, 'conv_9_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_10_dw_conv2d = self.__conv(2, name='conv_10_dw_conv2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=512, bias=False)
        self.conv_10_dw_batchnorm = self.__batch_normalization(2, 'conv_10_dw_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_10_conv2d = self.__conv(2, name='conv_10_conv2d', in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_10_batchnorm = self.__batch_normalization(2, 'conv_10_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_11_dw_conv2d = self.__conv(2, name='conv_11_dw_conv2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=512, bias=False)
        self.conv_11_dw_batchnorm = self.__batch_normalization(2, 'conv_11_dw_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_11_conv2d = self.__conv(2, name='conv_11_conv2d', in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_11_batchnorm = self.__batch_normalization(2, 'conv_11_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_12_dw_conv2d = self.__conv(2, name='conv_12_dw_conv2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=512, bias=False)
        self.conv_12_dw_batchnorm = self.__batch_normalization(2, 'conv_12_dw_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_12_conv2d = self.__conv(2, name='conv_12_conv2d', in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_12_batchnorm = self.__batch_normalization(2, 'conv_12_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_13_dw_conv2d = self.__conv(2, name='conv_13_dw_conv2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), groups=512, bias=False)
        self.conv_13_dw_batchnorm = self.__batch_normalization(2, 'conv_13_dw_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_13_conv2d = self.__conv(2, name='conv_13_conv2d', in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_13_batchnorm = self.__batch_normalization(2, 'conv_13_batchnorm', num_features=1024, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_14_dw_conv2d = self.__conv(2, name='conv_14_dw_conv2d', in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), groups=1024, bias=False)
        self.conv_14_dw_batchnorm = self.__batch_normalization(2, 'conv_14_dw_batchnorm', num_features=1024, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_14_conv2d = self.__conv(2, name='conv_14_conv2d', in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_14_batchnorm = self.__batch_normalization(2, 'conv_14_batchnorm', num_features=1024, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.bn1 = self.__batch_normalization(2, 'bn1', num_features=1024, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.pre_fc1 = self.__dense(name = 'pre_fc1', in_features = 50176, out_features = 512, bias = True)
        self.fc1 = self.__batch_normalization(0, 'fc1', num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.fc7 = self.__dense(name = 'fc7', in_features = 512, out_features = 10572, bias = True)

    def forward(self, x):
        x = (x * 255./2) + 127.5
        self.minusscalar0_second = torch.autograd.Variable(
            torch.from_numpy(_weights_dict['minusscalar0_second']['value']), requires_grad=False
        ).to(x.device)
        self.mulscalar0_second = torch.autograd.Variable(
            torch.from_numpy(_weights_dict['mulscalar0_second']['value']), requires_grad=False
        ).to(x.device)
        minusscalar0 = x - self.minusscalar0_second
        mulscalar0   = minusscalar0 * self.mulscalar0_second

        # --- Layer 1 ---
        conv_1_conv2d_pad = F.pad(mulscalar0, (1, 1, 1, 1))
        conv_1_conv2d     = self.conv_1_conv2d(conv_1_conv2d_pad)
        conv_1_batchnorm   = self.conv_1_batchnorm(conv_1_conv2d)
        num_channels = conv_1_batchnorm.shape[1]
        conv_1_relu = F.prelu(conv_1_batchnorm, torch.ones(num_channels, device=x.device))

        # --- Layer 2 ---
        conv_2_dw_conv2d_pad = F.pad(conv_1_relu, (1, 1, 1, 1))
        conv_2_dw_conv2d     = self.conv_2_dw_conv2d(conv_2_dw_conv2d_pad)
        conv_2_dw_batchnorm   = self.conv_2_dw_batchnorm(conv_2_dw_conv2d)
        num_channels = conv_2_dw_batchnorm.shape[1]
        conv_2_dw_relu = F.prelu(conv_2_dw_batchnorm, torch.ones(num_channels, device=x.device))
        conv_2_conv2d     = self.conv_2_conv2d(conv_2_dw_relu)
        conv_2_batchnorm   = self.conv_2_batchnorm(conv_2_conv2d)
        num_channels = conv_2_batchnorm.shape[1]
        conv_2_relu = F.prelu(conv_2_batchnorm, torch.ones(num_channels, device=x.device))

        # --- Layer 3 ---
        conv_3_dw_conv2d_pad = F.pad(conv_2_relu, (1, 1, 1, 1))
        conv_3_dw_conv2d     = self.conv_3_dw_conv2d(conv_3_dw_conv2d_pad)
        conv_3_dw_batchnorm   = self.conv_3_dw_batchnorm(conv_3_dw_conv2d)
        num_channels = conv_3_dw_batchnorm.shape[1]
        conv_3_dw_relu = F.prelu(conv_3_dw_batchnorm, torch.ones(num_channels, device=x.device))
        conv_3_conv2d     = self.conv_3_conv2d(conv_3_dw_relu)
        conv_3_batchnorm   = self.conv_3_batchnorm(conv_3_conv2d)
        num_channels = conv_3_batchnorm.shape[1]
        conv_3_relu = F.prelu(conv_3_batchnorm, torch.ones(num_channels, device=x.device))

        # --- Repeat the same pattern for conv_4 to conv_14 ---
        prev_relu = conv_3_relu
        for i in range(4, 15):
            dw_conv2d = getattr(self, f'conv_{i}_dw_conv2d', None)
            dw_batchnorm = getattr(self, f'conv_{i}_dw_batchnorm', None)
            conv2d = getattr(self, f'conv_{i}_conv2d', None)
            batchnorm = getattr(self, f'conv_{i}_batchnorm', None)

            if dw_conv2d is not None:
                dw_conv2d_pad = F.pad(prev_relu, (1, 1, 1, 1))
                dw_out = dw_conv2d(dw_conv2d_pad)
                dw_bn_out = dw_batchnorm(dw_out)
                num_channels = dw_bn_out.shape[1]
                prev_relu = F.prelu(dw_bn_out, torch.ones(num_channels, device=x.device))

            conv_out = conv2d(prev_relu)
            bn_out = batchnorm(conv_out)
            num_channels = bn_out.shape[1]
            prev_relu = F.prelu(bn_out, torch.ones(num_channels, device=x.device))

        bn1 = self.bn1(prev_relu)
        dropout0 = F.dropout(input=bn1, p=0.4, training=self.training, inplace=True)
        pre_fc1 = self.pre_fc1(dropout0.view(dropout0.size(0), -1))
        fc1     = self.fc1(pre_fc1)
        fc7     = self.fc7(fc1)
        softmax = F.softmax(fc7, dim=1)
        return fc1



    @staticmethod
    def __conv(dim, name, **kwargs):
        import torch
        import torch.nn as nn

        # create layer
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError(f"Conv dim={dim} not supported")

        # load MXNet weights
        w = torch.from_numpy(_weights_dict[name]['weights'])

        # depthwise conv check
        groups = kwargs.get('groups', 1)
        in_channels = kwargs['in_channels']
        if dim == 2 and groups > 1 and groups == in_channels:
            # MXNet: (kH, kW, inC, 1) → PyTorch: (inC, 1, kH, kW)
            w = w.permute(2, 3, 0, 1).contiguous()
        elif dim == 2 and w.ndim == 4:
            # regular conv: MXNet (kH,kW,inC,outC) → PyTorch (outC,inC,kH,kW)
            w = w.permute(3, 2, 0, 1).contiguous()
        elif dim == 3 and w.ndim == 5:
            w = w.permute(4, 3, 0, 1, 2).contiguous()
        elif dim == 1 and w.ndim == 3:
            w = w.permute(2, 1, 0).contiguous()
        else:
            # fallback: ensure shapes match
            if w.shape != layer.weight.shape:
                raise RuntimeError(f"[conv] Unexpected weight shape for {name}: {w.shape} -> expected {tuple(layer.weight.shape)}")

        # copy weights and bias
        layer.weight.data.copy_(w)
        if 'bias' in _weights_dict[name]:
            layer.bias.data.copy_(torch.from_numpy(_weights_dict[name]['bias']))

        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        w = torch.from_numpy(_weights_dict[name]['weights'])
        
        # MXNet->PyTorch: transpose to match PyTorch (out_features, in_features)
        if w.shape != layer.weight.shape:
            if w.shape[::-1] == layer.weight.shape:
                w = w.T.contiguous()
            else:
                raise RuntimeError(f"[Dense] Unexpected weight shape for {name}: {w.shape} -> expected {layer.weight.shape}")
        
        layer.weight.data.copy_(w)
        
        if 'bias' in _weights_dict[name]:
            layer.bias.data.copy_(torch.from_numpy(_weights_dict[name]['bias']))
        
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

