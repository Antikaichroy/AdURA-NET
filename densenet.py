# model implementation credit goes to https://github.com/pytorch/pytorch
# model design credit goes to https://arxiv.org/abs/1608.06993 

import re
import torch
#from mcbam import MCBAM
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
try:
    from torch.hub import load_state_dict_from_url
except:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torch import Tensor
from torch.jit.annotations import List
# Deformable convolution block adding offsets
from torchvision.ops import DeformConv2d
import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiScaleDCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDCN, self).__init__()
        self.offset = nn.Conv2d(in_channels = in_channels, out_channels = 18, kernel_size=1) # 18 because it is 2*3*3
        self.db1 = DeformConv2d(in_channels = in_channels, out_channels = out_channels, dilation = 1, padding = 1, kernel_size = 3)
        self.db3 = DeformConv2d(in_channels = in_channels, out_channels = out_channels, dilation = 3, padding = 3, kernel_size = 3)
        self.db5 = DeformConv2d(in_channels = in_channels, out_channels = out_channels, dilation = 5, padding = 5, kernel_size = 3)
        self.db7 = DeformConv2d(in_channels = in_channels, out_channels = out_channels, dilation = 7, padding = 7, kernel_size = 3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(4*out_channels, out_channels//4)
        self.fc2 = nn.Linear(out_channels//4, 4)
        self.down1 = nn.AvgPool2d(2) #2^1 group 1
        self.down2 = nn.AvgPool2d(4) #2^2 group 2
        self.down3 = nn.AvgPool2d(8) #2^3 group 3
        self.down4 = nn.AvgPool2d(16) # 2^4 group 4
        self.unshuffle1 = nn.PixelUnshuffle(downscale_factor = 16)
        self.unshuffle2 = nn.PixelUnshuffle(downscale_factor=8)
        self.unshuffle3 = nn.PixelUnshuffle(downscale_factor=4)
        self.unshuffle4 = nn.PixelUnshuffle(downscale_factor=2)
        self.branch_conv1 = nn.Conv2d(in_channels = out_channels//4, out_channels = out_channels//4, kernel_size = 3, padding = 1)
        self.branch_conv2 = nn.Conv2d(in_channels = out_channels//4, out_channels = out_channels//4, kernel_size = 3, padding = 1)
        self.branch_conv3 = nn.Conv2d(in_channels = out_channels//4, out_channels = out_channels//4, kernel_size = 3, padding = 1)
        self.branch_conv4 = nn.Conv2d(in_channels = out_channels//4, out_channels = out_channels//4, kernel_size = 3, padding = 1)
        # 256 + 64 + 16 + 4  = 340
        self.compress_conv = nn.Conv2d(in_channels = 340*(out_channels//4), out_channels = out_channels, kernel_size =3, padding = 1)
        #self.ms_dcn_out = {"offset": [], "features": []}
    def forward(self, x):
        raw_offset = self.offset(x)
        # now we need to restrict the range from -5 to 5

        offset = 5*torch.tanh(raw_offset)  # this will limit it to [-5,5]
        f1 = self.db1(x, offset) #branch 1
        f2 = self.db3(x, offset)
        f3 = self.db5(x, offset)
        f4 = self.db7(x, offset) #branch 4


        # now we need to do the global average pooling of each
        b1 = self.gap(f1).flatten(1)
        b2 = self.gap(f2).flatten(1)
        b3 = self.gap(f3).flatten(1)
        b4 = self.gap(f4).flatten(1)

        f = torch.cat([b1,b2,b3,b4], dim = 1) # B,4C   # concatenated_feature
        
        branch_weight_vector = F.softmax(self.fc2(F.relu(self.fc1(f))), dim = 1) # [a1,a2,a3,a4] # attention
        # the above will give us 4 scalar which needs to be multiplied to the b1 b2 b3 and b4 for the fusion
        #print(branch_weight_vector.shape)
        a1 = branch_weight_vector[:,0].view(-1,1,1,1) # takes the first value it has the shape [1,1,1,1]
        a2 = branch_weight_vector[:,1].view(-1,1,1,1) # takes the second value
        a3 = branch_weight_vector[:,2].view(-1,1,1,1) # takes the third 
        a4 = branch_weight_vector[:,3].view(-1,1,1,1) # takes the fourth


        fused_vector = a1*f1 + a2*f2 + a3*f3 + a4*f4  # [1, out_channels, H, W] if out_channels = 64 we will split it into 4 i.e 16*4

        # Now we need Hirarchical Recognized Pyramid
        # we need to split the fused result into 4 group, each group will use channel wise down sampling
        # now to increase the scale specificity we will split it into 4 groups G1, G2, G3, G4 along the channel dimension
        # each group will pefrom 2^j times down sampling where j is 1,2,3,4
        g1 = torch.split(fused_vector, split_size_or_sections = fused_vector.size(1)//4, dim = 1)[0]
        g2 = torch.split(fused_vector, split_size_or_sections = fused_vector.size(1)//4, dim = 1)[1]
        g3 = torch.split(fused_vector, split_size_or_sections = fused_vector.size(1)//4, dim = 1)[2]
        g4 = torch.split(fused_vector, split_size_or_sections = fused_vector.size(1)//4, dim = 1)[3]

        d1 = self.down1(g1)
        d2 = self.down2(g2)
        d3 = self.down3(g3)
        d4 = self.down4(g4)

        # upsample r = 16, 2^4
        original_h, original_w = x.size(2), x.size(3)
        u1 = F.interpolate(d1, scale_factor=16, mode = "nearest")
        u1 = self.branch_conv1(u1)
        u1 = self.unshuffle1(u1)

        u2 = F.interpolate(d2, scale_factor = 8, mode = "nearest")
        u2 = self.branch_conv2(u2)
        u2 = self.unshuffle2(u2)

        u3 = F.interpolate(d3, scale_factor = 4, mode = "nearest")
        u3 = self.branch_conv3(u3)
        u3 = self.unshuffle3(u3)

        u4 = F.interpolate(d4, scale_factor = 2, mode = "nearest")
        u4 = self.branch_conv4(u4)
        u4 = self.unshuffle4(u4)

        # bringing back to the original h, w

        u1 = F.interpolate(input = u1, size = (original_h, original_w), align_corners=False, mode = 'bilinear')
        u2 = F.interpolate(input = u2, size = (original_h, original_w), align_corners=False, mode = 'bilinear')
        u3 = F.interpolate(input = u3, size = (original_h, original_w), align_corners=False, mode = 'bilinear')
        u4 = F.interpolate(input = u4, size = (original_h, original_w), align_corners=False, mode = 'bilinear')

        final_concat = torch.cat([u1, u2, u3, u4], dim = 1) #[1, 5440, 28, 28] if img = 28,28, c = 64

        out = self.compress_conv(final_concat)
        #self.ms_dcn_out['offset'].append(offset)
        #self.ms_dcn_out['features'].append(out)
        return out, offset
    

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, momentum=0.01)),
        self.add_module('elu1', activation_func(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate, momentum=0.01)),
        self.add_module('elu2', activation_func(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.elu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.elu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, momentum=0.01))
        self.add_module('ELU1', activation_func(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        
        #self.add_module("mcbam", MCBAM(num_output_features))
        self.add_module("ELU2", activation_func(inplace=True))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=1, memory_efficient=False, last_activation=None):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features, momentum=0.01)),  #epsilon=0.001
            ('elu0', activation_func(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        #self.multi_dcn_out = {}
        # Each denseblock
        num_features = num_init_features
        self.features.add_module(f"MultiScaleDCN_0", MultiScaleDCN(num_features, num_features))
        #self.features.add_module(f"MCBAM_0", MCBAM(num_features))
        
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            #if i == 0:
                #self.features.add_module(f"MultiScaleDCN_{i+1}", MultiScaleDCN(num_features, num_features))
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features, momentum=0.01))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.num_classes = num_classes


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight) 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #features = self.features(x)
        multi_dcn_out = {}
        features = x
        for layer in self.features:
            if isinstance(layer, MultiScaleDCN):
                features, offset = layer(features)
                multi_dcn_out['offset'] = offset
                multi_dcn_out['features'] = features
            else:
                features = layer(features)
        out = F.relu(features, inplace=True) #F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out  
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        else:
            out = self.sigmoid(out)
        return multi_dcn_out, out


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'ELU.1', 'conv.1', 'norm.2', 'ELU.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|ELU|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    state_dict.pop('classifier.weight', None)
    state_dict.pop('classifier.bias', None)        
    model.load_state_dict(state_dict, strict=False)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def DenseNet121(pretrained=False, progress=True, activations='relu', **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    global activation_func
    activation_func = nn.ReLU if activations=='relu' else nn.ELU
    # print (activation_func)
    return _densenet('densenet121', 32, (6,12,24,16), 64, pretrained, progress,
                     **kwargs)


def DenseNet161(pretrained=False, progress=True, activations='relu', **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    global activation_func
    activation_func = nn.ReLU if activations=='relu' else nn.ELU
    # print (activation_func)
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def DenseNet169(pretrained=False, progress=True, activations='relu', **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    global activation_func
    activation_func = nn.ReLU if activations=='relu' else nn.ELU
    # print (activation_func)
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def DenseNet201(pretrained=False, progress=True, activations='relu', **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    global activation_func
    activation_func = nn.ReLU if activations=='relu' else nn.ELU
    # print (activation_func)
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)


# alias
densenet121 = DenseNet121
densenet161 = DenseNet161
densenet169 = DenseNet169
densenet201 = DenseNet201

