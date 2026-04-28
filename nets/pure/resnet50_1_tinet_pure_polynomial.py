import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes   # Load the shared library
import matplotlib.pyplot as plt
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.ctypes import params

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2D, self).__init__()
        self.stride = stride
        self.padding = padding
        # [수정] __init__ 시점에 최신 params를 가져오도록 수정
        current_args = params.get_params()
        self.terms_count = current_args.terms
        self.unknown = current_args.unknown

        print(f"CustomConv2D self.terms_count: {self.terms_count}")
        print(f"CustomConv2D self.unknown: {self.unknown}")

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def get_poly_tensors(self, x):
        """
        [핵심] -1 < x < 1 범위 전용 다항식 분해
        모든 TERMS를 소수점 아래 자릿수를 표현하는 데 사용합니다.
        """
        device = x.device
        poly_tensors = []
        
        # 1. 부호 분리 (양수 파트와 음수 파트의 절대값)
        current_pos = torch.clamp(x, min=0)
        current_neg = torch.clamp(x, max=0).abs()
        
        # 2. 모든 차수(0 ~ TERMS-1)에 대해 소수점 자릿수 추출
        # -1 < x < 1이므로 정수부를 떼는 과정 없이 바로 unknown을 곱하며 시작합니다.
        for _ in range(self.terms_count):
            # 자릿수를 올림 (예: 0.123 -> 1.23)
            current_pos = current_pos * self.unknown
            current_neg = current_neg * self.unknown
            
            # 정수부만 취함 (이것이 해당 차수의 다항식 계수가 됨)
            term_pos = current_pos.trunc()
            term_neg = current_neg.trunc()
            
            poly_tensors.append([term_pos, term_neg])
            
            # 다음 차수를 위해 다시 소수점 아래만 남김 (예: 1.23 -> 0.23)
            current_pos = torch.clamp(current_pos - term_pos, min=0)
            current_neg = torch.clamp(current_neg - term_neg, min=0)
            
        return poly_tensors

    def max_normalize(self, x):
        x_max = torch.max(torch.abs(x)) + 1e-8 # 절댓값 기준 최댓값 사용
        x = x / x_max
        return x, x_max

    def forward(self, x, batch_shape=None, stride=None):
        device = x.device
        target_stride = stride if stride is not None else self.stride

        # [추가] 최댓값 추출 (정규화 계수)
        # x_max와 w_max는 스칼라(Scalar) 값입니다.
        x_max = torch.max(torch.abs(x)) + 1e-8
        w_max = torch.max(torch.abs(self.weight)) + 1e-8

        # 1. 정규화된 텐서로 다항식 분해 수행
        # 이제 모든 원소는 -1 ~ 1 사이의 값을 가집니다.
        i_poly = self.get_poly_tensors(x / x_max)
        w_poly = self.get_poly_tensors(self.weight / w_max)

        # 2. 결과 텐서 초기화
        out_h = (x.shape[2] + 2*self.padding - self.weight.shape[2]) // target_stride + 1
        out_w = (x.shape[3] + 2*self.padding - self.weight.shape[3]) // target_stride + 1
        final_output = torch.zeros((x.shape[0], self.weight.shape[0], out_h, out_w), device=device)

        # 3. 다항식 내적 연산
        for it in range(self.terms_count):
            for ft in range(self.terms_count):
                pos_res = F.conv2d(i_poly[it][0], w_poly[ft][0], stride=target_stride, padding=self.padding) + \
                          F.conv2d(i_poly[it][1], w_poly[ft][1], stride=target_stride, padding=self.padding)
                
                neg_res = F.conv2d(i_poly[it][0], w_poly[ft][1], stride=target_stride, padding=self.padding) + \
                          F.conv2d(i_poly[it][1], w_poly[ft][0], stride=target_stride, padding=self.padding)
                
                term_scale = 1.0 / (self.unknown ** (it + ft + 2))
                
                # [수정] 계산된 결과에 정규화했던 계수(x_max * w_max)를 다시 곱해줍니다.
                final_output += (pos_res - neg_res) * term_scale * x_max * w_max

        if self.bias is not None:
            final_output += self.bias.view(1, -1, 1, 1)

        return final_output
    
def encrypt_data(input, batch_shape, stride):
    return input

class Bottleneck_polynomial(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_custom_conv=False, planes_per_use_custom_planes=4, custom_conv=None):
        super(Bottleneck_polynomial, self).__init__()

        self.use_custom_conv = use_custom_conv
        self.custom_conv = custom_conv
        self.stride = stride

        if use_custom_conv:
            self.conv1 = CustomConv2D(inplanes, planes, kernel_size=1, stride=stride, bias=False)   # 1x1 conv
            self.bn1 = nn.BatchNorm2d(planes)
            
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            
            self.conv3 = nn.Conv2d(planes, planes * 4 * planes_per_use_custom_planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4 * planes_per_use_custom_planes)
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

            self.bn1 = nn.BatchNorm2d(planes)
            
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride


        
    def forward(self, x):
        residual = x

        enc = None
        batch_shape = [1, x.shape[1], x.shape[2], x.shape[3]]

        if self.use_custom_conv:
            if self.stride > 1:
                x_sub = x[:, :, ::self.stride, ::self.stride]
            else:
                x_sub = x
                
            batch_shape = [1, x_sub.shape[1], x_sub.shape[2], x_sub.shape[3]]
            enc = encrypt_data(x_sub, batch_shape, stride=1)
            out = self.conv1(enc, batch_shape, stride=1)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)                                                                   # 3x3 conv  
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)                                                                   # 1x1 conv                                              
        out = self.bn3(out)

        if self.downsample is not None:
            if self.custom_conv is not None:
                res = self.custom_conv(enc, batch_shape, stride=1)
            else:
                res = residual
            residual = self.downsample(res)


        out += residual
        out = self.relu(out)
        return out
 

class ResNet_polynomial(nn.Module):
    def __init__(self, block, layers, num_classes=10, custom_conv_layer_index=1):
        
        self.inplanes = 64
        self.custom_conv_layer_index = custom_conv_layer_index
        super(ResNet_polynomial, self).__init__()

        self.custom_conv_layer_index = custom_conv_layer_index
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)                    #TODO: original conv 1x1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], skip_planes=32, layer_index=1, use_custom_planes=16)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, skip_planes=128, layer_index=2, use_custom_planes=64)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, skip_planes=256, layer_index=3, use_custom_planes=128)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, skip_planes=512, layer_index=4, use_custom_planes=256)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, skip_planes, stride=1, layer_index=1, use_custom_planes=16):
        
        downsample = None
        custom_conv = None
        use_custom = (layer_index == self.custom_conv_layer_index)
        
        if stride != 1 or self.inplanes != planes * block.expansion:# block.expansion=4

            if use_custom:
                custom_conv = CustomConv2D(self.inplanes, skip_planes, kernel_size=1, stride=stride, bias=False)
                downsample = nn.Sequential(
                    nn.BatchNorm2d(skip_planes),
                    nn.Conv2d(skip_planes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
       
        layers = []

        if use_custom:
            layers.append(block(self.inplanes, use_custom_planes, stride, downsample, use_custom_conv=use_custom, planes_per_use_custom_planes=planes // use_custom_planes, custom_conv=custom_conv))
            self.inplanes = use_custom_planes * block.expansion * (planes // use_custom_planes)
        
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, use_custom_conv=use_custom))
            self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x