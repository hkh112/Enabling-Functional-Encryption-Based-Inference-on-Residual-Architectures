import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes   # Load the shared library
import matplotlib.pyplot as plt
import math
import gc

# 1. (변경) 코드 2에서 실제 구현 함수들을 임포트합니다.
from nets.ctypes.utils import *
from nets.ppml.custom_conv_2d import CustomConv2D

# class TimeManager:
#     _client_start = None
#     _client_end = None

#     _server_start = None
#     _server_end = None

#     _server_start2 = None
#     _server_end2 = None
    
#     def client_start(self):
#         self._client_start = time.time()
    
#     def client_end(self):
#         self._client_end = time.time()
#         return self._client_end - self._client_start

#     def server_start(self):
#         self._server_start = time.time()
    
#     def server_end(self):
#         self._server_end = time.time()
#         return self._server_end - self._server_start

#     def server_start2(self):
#         self._server_start2 = time.time()
    
#     def server_end2(self):
#         self._server_end2 = time.time()
#         return self._server_end2 - self._server_start2

#     def get_client_time(self):
#         return self._client_end - self._client_start

#     def get_server_time(self):
#         return self._server_end - self._server_start

#     def get_server_time2(self):
#         return self._server_end2 - self._server_start2

# 2. (변경) 'return input' 대신 코드 2의 실제 CTypes 함수 호출 로직으로 교체합니다.
def encrypt_data(input, batch_shape, stride):
    secData = []
    for batch in input:
        secData.append(loadSecDataConv1x1(batch, batch_shape, stride))
    return secData

class Bottleneck_naive(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_custom_conv=False, planes_per_use_custom_planes=4, custom_conv=None):
        super(Bottleneck_naive, self).__init__()

        self.use_custom_conv = use_custom_conv
        self.custom_conv = custom_conv
        self.stride = stride
        # self.timer = timer

        # Original Route에서 Custom Convolution 제거 (FE 제거)
        
        if use_custom_conv:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)   # 1x1 conv
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
            # if self.timer != False:
            #     self.timer.client_end()

            if self.stride > 1:
                x_sub = x[:, :, ::self.stride, ::self.stride]
            else:
                x_sub = x
                
            batch_shape = [1, x_sub.shape[1], x_sub.shape[2], x_sub.shape[3]]
            x_max = torch.max(torch.abs(x_sub)) + 1e-8
            enc = encrypt_data(x_sub / x_max, batch_shape, stride=1)
            # out = self.conv1(enc, batch_shape, stride=1) * x_max

        #     # if self.timer != False:
        #     #     self.timer.server_start()
        # else:

        # Original Route에서 Custom Convolution 제거 (FE 제거)

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
                # enc: List of ctypes buffers (one per batch item)
                # batch_shape: [1, C, H_out, W_out]
                H_out, W_out = batch_shape[2], batch_shape[3]
                
                extracted_batches = []
                for buf in enc:
                    # buffer to numpy array
                    arr = np.frombuffer(buf, dtype=np.uint32)
                    num_pixels = H_out * W_out
                    block_size = len(arr) // num_pixels
                    
                    # 각 픽셀 블록의 가장 마지막 원소 추출
                    # arr shape: (H_out * W_out * block_size,)
                    # reshape to (num_pixels, block_size) and take last
                    last_elements = arr.reshape(num_pixels, block_size)[:, -1]
                    extracted_batches.append(last_elements.reshape(H_out, W_out))
                
                # (B, 1, H_out, W_out) 텐서 생성
                res_temp = torch.from_numpy(np.array(extracted_batches)).unsqueeze(1).float()
                
                # weight의 in_channels에 맞추어 채널 복제 (B, in_channels, H_out, W_out)
                res_temp = res_temp.repeat(1, self.custom_conv.in_channels, 1, 1)
                
                # PyTorch conv2d 수행 (stride=1, padding=0 으로 고정 - CustomConv2D의 내부 동작 모사)
                res = F.conv2d(res_temp, self.custom_conv.weight, self.custom_conv.bias, stride=1, padding=0) * x_max
            else:
                res = residual
            residual = self.downsample(res)


        out += residual
        out = self.relu(out)

        # [메모리 누수 방지] C++ 연산이 모두 끝난 후 거대한 메모리 공간을 강제로 반환합니다.
        if enc is not None:
            del enc
            import gc
            gc.collect()

        return out
 

class ResNet_naive(nn.Module):
    def __init__(self, block, layers, num_classes=10, custom_conv_layer_index=1):
        
        self.inplanes = 64
        self.custom_conv_layer_index = custom_conv_layer_index
        super(ResNet_naive, self).__init__()

        # if timer != False:
        #     self.timer = TimeManager()

        self.custom_conv_layer_index = custom_conv_layer_index
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
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

        # if self.timer != False:
        #     self.timer.client_start()
        
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

        # if self.timer != False:
        #     self.timer.server_end2()
        #     print(f"Client Prep Time: {self.timer.get_client_time():.6f}s")
        #     print(f"Server Proc Time: {self.timer.get_server_time():.6f}s")
        #     print(f"Server Proc Time2: {self.timer.get_server_time2():.6f}s")

        return x