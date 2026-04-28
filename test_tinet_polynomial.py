from nets.ctypes.params import CHECKPOINT_PATH
from nets.ctypes.params import set_params
from nets.pure.resnet50_1_tinet_pure_polynomial import Bottleneck_polynomial, ResNet_polynomial
# from nets.ppml.resnet50_1_tinet import Bottleneck, ResNet
from nets.pure.resnet50_1_tinet_pure import Bottleneck_pure, ResNet_pure
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image
import utils
import argparse
from tiny_imagenet_dataset import *
from argparse import Namespace

parser = argparse.ArgumentParser(description='ResNet Test')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
parser.add_argument('--model', type=int, default=1, help='model number')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
parser.add_argument('--print-freq', type=int, default=1, help='print frequency')
parser.add_argument('--sife-l', type=int, default=128, help='sife length')
parser.add_argument('--terms', type=int, default=2, help='number of terms')
parser.add_argument('--unknown', type=int, default=16, help='number of unknown')
args = parser.parse_args()

set_params(Namespace(sife_l=args.sife_l, terms=args.terms, unknown=args.unknown))

from nets.ctypes.utils import makeKeys

# --- 하이퍼파라미터 및 경로 설정 (기존과 동일) ---
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
CUSTOM_CONV_LAYER_INDEX = args.cusin

WEIGHT_PATH = CHECKPOINT_PATH[CUSTOM_CONV_LAYER_INDEX - 1]

# --- 메인 실행부 ---
device = torch.device("cpu")

train_dir = os.path.join("/datasets01/imagenet_full_size/061417/", "train")
val_dir = os.path.join("/datasets01/imagenet_full_size/061417/", "val")
dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=None,
)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=NUM_WORKERS, pin_memory=True
)

log_suffix = ""
print_freq = args.print_freq

model_pure = ResNet_pure(Bottleneck_pure, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
# model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
model_polynomial = ResNet_polynomial(Bottleneck_polynomial, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

model_pure.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model_polynomial.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)

checkpoint = torch.load(WEIGHT_PATH, map_location="cpu")
model_polynomial.load_state_dict(checkpoint)
model_pure.load_state_dict(checkpoint)
# model.load_state_dict(checkpoint["model"])

model_polynomial.to(device)
model_pure.to(device)
# model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0)

# 평가 루프
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model_polynomial.eval()
model_pure.eval()
# model.eval()
metric_logger = utils.MetricLogger(delimiter="  ")
header = f"Test: {log_suffix}"

num_processed_samples = 0
total_samples = 0
# 맞은 개수를 '샘플 수' 단위로 저장할 변수
accumulated_correct = 0
accumulated_correct_pure = 0

import time

# ... (기존 설정 코드 생략) ...

with torch.inference_mode():
    makeKeys() 
    
    for image, target in metric_logger.log_every(data_loader_test, print_freq, header):
        image = image.to(device)
        target = target.to(device)
        batch_size = image.shape[0]
        
        # --- 모델별 추론 및 시간 측정 ---
        # 1. Polynomial
        st = time.perf_counter()
        output_poly = model_polynomial(image)
        t_poly = time.perf_counter() - st
        
        # 2. Pure
        st = time.perf_counter()
        output_pure = model_pure(image)
        t_pure = time.perf_counter() - st

        # 3. PPML
        # st = time.perf_counter()
        # output_ppml = model(image)
        # t_ppml = time.perf_counter() - st

        # --- 지표 계산 ---
        acc1_poly, acc5_poly = utils.accuracy(output_poly, target, topk=(1, 5))
        acc1_pure, acc5_pure = utils.accuracy(output_pure, target, topk=(1, 5))
        # acc1_ppml, acc5_ppml = utils.accuracy(output_ppml, target, topk=(1, 5))
        # mae = torch.abs(output_poly - output_ppml).mean().item()

        # --- MetricLogger 실시간 업데이트 ---
        # update()에 넣어주면 log_every가 매번 평균을 계산해서 화면에 출력합니다.
        metric_logger.update(
            # mae=mae,
            t_poly=t_poly,   # 실시간 평균 polynomial 시간
            t_pure=t_pure,   # 실시간 평균 pure 시간
            # t_ppml=t_ppml    # 실시간 평균 ppml 시간
        )
        
        # Accuracy 지표들 업데이트 (n=batch_size 적용)
        metric_logger.meters['acc1_poly'].update(acc1_poly.item(), n=batch_size)
        metric_logger.meters['acc5_poly'].update(acc5_poly.item(), n=batch_size)
        metric_logger.meters['acc1_pure'].update(acc1_pure.item(), n=batch_size)
        metric_logger.meters['acc5_pure'].update(acc5_pure.item(), n=batch_size)
        # metric_logger.meters['acc1_ppml'].update(acc1_ppml.item(), n=batch_size)
        # metric_logger.meters['acc5_ppml'].update(acc5_ppml.item(), n=batch_size)

# 모든 GPU 프로세스 동기화 (분산 환경일 경우)
metric_logger.synchronize_between_processes()

# --- 최종 결과 요약 ---
print(f"\nindex={args.cusin}/sife_l={args.sife_l}/terms={args.terms}/unknown={args.unknown} {header} 최종 요약:")
print(f"  [Pure]       Acc@1: {metric_logger.acc1_pure.global_avg:.3f} | Acc@5: {metric_logger.acc5_pure.global_avg:.3f} | Time: {metric_logger.t_pure.global_avg:.4f}s")
print(f"  [Polynomial] Acc@1: {metric_logger.acc1_poly.global_avg:.3f} | Acc@5: {metric_logger.acc5_poly.global_avg:.3f} | Time: {metric_logger.t_poly.global_avg:.4f}s")
# print(f"  [PPML]       Acc@1: {metric_logger.acc1_ppml.global_avg:.3f} | Acc@5: {metric_logger.acc5_ppml.global_avg:.3f} | Time: {metric_logger.t_ppml.global_avg:.4f}s")
# print(f"  MAE (Poly vs PPML): {metric_logger.mae.global_avg:.6f}")