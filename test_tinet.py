from nets.ctypes.params import CHECKPOINT_PATH
from nets.pure.resnet50_1_tinet_pure_polynomial import Bottleneck_polynomial
from nets.pure.resnet50_1_tinet_pure_polynomial import ResNet_polynomial
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image
import utils
# from nets.resnet50_2_tinet import ResNet, Bottleneck
import argparse
from tiny_imagenet_dataset import *
from nets.ctypes.params import set_params
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
# --- [추가] 테스트 이미지 개수 제한 인자 ---
parser.add_argument('--test-num', type=int, default=0, help='number of test images (0 for all)')
# ---------------------------------------
args = parser.parse_args()

set_params(Namespace(sife_l=args.sife_l, terms=args.terms, unknown=args.unknown))

from nets.ctypes.utils import makeKeys
from nets.pure.resnet50_1_tinet_pure import Bottleneck_pure
from nets.pure.resnet50_1_tinet_pure import ResNet_pure
from nets.ppml.resnet50_1_tinet import Bottleneck, ResNet

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
model_pure.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)

model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).cpu()
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)

model_polynomial = ResNet_polynomial(Bottleneck_polynomial, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).cpu()
model_polynomial.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)

checkpoint = torch.load(WEIGHT_PATH, map_location="cpu")
model.load_state_dict(checkpoint)
model_pure.load_state_dict(checkpoint)
model_polynomial.load_state_dict(checkpoint)

model.cpu()
model_pure.to(device)
model_polynomial.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0)

# 평가 루프
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()
model_pure.eval()
model_polynomial.eval()
metric_logger = utils.MetricLogger(delimiter="  ")
header = f"Test: {log_suffix}"

num_processed_samples = 0
total_samples = 0
# 맞은 개수를 '샘플 수' 단위로 저장할 변수
accumulated_correct = 0
accumulated_correct_pure = 0

with torch.inference_mode():
    makeKeys()
    for image, target in metric_logger.log_every(data_loader_test, print_freq, header):
        # --- [추가] 지정된 개수 초과 시 중단 ---
        if args.test_num > 0 and num_processed_samples >= args.test_num:
            break
        
        # 마지막 배치에서 test_num을 정확히 맞추기 위한 슬라이싱
        if args.test_num > 0 and (num_processed_samples + image.shape[0] > args.test_num):
            diff = args.test_num - num_processed_samples
            image = image[:diff]
            target = target[:diff]
        # -----------------------------------

        image = image.cpu()
        target = target.cpu()
        batch_size = image.shape[0]
        
        output = model(image)
        output_pure = model_pure(image)
        output_polynomial = model_polynomial(image)
        
        # 1. 배치의 정확도(%) 가져오기
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        acc1_p, acc5_p = utils.accuracy(output_pure, target, topk=(1, 5))
        acc1_poly, acc5_poly = utils.accuracy(output_polynomial, target, topk=(1, 5))
        mae = torch.abs(output - output_polynomial).mean().item()

        # 2. MetricLogger 업데이트 (배치당 값을 그대로 전달)
        # logger가 알아서 (값 * n)의 합계를 유지하다가 나중에 평균을 내줍니다.
        metric_logger.update(loss=criterion(output, target).item())
        metric_logger.update(loss_pure=criterion(output_pure, target).item())
        metric_logger.update(mae=mae)
        
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['acc1_pure'].update(acc1_p.item(), n=batch_size)
        metric_logger.meters['acc5_pure'].update(acc5_p.item(), n=batch_size)
        metric_logger.meters['acc1_poly'].update(acc1_poly.item(), n=batch_size)
        metric_logger.meters['acc5_poly'].update(acc5_poly.item(), n=batch_size)

        num_processed_samples += batch_size

metric_logger.synchronize_between_processes()

# 최종 출력: metric_logger의 global_avg가 전체 평균 정확도입니다.
print(f"{header} "
      f"Acc@1 {metric_logger.acc1.global_avg:.3f} "
      f"Acc@1_pure {metric_logger.acc1_pure.global_avg:.3f} "
      f"Acc@1_poly {metric_logger.acc1_poly.global_avg:.3f} "
      f"Acc@5 {metric_logger.acc5.global_avg:.3f} "
      f"Acc@5_pure {metric_logger.acc5_pure.global_avg:.3f} "
      f"Acc@5_poly {metric_logger.acc5_poly.global_avg:.3f} "
      f"MAE {metric_logger.mae.global_avg:.6f}")