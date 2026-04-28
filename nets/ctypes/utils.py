from nets.ctypes.typedict import *
import numpy as np
import math
import gc
import time
import torch
import ctypes # ctypes를 직접 사용

from . import params

# --- C 라이브러리 및 상수 정의 ---
# (이 상수들은 C 코드의 .h 파일과 정확히 일치해야 합니다)
args = params.get_params()
SIFE_L = args.sife_l
TERMS = args.terms
UNKNOWN = args.unknown
SIFE_NMODULI = args.sife_nmoduli
SIFE_N = args.sife_n

try:
    library = ctypes.cdll.LoadLibrary("./caltest.so")
except OSError as e:
    print(f"FATAL ERROR: './caltest.so' 라이브러리를 로드할 수 없습니다.")
    print(f"C 코드를 컴파일했는지, 파일이 올바른 위치에 있는지 확인하세요.")
    print(f"Error: {e}")
    raise

# --- 전역 키 버퍼 (Python이 메모리 소유) ---
mpk = (ctypes.c_uint32 * (SIFE_L+1) * SIFE_NMODULI * SIFE_N)()
msk = (ctypes.c_uint32 * SIFE_L * SIFE_NMODULI * SIFE_N)()

# --- C 함수 프로토타입 정의 ---
library.makeKeys.argtypes = [typedict['uint32*'], typedict['uint32*']]
library.makeKeys.restype = ctypes.c_void_p

library.loadSecInput1x1.argtypes = [
    typedict['uint32*'], # encryptedImage (출력)
    typedict['double*'], # image (입력)
    typedict['int*'],    # imageSize
    typedict['int'],     # stride
    typedict['uint32*']  # mpk
]
library.loadSecInput1x1.restype = typedict['void']

library.convolution1x1.argtypes = [
    typedict['double*'], # output (출력)
    typedict['uint32*'], # secImage (입력)
    typedict['int*'],    # imageSize
    typedict['double*'], # filter (입력)
    typedict['int*'],    # filterSize
    typedict['int'],     # stride
    typedict['uint32*']  # msk
]
library.convolution1x1.restype = typedict['void']

# --- 직렬화 헬퍼 함수 (작은 메타데이터 배열용) ---
def flatten(l):
    """Python 리스트를 1차원으로 평탄화합니다."""
    for item in l:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def carr(arr, type_str="double"):
    """Python 리스트를 Ctypes 배열로 변환합니다."""
    if type_str not in typedict:
        raise ValueError(f"Unknown type_str: {type_str}")
    flat = list(flatten(arr))
    ArrayType = typedict[type_str] * len(flat)
    return ArrayType(*flat)

def ptr_to_list(ptr, length):
    """(디버깅용) C 포인터를 Python 리스트로 변환합니다."""
    return [ptr[i] for i in range(length)]

# --- 파이썬 래퍼 함수 ---

def makeKeys():
    """C 라이브러리의 makeKeys를 호출하여 전역 mpk, msk를 채웁니다."""
    # print("Generating SIFE keys...")
    library.makeKeys(
        ctypes.cast(mpk, typedict['uint32*']), 
        ctypes.cast(msk, typedict['uint32*'])
    )
    # print("Keys generated.")
    return mpk, msk
    
def loadSecDataConv1x1(image_tensor, imageShape, stride=1):
    """
    Python 텐서 또는 NumPy 배열을 C 라이브러리를 통해 암호화합니다.
    (★ stride > 1 일 때의 메모리 레이아웃 버그 수정됨)
    """
    
    B, C, H, W = imageShape

    # (★ 핵심 수정 1 - "Split Layer" 버그 수정)
    # C 코드와 동일하게 outputWidth/Height를 기준으로 메모리 크기를 계산합니다.
    outputWidth = math.floor((W - 1) / stride) + 1
    outputHeight = math.floor((H - 1) / stride) + 1
    channelSplit = math.ceil(C / SIFE_L)
    
    # C 코드의 `encryptedImage_t` 캐스팅과 정확히 일치하는 크기
    # (B, H_out, W_out, CS, TERMS, 2, L+1, NMOD, N)
    try:
        buffer_size = B * outputHeight * outputWidth * channelSplit * TERMS * 2 * (SIFE_L+1) * SIFE_NMODULI * SIFE_N
        _secData_buffer = (ctypes.c_uint32 * buffer_size)()
    except MemoryError:
        print(f"FATAL ERROR: Failed to allocate memory for _secData_buffer.")
        print(f"Size: {buffer_size} elements ({buffer_size * 4 / (1024**3):.2f} GB)")
        raise
        
    _secData_ptr = ctypes.cast(_secData_buffer, typedict['uint32*'])
    
    # (★ 핵심 수정 2 - AttributeError 수정)
    # image_tensor가 PyTorch 텐서가 아닌 NumPy 배열로 전달되었으므로,
    # .detach().cpu().numpy() 호출을 제거하고 .astype()만 사용합니다.
    # C 함수가 `double*`를 요구하므로, `np.float64` (double)로 변환합니다.
    if not isinstance(image_tensor, np.ndarray):
        # 만약 PyTorch 텐서가 들어올 경우에 대비 (안전 장치)
        image_np = image_tensor.detach().cpu().numpy().astype(np.float64)
    else:
        # NumPy 배열이 직접 들어온 경우 (현재 상황)
        image_np = image_tensor.astype(np.float64) 
        
    image_ptr = image_np.ctypes.data_as(typedict['double*'])

    # imageShape는 크기가 작으므로 carr 사용
    imageShape_carr = carr(list(imageShape), "int") 

    # C 함수 호출
    library.loadSecInput1x1(
        _secData_ptr, 
        image_ptr, 
        imageShape_carr,
        stride, 
        ctypes.cast(mpk, typedict['uint32*'])
    )
    
    # Python이 메모리(_secData_buffer)를 계속 관리하도록 버퍼 객체 자체를 반환합니다.
    return _secData_buffer 

def conv1x1(secData_buffer, imageShape, filter_tensor, stride=1):
    """
    암호화된 데이터를 C 라이브러리를 통해 복호화/컨볼루션합니다.
    """
    
    B, C, H, W = imageShape
    # filter_tensor는 [Out_C, In_C, K, K] 형태
    
    # (★ AttributeError 방지)
    # filter_tensor도 PyTorch 텐서가 아닐 수 있으므로 확인
    if isinstance(filter_tensor, np.ndarray):
        filter_np = filter_tensor.astype(np.float64)
        F_out, F_in, _, _ = filter_np.shape
    else:
        # PyTorch 텐서인 경우
        F_out, F_in, _, _ = filter_tensor.shape 
        filter_np = filter_tensor.detach().cpu().numpy().astype(np.float64)

    if C != F_in:
        print(f"Warning: Image In-Channels ({C}) != Filter In-Channels ({F_in})")

    # 출력 크기 계산
    outputWidth, outputHeight = math.floor((W - 1) / stride) + 1, math.floor((H - 1) / stride) + 1
    
    # C 함수가 결과를 채울 출력 버퍼 생성 (C는 double*에 씀)
    _res_shape = (B, F_out, outputHeight, outputWidth)
    buffer_size = B * F_out * outputHeight * outputWidth
    _res_buffer = (ctypes.c_double * buffer_size)()
    _res_ptr = ctypes.cast(_res_buffer, typedict['double*'])
    
    # (★ 핵심 수정 3 - 데이터 타입 및 효율성)
    # 필터를 C가 요구하는 'double' (float64)로 변환하고 포인터를 직접 전달합니다.
    filter_ptr = filter_np.ctypes.data_as(typedict['double*'])
    
    # (★ 핵심 수정 4 - 필터 Shape 버그)
    # C가 요구하는 필터 shape [Out_Channels, In_Channels] (2개 원소)만 전달
    filterShape_carr = carr([F_out, F_in], "int")
    imageShape_carr = carr(list(imageShape), "int")

    # C 함수 호출
    library.convolution1x1(
        _res_ptr,
        ctypes.cast(secData_buffer, typedict['uint32*']), # secData 버퍼 포인터
        imageShape_carr,
        filter_ptr,                                     # double* 필터 포인터
        filterShape_carr,
        stride,
        ctypes.cast(msk, typedict['uint32*'])
    )
    
    # C 버퍼(_res_buffer)를 numpy 배열로 변환 (메모리 복사 없음)
    _resNp = np.ctypeslib.as_array(_res_buffer).reshape(_res_shape)
    
    # 암호화된 데이터는 더 이상 필요 없으므로 명시적으로 삭제
    del secData_buffer
    gc.collect()
    
    # numpy 배열을 반환 (이후 PyTorch 모델에서 .from_numpy()로 텐서 변환)
    return _resNp