# Anonymous FE-Compatible Neural Inference Repository

This repository contains the anonymized research code for reproducing the experiments reported in our paper on "Enabling Functional Encryption-Based Inference on Residual Architectures".

The main goal of this repository is to support reproducibility of the paper's experimental results, including:

- replacing the privacy-degenerate identity branch in residual networks with FE-admissible linear transformations;
- applying dimensionality constraints to reduce structural reconstruction risk;
- executing the proposed architecture under an FE-compatible inference pipeline;
- evaluating the accuracy, runtime, and cryptographic execution characteristics on public benchmark datasets.

This repository is provided for **academic research and reproducibility purposes only**.


## 1. Installation

### Environment

1) Set up the Python environment and install the necessary libraries.
```bash
# conda create -n fe_inference python=[x.xx]
# conda activate fe_inference
# [pytorch 등 필요한 라이브러리 설치 방법]
# You can check the command corresponding to your GPU at https://pytorch.org/get-started/previous-versions/
conda create -n qnet python=3.10
conda activate qnet
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==0.2.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas matplotlib numpy opencv-python-headless tqdm wandb beautifulsoup4 pyyaml scipy scikit-learn
conda install pyqt=6
```

2) You need to compile the FE-related C files. Modify CUDA_ROOT_DIR value to your CUDA installation path. Then, enter the sm_xx corresponding to your GPU into the makefile.
```bash
[makefile 어느 변수 고치는지, 어떻게 컴파일 하는지]
# Open and modify Makefile
nano Makefile
# Modify CUDA_ROOT_DIR and sm_xx values

# Compile
make clean & make
```


## 2. Data Preparation

This repository does **not** redistribute raw datasets.
The experiments use publicly available benchmark datasets:

- CIFAR-10
- Tiny-ImageNet

Users should download the datasets from their official sources and comply with the original dataset licenses, citation requirements, and terms of use.
Please check if the path of the downloaded dataset is the same as the path in the .py file.


## 3. Reproducing the Main Experiments

This section provides the commands for reproducing the main experimental results reported in the paper.

### 3.1 Train Baseline ResNet-50

These models are trained in plaintext and do not perform functional-encryption execution during training. 
For comparison with the FE-compatible pipeline, we train/evaluate variants corresponding to split points 1--4 and the dimensionality-constrained architecture used by the proposed method.
FE-Compatible ResNet-50 utilizes information learned from plaintext.

```bash
[Split, 평문 모델의 학습 명령어]
# python ~~~.py ~~~~~~
cd train

# If you want to train CIFAR-10 dataset with all splits, use this command.
chmod +x ./run_train_cifar10.sh
./run_train_cifar10.sh

# If you want to train CIFAR-10 dataset with each split, use this command.
python train_cifar10.py --cusin 1
python train_cifar10.py --cusin 2
python train_cifar10.py --cusin 3
python train_cifar10.py --cusin 4

# If you want to train Tiny-ImageNet dataset with all splits, use this command.
chmod +x ./run_train_tinet1.sh
./run_train_tinet1.sh

# If you want to train Tiny-ImageNet dataset with each split, use this command.
python train_tinet_1.py --cusin 1
python train_tinet_1.py --cusin 2
python train_tinet_1.py --cusin 3
python train_tinet_1.py --cusin 4

```
Here, x specifies the split position, where x ∈ {1,2,3,4}.

### 3.2 Evaluation

Tiny-ImageNet
```bash
[Split, 평문 모델의 추론 명령어]
python -u test_tinet_pure.py --batch-size=128 --cusin=1
python -u test_tinet_pure.py --batch-size=128 --cusin=2
python -u test_tinet_pure.py --batch-size=128 --cusin=3
python -u test_tinet_pure.py --batch-size=128 --cusin=4
```
CIFAR-10
```bash
python -u test_cifar10_pure.py --batch-size=128 --cusin=1
python -u test_cifar10_pure.py --batch-size=128 --cusin=2
python -u test_cifar10_pure.py --batch-size=128 --cusin=3
python -u test_cifar10_pure.py --batch-size=128 --cusin=4
```
Here, x specifies the split position, where x ∈ {1,2,3,4}.


### 3.3 FE-Compatible Inference

The FE-compatible inference pipeline includes integer-domain encoding, cryptographic encryption/decryption operations, and stage-wise execution of the proposed architecture.

```bash
[Split 함수암호 모델의 추론 명령어]
# Tiny-ImageNet dataset
python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=64 --cusin=1
python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=2
python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=3
python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=4

# CIFAR-10 dataset
python -u test_cifar10.py --terms=2 --unknown=16 --batch-size=1 --sife-l=64 --cusin=1
python -u test_cifar10.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=2
python -u test_cifar10.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=3
python -u test_cifar10.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=4
```
Here, x specifies the split position, where x ∈ {1,2,3,4}.


## 4. External Assets, Licenses, and Terms

This repository uses or references the following external assets. Raw datasets are not redistributed in this repository. Users are responsible for downloading datasets from their official sources and complying with the original licenses and terms of use.

| Asset | Use in this work | Source / Citation | License / Terms |
|---|---|---|---|
| CIFAR-10 | Image classification experiments | Official CIFAR-10 dataset; Krizhevsky, "Learning Multiple Layers of Features from Tiny Images" | This repository does not redistribute the raw CIFAR-10 dataset. Users should download it from the official source and comply with the applicable terms. |
| Tiny-ImageNet | Image classification experiments | Tiny-ImageNet / ImageNet-derived dataset | This repository does not redistribute raw Tiny-ImageNet images. Users should obtain the dataset from the appropriate source and comply with the applicable ImageNet/Tiny-ImageNet terms of use. |
| ResNet-50 architecture | Backbone architecture for CIFAR-10 and Tiny-ImageNet experiments | He et al., "Deep Residual Learning for Image Recognition" | Original architecture cited in the paper. |
| ResNet-50 implementation reference | Implementation reference for the ResNet-50 backbone | `bubbliiiing/faster-rcnn-pytorch` | MIT License. Our implementation was developed with reference to this MIT-licensed implementation and substantially modified for FE-compatible inference, including identity-branch replacement, dimensionality constraints, and integer-domain execution. |
| RLWE-IPFE | Functional encryption implementation/component used in the FE-compatible inference pipeline | `fentec-project/IPFE-RLWE` | MIT License. The original copyright and license notice are preserved where applicable. |


## 5. Responsible Release and Intended Use

This repository is released for academic research and reproducibility purposes.

The code is intended to support the experimental results reported in the paper on FE-compatible neural inference and architecture--cryptography co-design.

The implementation is **not** intended for:

- production deployment;
- security-critical systems;
- processing sensitive personal data without independent review;
- making unverified claims of cryptographic security.

The cryptographic routines and Python/C integration are provided to reproduce the experimental pipeline. They should not be treated as an audited production-grade cryptographic library.

The paper identifies a privacy-degenerate identity branch in FE-based inference and evaluates architecture-level alternatives that are more compatible with functional encryption. The proposed method should be interpreted within the scope and limitations stated in the paper.

During anonymous review, issues or concerns can be reported through the conference review process or the anonymized repository issue tracker.


## 6. Limitations

This repository supports the experiments reported in the paper, but the following limitations should be noted:

- The implementation is research code and has not undergone production-level security auditing.
- The security analysis focuses on structural reconstruction resistance under dimensionality constraints.
- Formal guarantees against adaptive or learning-based attacks are outside the current scope.
- Current FE schemes impose computational and numerical constraints, including integer-domain encoding and cryptographic overhead.
- The reported accuracy results are intended to characterize the utility impact of FE-compatible architectural changes, not to claim statistically significant accuracy superiority.
- The reported runtime results characterize the cost of the FE-compatible pipeline, which is largely determined by repeated cryptographic operations.


## 7. License

The code in this repository is released under the license provided in the `LICENSE` file.

If this repository includes or adapts MIT-licensed components, the original copyright and license notices are preserved where applicable.

## 9. Contact

During anonymous review, please use the anonymized repository issue tracker or the official conference review process.

