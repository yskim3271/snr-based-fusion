

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stft import ConvSTFT, ConviSTFT

# 소스 [1]에 제시된 Multi-Scale Channel Attention Module (MS-CAM) 구현
# 기반 정보: 소스 [5, 6], 그림 1 [7]
class MSCAM(nn.Module):
    # channels: 입력 채널 수 C [6, 8]
    # reduction: 채널 감소 비율 r [4, 6, 8]
    def __init__(self, channels, reduction=16):
        super(MSCAM, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # 로컬 컨텍스트 집계 경로 (L(X)) [5, 6]
        # 병목 구조: PWConv -> BN -> ReLU -> PWConv -> BN [5-7]
        self.local_context = nn.Sequential(
            # 첫 번째 포인트와이즈 컨볼루션 (채널 감소) [5, 6]
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            # 배치 정규화 [5, 6, 8]
            nn.BatchNorm2d(channels // reduction),
            # ReLU 활성화 [5, 6, 8]
            nn.ReLU(inplace=True),
            # 두 번째 포인트와이즈 컨볼루션 (채널 복원) [6]
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            # 배치 정규화 [5, 6]
            nn.BatchNorm2d(channels)
        )

        # 글로벌 컨텍스트 집계 경로 (g(X)를 병목 구조 통과) [5, 6, 8]
        self.global_context = nn.Sequential(
            # 글로벌 평균 풀링 [5, 8] - 공간 크기를 1x1로 줄임
            nn.AdaptiveAvgPool2d(1),
            # 첫 번째 포인트와이즈 컨볼루션 (채널 감소) [6, 8]
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            # 배치 정규화 [8]
            nn.BatchNorm2d(channels // reduction),
            # ReLU 활성화 [8]
            nn.ReLU(inplace=True),
            # 두 번째 포인트와이즈 컨볼루션 (채널 복원) [4, 6]
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            # 배치 정규화 [8]
            nn.BatchNorm2d(channels)
        )

        # 시그모이드 활성화 함수 - 어텐션 가중치 생성 [6, 8]
        self.sigmoid = nn.Sigmoid()

    # x: 입력 피쳐 맵 (RC×H×W) [6, 8]
    def forward(self, x):
        # 1. 로컬 컨텍스트 계산 [6]
        # local_context는 입력 x와 동일한 RC×H×W 형태
        local_features = self.local_context(x)

        # 2. 글로벌 컨텍스트 계산 [6]
        # global_features는 병목 구조를 거치면서 RC×1×1 형태가 됨
        global_features = self.global_context(x)

        # 3. 로컬 및 글로벌 컨텍스트 융합 (합산) [6, 7]
        # PyTorch 브로드캐스팅 기능으로 global_features (RC×1×1)가
        # local_features (RC×H×W)와 합산될 때 자동으로 확장됨
        combined_features = local_features + global_features # L(X) ⊕ g(X) [6]

        # 4. 어텐션 가중치 생성 [6]
        attention_weights = self.sigmoid(combined_features) # M(X) = σ(...) [6]

        # 5. 입력 피쳐에 어텐션 가중치 적용 [6]
        # 요소별 곱셈 [6]
        refined_feature = x * attention_weights # X' = X ⊗ M(X) [6]

        return refined_feature


class AFF(nn.Module):
    # channels: 입력 채널 수 C (X와 Y의 채널 수는 같아야 함)
    # reduction: MS-CAM 내부 채널 감소 비율 r
    def __init__(self, channels, reduction=16):
        super(AFF, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # MS-CAM 모듈 초기화
        # MS-CAM은 초기 통합 결과 (X+Y)에 대한 어텐션 가중치를 생성합니다.
        self.mscam = MSCAM(channels, reduction=reduction)

    # x, y: 융합할 두 개의 입력 피쳐 맵 (RC×H×W)
    def forward(self, x, y):
        # 1. 초기 통합 (Initial Integration) - 여기서는 요소별 합산 사용 [6]
        # 다른 통합 방식도 가능하지만, 논문 기본 AFF는 합산입니다.
        initial_integrated_features = x + y # X + Y

        # 2. MS-CAM을 사용하여 어텐션 가중치 생성 [19, Figure 2(a)]
        # MS-CAM은 초기 통합 결과에 대한 어텐션 맵 M(X+Y)를 반환합니다.
        attention_weights = self.mscam(initial_integrated_features) # M(X+Y)

        # 3. 어텐션 가중치를 사용하여 X와 Y를 선택적으로 융합 [19, 수식 (4)]
        # Z = M ⊗ X + (1 − M) ⊗ Y
        # attention_weights는 M, (1 - attention_weights)는 (1-M)
        fused_feature = attention_weights * x + (1 - attention_weights) * y # M ⊗ X + (1-M) ⊗ Y

        return fused_feature



class fusion(nn.Module):
    def __init__(self, win_len, win_inc, fft_len, win_type, model_name, param):
        super(fusion, self).__init__()
        self.model_name = model_name
        self.model_param = param

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        self.stft = ConvSTFT(
            win_len=self.win_len,
            win_inc=self.win_inc,
            fft_len=self.fft_len,
            win_type=self.win_type,
            feature_type='complex'
        )
        self.istft = ConviSTFT(
            win_len=self.win_len,
            win_inc=self.win_inc,
            fft_len=self.fft_len,
            win_type=self.win_type,
            feature_type='complex'
        )

        module = importlib.import_module("models." + self.model_name)

        model_class = getattr(module, self.model_name)

        self.attention_fusion = AFF(channels=2, reduction=1)

        self.model_x1 = model_class(**self.model_param)
        self.model_x2 = model_class(**self.model_param)

    def forward(self, x1, x2):

        in_len = x1.shape[-1]
        x1_ = self.model_x1(x1)
        x2_ = self.model_x2(x2)

        x1_spec = self.stft(x1_)
        x2_spec = self.stft(x2_)

        x1_real, x1_imag = torch.chunk(x1_spec, 2, dim=1)
        x2_real, x2_imag = torch.chunk(x2_spec, 2, dim=1)

        x1_spec = torch.stack([x1_real, x1_imag], dim=1)
        x2_spec = torch.stack([x2_real, x2_imag], dim=1)

        xf_spec = self.attention_fusion(x1_spec, x2_spec)

        xf_real = xf_spec[:, 0, :, :]
        xf_imag = xf_spec[:, 1, :, :]

        xf_spec = torch.cat([xf_real, xf_imag], dim=1)
        out_wav = self.istft(xf_spec)

        out_len = out_wav.shape[-1]

        if out_len > in_len:
            leftover = out_len - in_len 
            out_wav = out_wav[..., leftover//2:-(leftover//2)]

        return out_wav






