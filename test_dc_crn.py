import torch
import matplotlib.pyplot as plt
import numpy as np
from models.dc_crn import dc_crn

def test_dc_crn():
    # 모델 파라미터 설정
    input_dim = 1  # 단일 채널 입력
    feature_dim = 256  # FFT_LENGTH//2 + 1
    window_size = 512
    hop_size = 256
    fft_length = 512
    
    # 모델 초기화
    model = dc_crn(
        input_dim=input_dim,
        window_size=window_size,
        hop_size=hop_size,
        fft_length=fft_length,
        channels=[16, 32, 64, 128, 256],
        dcblock_depth=4,
        dcblock_growth_rate=8
    )
    
    # 테스트 입력 생성 (임의의 오디오 신호)
    # 배치 크기 2, 각 신호는 1초(16kHz 샘플레이트 가정)
    batch_size = 2
    audio_length = 16000
    x = torch.randn(batch_size, audio_length)
    
    print(f"입력 텐서 크기: {x.shape}")
    
    # 모델에 입력 전달 (단일 입력 케이스)
    print("단일 입력 테스트 중...")
    output = model(x)
    
    print(f"출력 텐서 크기: {output.shape}")
    
    # 출력 형태 확인
    print(f"출력 채널 수: {output.shape[1]}")  # 실수부/허수부
    
    model = dc_crn(
        input_dim=2,
        window_size=window_size,
        hop_size=hop_size,
        fft_length=fft_length,
        channels=[16, 32, 64, 128, 256],
        dcblock_depth=4,
        dcblock_growth_rate=8
    )

    # 두 개의 입력 케이스 테스트 (x1, x2)
    print("\n두 개의 입력 테스트 중...")
    x2 = torch.randn(batch_size, audio_length)
    output_dual = model(x, x2)
    
    print(f"이중 입력 출력 텐서 크기: {output_dual.shape}")
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 모델 파라미터 수: {total_params:,}")
    
    

if __name__ == "__main__":
    test_dc_crn() 