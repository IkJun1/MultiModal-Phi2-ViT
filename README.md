# 프로젝트 개요

Phi-2 기반 언어 모델과 CLIP ViT 인코더를 교차 어텐션으로 결합한 멀티모달 캡셔닝 및 질의응답 실험용 레포지토리입니다. 학습 단계는 일반 캡션 데이터 전처리 → LoRA 적응형 교차 어텐션 학습 → LLaVA 지시 튜닝 → 추론 파이프라인으로 구성됩니다.

## 데이터셋

- **MS COCO Captions 2017**: 기본 캡션 학습용. [`preprocessing.py`](preprocessing.py)에서 [`module.preprocessor.save_preprocessed_dataset1`](module/preprocessor.py) 호출로 전처리되어 [dataset/preprocessed_mscoco](dataset/preprocessed_mscoco)에 저장됩니다.
- **LLaVA Instruction 150k**: 지시 튜닝용. 동일 스크립트에서 [`module.preprocessor.save_preprocessed_dataset2`](module/preprocessor.py)를 통해 [dataset/preprocessed_llava_instruct](dataset/preprocessed_llava_instruct)에 저장됩니다.

## 코드 구조

| 파일 | 설명 |
| ---- | ---- |
| [`module/model.py`](module/model.py) | 모델/LoRA 정의. [`module.model.MultimodalPhi2`](module/model.py)는 Phi-2 후반 4개 레이어에 이미지 피처를 주입합니다. |
| [`module/preprocessor.py`](module/preprocessor.py) | COCO 캡션 평탄화 및 LLaVA 질의응답 마스킹 전처리 유틸. |
| [`preprocessing.py`](preprocessing.py) | 두 데이터셋을 불러와 저장하는 엔트리 스크립트. |
| [`train.py`](train.py) | COCO 캡션 학습 후 LLaVA 지시 튜닝까지 수행하고 어댑터·프로젝션을 저장합니다. |
| [`inference.py`](inference.py) | 저장된 어댑터를 불러와 선택형 질의에 대한 답변을 생성합니다. |

## 모델 아키텍처

- 텍스트 백본: Microsoft Phi-2 (4-bit 로드, $\\mathrm{bfloat16}$ 계산)
- 비전 백본: CLIP ViT-B/32
- 교차 어텐션: Phi-2 마지막 4개 레이어의 self-attention 출력에 이미지 토큰을 주입
- 적응형 튜닝: [`module.model.create_LoRA_model`](module/model.py)가 $r=32$, $\\alpha=64$, 드롭아웃 0.05의 LoRA 어댑터를 `q_proj`, `k_proj`, `v_proj`, `dense` 모듈에 장착

## 학습 파이프라인

1. **전처리 실행**  
   ```bash
   python preprocessing.py
   ```
2. **캡션 학습**  
   [`train.py`](train.py)는 COCO 전처리 결과를 사용해 $4$ epoch 학습 후 LLaVA 데이터로 $2$ epoch 지시 튜닝을 이어서 수행합니다.  
   학습 루프는 AdamW + 선형 스케줄러($\\text{warmup}=0$ 또는 $0.05$ 비율)를 사용합니다.
3. **가중치 저장**  
   학습 완료 시 LoRA 어댑터(`llm_adapters*`), 비전 프로젝션(`vision_projection*.pt`), 교차 어텐션(`cross_attentions*.pt`)이 [saved_models](saved_models) 폴더에 저장됩니다.

## 추론

- [`inference.py`](inference.py)의 [`load_trained_model`](inference.py)로 저장된 어댑터를 로드합니다.
- [`inference.py`](inference.py) 내부 [`inference`](inference.py)는 프롬프트와 이미지 경로를 입력받아 Phi-2로부터 선택형 답변을 생성합니다.

## 실행 환경

필수 패키지는 [requirements.txt](requirements.txt)를 참고하세요. 주요 의존성: PyTorch $2.6.0+\\mathrm{cu124}$, Transformers $4.53.0$, PEFT, bitsandbytes, datasets 등.