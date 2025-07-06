# 프로젝트 개요

본 Repository는 Microsoft의 Phi-2 모델과 OpenAI의 ViT-Base-Patch32 모델을 결합하여 멀티모달을 구현하기 위해 개발되었습니다.

## 사용된 데이터셋  
- MS COCO Captions 2017  
- LLaVA Instruction Tuning Dataset (150k 샘플)

## 주요 파일 설명  
- `model_pipeline.ipynb`  
  - Teacher Forcing 기법을 활용한 모델 학습 파이프라인 구현  
- `instruction_train.ipynb`  
  - Instruction Tuning을 수행하기 위한 학습 스크립트

## 모델 아키텍처  
- Phi-2 모델 32개 layers중 후반부 4개 layers에 OpenAI ViT-Base-Patch32 모델을 활용한 Cross-Attention hook을 삽입
- 각 layer의 self-attention에 LoRA를 삽입하여 학습간 최적화와 Phi2의 성능을 유지시킴
