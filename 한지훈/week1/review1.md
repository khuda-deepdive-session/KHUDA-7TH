---
name: 📝 리뷰 과제
about: 2장 ML 기본지식
title: '[Week 1] 주차 리뷰 - 한지훈'
labels: ['review']
assignees: ''
---

## 주제
<!-- 이번 주차에 다룬 주요 주제를 작성해주세요 -->

- 경사하강법 방식들의 차이점, 쓰임새 등
- 해석가능한 인공지능
- 딥러닝은 지도/비지도 방법과 별개의 기법
- 누락된 데이터를 처리하는 방법
- 모델/데이터 면에서 과적합을 대처하는 방법

## 내용
<!-- 주요 개념과 내용을 정리해주세요 -->

### 핵심 개념 1: Adam optimizer은 왜 많이 쓰이는가?
- 설명:

기본적인 경사하강법의 경우 특정 포인트에서 x값이 증가함에 따라 함수값도 증가하면 x를 (-) 방향으로 이동해야 하고, x값이 증가함에 따라 함수값이 감소하면 x를 (+) 방향으로 이동하여 손실함수의 최솟값을 찾을 수 있다.

![image](https://github.com/user-attachments/assets/d01ae779-a0bd-4264-953c-95cce2c02d21)

![image (1)](https://github.com/user-attachments/assets/4ae183d0-977c-48c4-ae88-e41ebb780a8d)

즉 위 그림의 경우 빨간 구간에서는 x값이 증가함에 따라 함수값이 감소하기 때문에 x를 (+) 방향으로 이동하여 최솟값을 찾는다. 하지만 경사하강법에는 크게 두가지 문제(적절한 step size 찾기, Local minima)가 존재한다.

1. 적절한 step size(한번에 이동하는 크기) 찾기
    - Step size 너무 커지게 되면: 수렴 속도 빨라짐 but 최솟값 X/발산 가능
    - Step size너무 작아지면: 발산 X but 수렴 속도 오래 걸릴 수 있음
2. Local minima
    - 경사하강법: 시작하는 x 위치가 랜덤
    - 이 때문에 실제 최솟값이 아니라 지역 극솟값에 빠져 나오지 못할 수 있음

이런 경사하강법의 문제를 해결하기 위해 다양한 방법들이 제안되었다. 이 중 현재 가장 널리 쓰이는 방법은 Adaptive Moment Estimation (Adam)이다. Adam은 Momentum과 RMSProp을 합친 방식인데, 그렇기에 이 두가지 방법을 먼저 알아볼 필요가 있다.

**Momentum**

- 정의: 학습 속도를 높이고 Local minima 해결을 위해 관성 개념 추가

![image (2)](https://github.com/user-attachments/assets/eb11ae2b-aae1-4e07-91d3-990d6eebfd16)

![image (3)](https://github.com/user-attachments/assets/5daaa599-4947-41f0-a876-5561f2241ec2)

- 식: m*V(t-1)이 추가로 더해짐
- 이전 이동 거리(V(t-1))의 방향을 관성계수(m)만큼 추가로 고려
- 일반적으로 관성계수 m = 0.9

**RMSProp**

- Adagrad의 문제를 해결하기 위해 등장한 방법
    - Adagrad: “지속적으로 변화하던 parameter는 최적값에 가까워졌다고 가정, 한 번도 변하지 않은 parameter는 더 큰 변화를 주자”
        
        ![image (4)](https://github.com/user-attachments/assets/5b0e7eb9-5a85-4f50-a1d6-6195d0f8ed32)
        
    - 즉 파라미터 별 변화량을 나타내는 Gt(경사값의 제곱 합)가 분모로 가서 변화가 적은 파라미터에 더 큰 변화
    - 문제: 학습 진행에 따라 파라미터 변화폭 크게 감소, 정지
- Adagrad과 식은 동일, Gt 계산법 달라짐
- Gt 계산 시 경사값의 제곱에 대한 이동평균 사용 → 경사가 큰 곳에서는 학습률 줄이고, 경사가 작은 곳에서는 학습률 늘리는 역할 → 0 수렴 방지
- 감쇠율(γ) 클수록 조절량 증가, 보통 0.9

**Adam**

![image (5)](https://github.com/user-attachments/assets/98a047b0-2aab-473c-a2e6-57d4b7b01157)

- Momentum + RMSProp
    - Momentum: 1차 모멘트(m(t)) → 경사의 방향
    
    ![image (6)](https://github.com/user-attachments/assets/940fc6d3-e98d-4ae5-8d3c-1e77acf41f83)
    
    - RMSProp: 2차 모멘트(v(t)) → 경사의 크기
    
    ![image (7)](https://github.com/user-attachments/assets/1ad9043b-442d-4fca-b882-90403b675673)
    
- 편향 보정: m(t), v(t)가 초기단계에 0에 가까워져 학습 느려지는 문제 해결

![image (8)](https://github.com/user-attachments/assets/eb192610-3c6a-4a45-a9ef-94793be1153b)

장점

- 경사의 방향과 크기를 모두 고려하기 때문에 효율적, 빠른 수렴
- 2차 모멘트 → 각 파라미터에 맞게 학습률 조정 → 수동 조정 덜해도 됨
- 복잡한 모델에 적합

하지만 수렴이 빨라 과적합 우려는 존재, early stopping/정규화 등 추천

- 예시: 앞서 언급한 것처럼 경사하강법의 문제를 해결하기 위해서 대표적으로 Momentum, RMSProp, Adam 등의 방법이 제안되었다.

### 핵심 개념 2: ML VS DL + 해석가능한 인공지능
- 설명:

ML, DL은 많은 차이점이 있지만, 그 중 ‘해석가능성’에 집중해보고자 한다.

딥러닝

- 답 도출 시 어떤 근거로 내었는지 알 수 없어 사람은 결과 해석 못함
- 블랙박스 모델

머신러닝

- Engineering한 범위내에서 결과에 대한 이유, 원인 등 알 수 있음

이런 이유로 전공자가 아닌 사람을 설득하는 등의 경우에는 Decision Tree, Linear Regression 등의 해석가능한 모델이 선호된다.

이렇듯 인공지능에 대한 이해, 신뢰 등의 필요성이 강조되어 최근에는 인공지능의 의사결정 과정을 인간이 이해할 수 있도록 하는 기술인 XAI(Explainable AI, 설명 가능한 AI)가 등장하고 있다. XAI는 크게 3가지 방법으로 구현된다.

1. 모델 내 해석 가능성 (Interpretable Models)
    - 앞서 언급한 Decision Tree 등의 경우 그 자체로 해석 가능
2. 사후 해석 방법 (Post-hoc Explainability)
    - LIME (Local Interpretable Model-agnostic Explanations)
        - 개별 예측 결과를 이해하기 위해 국소적인 선형 모델을 생성
    - **SHAP (SHapley Additive exPlanations)**
        - 각 피처의 기여도를 계산하여 설명
    - **Gradient-based Methods**
        - 모델의 입력값에 대한 기울기를 계산하여 영향을 시각화
3. 시각화 기법
    - 히트맵, 중요도 그래프, 결정 경계 시각화 등으로 표시

- 예시: XAI는 실제로 의료 분야에서 진단 이유, 혹은 금융 분야에서 신용점수 예측 이유 등에 대한 근거를 제공하여 사용자가 그 결정을 더 신뢰할 수 있도록 한다.

## 참고 문헌
<!-- 참고한 자료의 제목과 링크를 작성해주세요 -->
1. [머신 러닝] - 경사 하강법(Gradient descent) https://twojun-space.tistory.com/124
2. Optimizer 의 종류와 특성 (Momentum, RMSProp, Adam) https://onevision.tistory.com/entry/Optimizer-%EC%9D%98-%EC%A2%85%EB%A5%98%EC%99%80-%ED%8A%B9%EC%84%B1-Momentum-RMSProp-Adam
3. SGD / AdaGrad / RMSProp / Momentum / Adam https://velog.io/@min0731/SGD-Momentum-RMSProp-Adam
4. Explainable AI(설명 가능한 인공지능). XAI의 개념 https://aidd.medium.com/explainable-ai-%EC%84%A4%EB%AA%85-%EA%B0%80%EB%8A%A5%ED%95%9C-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-4b40bfb0a70
