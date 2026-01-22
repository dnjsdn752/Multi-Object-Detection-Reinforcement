# Multi-Object Detection with Deep Reinforcement Learning

해당 **심층 강화학습(Deep Reinforcement Learning)을 이용한 다중 객체 탐지(Multi-Object Detection)** 실험은 기존 단일 에이전트 방식을 확장하여, 이미지 내의 여러 객체를 동시에 탐지할 수 있는 멀티 에이전트 시스템(Multi-Agent System)을 구현하고 실험한 연구입니다.

[Hierarchical Object Detection with Deep Reinforcement Learning](https://arxiv.org/abs/1611.03718) 논문의 PyTorch 구현체를 기반으로 발전시켰습니다.

### 💡 논문 요약
논문의 에이전트는 이미지를 한 번에 모두 분석하는 것이 아니라, **강화학습**을 통해 **어디를 더 자세히 볼지**를 순차적으로 결정합니다.

1.  **계층적 탐색 (Hierarchical Search)**:
    - 에이전트는 전체 이미지에서 시작하여 유망한 영역(Region of Interest)을 선택해 **Zoom-in** 합니다.
    - 선택된 영역에 대해 다시 고해상도 특징(Feature)을 추출하므로, 작은 객체 탐지에 유리합니다.

2.  **MDP (Markov Decision Process) 모델링**:
    - **State (상태)**: 현재 관찰 중인 영역의 **시각적 특징(VGG-16 Feature)** + **과거의 행동 기록(Memory Vector)**.
    - **Action (행동)**: 5개의 하위 영역(좌상, 우상, 좌하, 우하, 중앙)으로 이동하거나, 탐색을 멈추고 객체를 감지(Trigger)하는 총 **6가지 이산 행동(Discrete Actions)**.
    - **Reward (보상)**: 행동 수행 후 Ground Truth와의 **IoU(Intersection over Union)가 상승**하면 양의 보상(+), 하락하면 음의 보상(-)을 부여.

이 방식은 수천 개의 후보 영역을 계산하는 기존 방식과 달리, 에이전트가 **필요한 영역만 효율적으로 탐색**한다는 장점이 있습니다. 본 프로젝트는 이 단일 객체 탐지 모델을 **다중 객체 탐지(Multi-Agent System)**로 확장한 실험입니다.

## 🚀 프로젝트 개요 및 발전 과정 (Project Evolution)

단일 객체 탐지 모델에서 시작하여, 다중 객체를 효율적으로 찾기 위한 독립적 에이전트 모델, 그리고 상호 협력하는 모델로 단계적으로 발전시킨 실험 과정을 담고 있습니다.

### 1. 1단계: 기본 단일 에이전트 (`origin_main.py`)
- **출처**: 원본 논문의 로직을 충실히 구현한 베이스라인.
- **목표**: 이미지 내에서 1개의 주요 객체 탐지.
- **방식**: 기본적인 계층적(Hierarchical) DQN을 사용하여 줌인(Zoom-in) 방식으로 탐색.

### 2. 2단계: 독립적 멀티 에이전트 실험 (`main.py`)
- **목표**: 단일 이미지 내의 **여러 객체 Multi-Object**를 동시에 탐지.
- **주요 혁신 및 변경점**:
    - **2-Agent 시스템 도입**: 두 개의 에이전트가 동시에 독립적으로 탐색을 수행.
    - **Repulsion Reward (밀어내기 보상)**: 에이전트들이 서로 같은 객체로 몰리는 것을 방지하기 위해 벌점 시스템 도입.
        - `IoU(Agent1, Agent2) > 0.8` (서로 너무 겹침) ➜ **보상 -1 (패널티)**
        - 서로 다른 영역 탐색 시 ➜ **보상 +1**
    - **학습 안정화 (DQN 2015 적용)**: [2015 Nature DQN](https://www.nature.com/articles/nature14236) 논문에서 제안한 **Target Network Update** 방식을 적용.
        - 매 스텝마다 업데이트하던 기존 방식 대신, `5 step`마다 타겟 네트워크를 업데이트하여 학습의 진동을 줄이고 안정성을 확보함.
          <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/9931ee21-7a1c-4776-9872-8ae361e39105" />

          <img width="600" height="130" alt="image" src="https://github.com/user-attachments/assets/846bf9d5-ffd5-49f0-b937-e7057ea587fa" />



### 3. 3단계: 협력적 멀티 에이전트 실험 (`multi_main.py`)
- **목표**: 에이전트 간의 **협력(Collaboration)**을 통해 탐지 정확도 및 효율성 증대.
- **참고 논문**: [Collaborative Deep Reinforcement Learning for Joint Object Search](https://arxiv.org/abs/1702.05573)
- **주요 혁신 및 변경점**:
    - **Sigmoid Gating Mechanism (동적 가중치 결합)**: 자신의 확신도에 따라 동료 에이전트의 정보를 얼마나 반영할지 결정하는 협력 알고리즘 설계.
        > **개념**: `최종 Q값 = (내 판단 × 내 확신도) + (동료의 제안 × (1 - 내 확신도))`
        - 확신도(Sigmoid 값)가 높으면 자신의 판단을 따르고, 낮으면 가상(Virtual) 에이전트의 제안을 더 신뢰하도록 설계.
    - **교차 학습 전략 (Cross-Training)**: '실제(Actual)' 모델과 '가상(Virtual)' 모델이 서로 가중치를 교환하며 학습하여 상호 보완적인 탐색 능력을 배양.
      <img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/03bf5767-7fd8-487c-a25e-45d705635625" />
      (이미지 출처: [Collaborative Deep Reinforcement Learning for Joint Object Search](https://arxiv.org/abs/1702.05573))
 - **시각화** :
   <img width="1016" height="382" alt="image" src="https://github.com/user-attachments/assets/f41df3c1-8f8a-4019-adb9-2f4168b1f3fa" />
   <img width="600" height="90" alt="image" src="https://github.com/user-attachments/assets/c1072d6a-de8e-439b-9da9-e92bc2b82501" />




## 📂 주요 파일 설명

| 파일명 | 설명 |
| :--- | :--- |
| **`origin_main.py`** | 원본 논문을 구현한 **베이스라인(Baseline)** 코드 (Single Agent). |
| **`main.py`** | **독립형 멀티 에이전트**. Repulsion Reward와 DQN 2015(Target Update) 방식이 적용됨. |
| **`multi_main.py`** | **협력형 멀티 에이전트**. Sigmoid Gating을 이용한 협력 학습 구조가 적용됨. |
| **`test copy.ipynb`** | 학습된 모델을 불러와 성능을 테스트하고 시각화하는 **테스트 실행 파일**. |
| `mix.py` | (Legacy) 초기 실험 버전 (9개의 액션 공간 및 전문가 가이드 포함). |
| `metrics.py` | IoU 계산 및 액션(Crop) 정의 유틸리티 파일. |

## 🛠 액션 공간 (Action Space)
에이전트는 다음과 같은 6가지 계층적 액션을 통해 이미지를 탐색합니다.
1. **Top-Left Crop**: 왼쪽 위 영역으로 줌인.
2. **Top-Right Crop**: 오른쪽 위 영역으로 줌인.
3. **Bottom-Left Crop**: 왼쪽 아래 영역으로 줌인.
4. **Bottom-Right Crop**: 오른쪽 아래 영역으로 줌인.
5. **Center Crop**: 중앙 영역으로 줌인.
6. **Trigger**: 탐색 종료 및 현재 영역을 최종 탐지 결과로 제안.


