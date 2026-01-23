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
> **실험 가설**: "에이전트를 2개로 늘리고, 서로 같은 곳을 보지 않게 하면(Repulsion), 한 이미지 안의 여러 물체를 동시에 찾을 수 있지 않을까?"

- **목표**: 단일 이미지 내의 **여러 객체(Multi-Object)**를 동시에 탐지.
- **주요 실험 및 구현 로직**:

    #### A. Repulsion Reward (밀어내기 보상)
    - **개요**: 에이전트들이 서로 같은 객체로 몰려서 중복 탐지하는 것을 방지.
    - **구현**: `IoU(Agent1, Agent2) > 0.8` 로 두 에이전트의 시야가 과도하게 겹치면 **보상 -1 (패널티)**를 부여하여 서로 다른 영역을 탐색하도록 유도.

    #### B. 타겟 설정 전략 실험 (Target Assignment Strategy)
    멀티 에이전트 환경에서 "누가 어떤 객체를 쫓아야 하는가?"를 정의하기 위해 두 가지 방식을 고안하고 실험했습니다.
    
    1.  **거리 기반 타겟 설정 (Distance-based Targeting)**
        -   **문제 의식**: 초기 학습 시, 에이전트가 객체와 전혀 겹치지 않는 빈 공간(IoU=0)을 볼 때 **보상이 없어(Sparse Reward)** 학습이 더뎌짐.
        -   **해결 아이디어**: 겹치지 않더라도 물리적으로 **"가장 가까운 객체"**를 타겟으로 지정하여, 빈 공간에서도 방향성을 가질 수 있게 하자.
        -   **구현 (`closet_distance_gt`)**: 현재 뷰의 중심점과 GT 객체들의 중심점 간 **유클리드 거리**를 계산하여 타겟 할당.

    2.  **Max IoU 기반 타겟 설정 (Max IoU Targeting)**
        -   **문제 의식**: 에이전트가 이동함에 따라 타겟이 모호해지거나, 여러 에이전트가 우연히 같은 객체를 목표로 삼을 수 있음.
        -   **해결 아이디어**: **"지금 내가 제일 잘 보고 있는 놈이 내 타겟이다!"**라는 직관적 기준 적용.
        -   **구현 (`find_max_bounding_box_and_gt`)**: 현재 뷰와 가장 **IoU가 높은 객체**를 동적으로 타겟으로 삼음. 동시에 객체의 고유 ID(좌표합)를 추출하여, 다른 에이전트와 **같은 타겟을 쫓고 있는지 식별**하는 용도로 활용 (Repulsion Reward 계산의 핵심).

    #### C. 학습 안정화 (DQN 2015)
    - [2015 Nature DQN](https://www.nature.com/articles/nature14236)의 **Target Network Update** 방식을 적용. (`5 step` 주기 업데이트)

<div align="center">
  <img width="400" height="300" alt="Result Image 1" src="https://github.com/user-attachments/assets/9931ee21-7a1c-4776-9872-8ae361e39105" />
  <br>
  <img width="600" height="130" alt="Result Image 2" src="https://github.com/user-attachments/assets/846bf9d5-ffd5-49f0-b937-e7057ea587fa" />
</div>

### 3. 3단계: 협력적 멀티 에이전트 실험 (`multi_main.py`)
> **실험 가설**: "각자 따로 노는 게 아니라, 내가 확신이 없을 때 친구의 판단을 참고하면(Collaboration) 더 정확해질 수 있을까?"

- **목표**: 에이전트 간의 **협력(Collaboration)**을 통해 탐지 정확도 및 효율성 증대.
- **참고 논문**: [Collaborative Deep Reinforcement Learning for Joint Object Search](https://arxiv.org/abs/1702.05573)
- **주요 혁신 및 변경점**:
    - **Sigmoid Gating Mechanism (동적 가중치 결합)**: 자신의 확신도에 따라 동료 에이전트의 정보를 얼마나 반영할지 결정하는 협력 알고리즘 설계.
        > **개념**: `최종 Q값 = (내 판단 × 내 확신도) + (동료의 제안 × (1 - 내 확신도))`
        - 확신도(Sigmoid 값)가 높으면 자신의 판단을 따르고, 낮으면 가상(Virtual) 에이전트의 제안을 더 신뢰하도록 설계.
    - **교차 학습 전략 (Cross-Training)**: '실제(Actual)' 모델과 '가상(Virtual)' 모델이 서로 가중치를 교환하며 학습하여 상호 보완적인 탐색 능력을 배양.

<div align="center">
  <img width="350" height="250" alt="Concept Image" src="https://github.com/user-attachments/assets/03bf5767-7fd8-487c-a25e-45d705635625" />
  <p>(이미지 출처: Collaborative Deep Reinforcement Learning for Joint Object Search)</p>
  <br>
  <h4>[시각화 결과]</h4>
  <img width="1016" height="382" alt="Visualization 1" src="https://github.com/user-attachments/assets/f41df3c1-8f8a-4019-adb9-2f4168b1f3fa" />
  <br>
  <img width="600" height="90" alt="Visualization 2" src="https://github.com/user-attachments/assets/c1072d6a-de8e-439b-9da9-e92bc2b82501" />
</div>

## 📝 요약 및 결론 (Summary & Conclusion)

본 프로젝트는 **`origin_main.py` (단일 객체 탐지)** 모델을 기반으로 하여 다음과 같은 연구 흐름을 통해 발전했습니다.

1.  **확장 (Scaling)**: `main.py`에서 에이전트 수를 늘리고, 서로 다른 객체를 찾도록 **"서로 밀어내는 보상(Repulsion)"**과 **"동적 타겟 할당 전략(Max IoU/Distance)"**을 추가하여 다중 객체 탐지를 시도했습니다.
2.  **협력 (Collaboration)**: `multi_main.py`에서는 단순한 독립 탐색을 넘어, **"Sigmoid 게이팅을 이용한 정보 공유"**라는 독창적인 아키텍처를 도입하여 에이전트 간의 협력 가능성을 실험했습니다.

이러한 실험을 통해, 강화학습 에이전트들이 단순히 개별적으로 동작하는 것을 넘어 서로 상호작용하고 정보를 교환할 때 더 복잡한 환경(Multi-Object)에서도 효과적으로 동작할 수 있음을 확인하고자 했습니다.

## 📂 주요 파일 설명

| 파일명 | 설명 |
| :--- | :--- |
| **`origin_main.py`** | 원본 논문을 구현한 **베이스라인(Baseline)** 코드 (Single Agent). |
| **`main.py`** | **독립형 멀티 에이전트**. Repulsion Reward와 DQN 2015(Target Update) 방식이 적용됨. |
| **`multi_main.py`** | **협력형 멀티 에이전트**. Sigmoid Gating을 이용한 협력 학습 구조가 적용됨. |
| **`test copy.ipynb`** | 학습된 모델을 불러와 성능을 테스트하고 시각화하는 **테스트 실행 파일**. |
| `mix.py` | Dynamic 방식의 액션 (9개의 동적 박스 선정 방식 액션) + Hierarichal 방식의 액션(6가지 줌인 방식 액션). (구현 중) |
| `metrics.py` | IoU 계산 및 액션(Crop) 정의 유틸리티 파일. |

## 🛠 액션 공간 (Action Space)
에이전트는 다음과 같은 6가지 계층적 액션을 통해 이미지를 탐색합니다.
1. **Top-Left Crop**: 왼쪽 위 영역으로 줌인.
2. **Top-Right Crop**: 오른쪽 위 영역으로 줌인.
3. **Bottom-Left Crop**: 왼쪽 아래 영역으로 줌인.
4. **Bottom-Right Crop**: 오른쪽 아래 영역으로 줌인.
5. **Center Crop**: 중앙 영역으로 줌인.
6. **Trigger**: 탐색 종료 및 현재 영역을 최종 탐지 결과로 제안.
