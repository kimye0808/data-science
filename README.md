# Data Science Programming Assignments

이 레포지토리는 데이터 과학 과제들을 제출하는 과정에서 작성한 코드 및 결과물을 포함합니다. 각 과제에 대한 목표와 요구 사항을 아래에 정리해두었습니다.

---

## 과제 1: Apriori 알고리즘을 사용한 연관 규칙 찾기

### 목표
- Apriori 알고리즘을 사용하여 자주 등장하는 아이템 집합을 찾고, 각 집합에 대해 연관 규칙을 도출합니다.

<details>
<summary>세부 사항</summary>
### 요구 사항
- 입력 파일과 출력 파일을 사용한 연산
- 입력 형식: 각 거래는 탭 구분으로 아이템들이 나열됨
- 출력 형식: 아이템 집합과 연관 아이템 집합, 지원도 및 신뢰도를 포함한 규칙 출력

### 실행 방법
```bash
python studentID_name_hw1.py <min_support> <input_file> <output_file>
```
- `min_support`: 최소 지원도 (퍼센트)
- `input_file`: 거래 데이터 파일
- `output_file`: 결과 출력 파일

### 예시
```bash
python 2025016242_albertkarlo_hw1.py 10 input.txt output.txt
```

### 코드 설명
이 코드는 Apriori 알고리즘을 구현하여 거래 데이터에서 빈발 아이템 집합을 찾고 연관 규칙을 생성합니다. 주요 구성 요소는 다음과 같습니다:

- **`getTransactionsList(fileName)`**: 탭으로 구분된 입력 파일을 읽어 거래 리스트를 생성합니다. 예: `1\t2\t3` → `[1, 2, 3]`.
- **`calcSupport(transactions, itemSet)`**: 주어진 아이템 집합의 지원도를 계산합니다. 거래 중 해당 집합이 포함된 비율(%)을 반환합니다.
- **`getFrequentItemSets(transactions, minSup)`**: Apriori 알고리즘을 사용해 최소 지원도(`minSup`) 이상의 빈발 아이템 집합을 찾습니다. 단일 아이템에서 시작해 크기를 늘리며 반복적으로 후보를 생성하고 필터링합니다.
- **`writeAssociationRules(transactions, frequentItemsets, outputFile)`**: 빈발 아이템 집합에서 연관 규칙을 도출하고, `{itemSet}\t{associativeItemSet}\t{support}\t{confidence}` 형식으로 파일에 기록합니다. 신뢰도는 `freqSet` 지원도를 `itemSet` 지원도로 나눈 값입니다.
- **`main()`**: 명령줄 인자를 받아 전체 과정을 실행합니다.

#### 동작 예시
- **입력 (`input.txt`)**:
  ```
  1	2	3
  1	2
  2	3
  ```
- **명령어**: `python 2025016242_albertkarlo_hw1.py 10 input.txt output.txt`
- **출력 (`output.txt`)**:
  ```
  {1}	{2}	66.67	100.00
  {2}	{1}	66.67	100.00
  {2}	{3}	66.67	100.00
  {3}	{2}	66.67	100.00
  ```

</details>

---

## 과제 2: 결정 트리를 사용한 분류

### 목표
- 주어진 데이터셋을 바탕으로 결정 트리를 구축하고, 이를 사용해 테스트 데이터셋을 분류합니다.

<details>
<summary>세부 사항</summary>

### 요구 사항
- 훈련 데이터셋과 테스트 데이터셋을 사용하여 분류
- 훈련 데이터셋 파일과 테스트 데이터셋 파일의 형식에 맞춰 코드 구현

### 실행 방법
```bash
python studentID_name_hw2.py <train_file> <test_file> <output_file>
```
- `train_file`: 훈련 데이터셋 파일
- `test_file`: 테스트 데이터셋 파일
- `output_file`: 분류 결과 출력 파일

### 예시
```bash
python 2025016242_albertkarlo_hw2.py dt_train.txt dt_test.txt dt_result.txt
```

### 코드 설명
이 코드는 정보 이득률(gain ratio)을 기반으로 결정 트리를 구축하고 테스트 데이터를 분류합니다. 주요 구성 요소는 다음과 같습니다:

- **`calcEntropy(classLabels)`**: 데이터의 엔트로피를 계산합니다. 클래스 분포를 기반으로 불확실성을 측정합니다.
- **`calcInformationGain(data, feature, classLabel)`**: 특정 피처의 정보 이득을 계산합니다. 엔트로피 감소량을 측정합니다.
- **`calcGainRatio(data, feature, classLabel)`**: 정보 이득률을 계산하여 분할 기준 피처를 선택합니다.
- **`buildDecisionTree(data, features, classLabel)`**: 재귀적으로 결정 트리를 생성합니다. 최적 피처로 데이터를 분할하며, 단일 클래스가 남거나 더 분할할 피처가 없으면 종료합니다.
- **`getTestResult(tree, rowData, major)`**: 테스트 데이터를 트리에 따라 분류합니다. 트리에 없는 값은 다수 클래스로 처리합니다.
- **`main()`**: 훈련/테스트 데이터를 읽고, 트리를 구축한 뒤 결과를 파일에 저장합니다.

</details>

---

## 과제 3: DBSCAN을 사용한 클러스터링

### 목표
- DBSCAN 알고리즘을 사용하여 주어진 데이터셋을 클러스터링합니다.

<details>
<summary>세부 사항</summary>

### 요구 사항
- 입력 파일과 클러스터 수, Eps, MinPts 값을 기반으로 클러스터링 수행
- 클러스터링을 위한 적절한 파라미터 값 선택

### 실행 방법
```bash
python studentID_name_hw3.py <input_file> <n> <Eps> <MinPts>
```
- `input_file`: 데이터셋 파일
- `n`: 클러스터 수
- `Eps`: 최대 반경
- `MinPts`: Eps-반경 내 최소 점수

### 예시
```bash
python 2025016242_albertkarlo_hw3.py input1.txt 8 15 22
```

### 코드 설명
이 코드는 DBSCAN 알고리즘을 구현하여 2D 좌표 데이터를 클러스터링합니다. 주요 구성 요소는 다음과 같습니다:

- **`eucDistanceFunc(a, b)`**: 두 점 간 유클리드 거리를 계산합니다.
- **`rangeQuery(db, p, eps)`**: 점 `p`에서 `eps` 반경 내 이웃 점을 찾습니다.
- **`dbscan(db, eps, minPts)`**: DBSCAN 알고리즘을 실행합니다. 핵심 점을 기준으로 클러스터를 확장하며, `minPts` 미만인 점은 노이즈(-1)로 처리합니다.
- **`main()`**: 입력 파일을 읽고, 클러스터링 후 상위 `n`개 클러스터를 크기 순으로 파일에 저장합니다.

</details>
---