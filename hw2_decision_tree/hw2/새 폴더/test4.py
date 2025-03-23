import math

# 함수 정의: 데이터셋에서 가능한 클래스의 빈도수 계산
def class_counts(rows):
    counts = {}  # 클래스: 빈도수 딕셔너리
    for row in rows:
        label = row[-1]  # 각 행의 클래스
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# 함수 정의: 데이터셋을 속성과 속성 값으로 분할
def partition(rows, attribute):
    partitions = {}  # 속성 값: 해당하는 행들 리스트
    for row in rows:
        value = row[attribute]
        if value not in partitions:
            partitions[value] = []
        partitions[value].append(row)
    return partitions

# 함수 정의: 엔트로피 계산
def entropy(rows):
    counts = class_counts(rows)
    entropy_val = 0
    total = len(rows)
    for label in counts:
        prob = counts[label] / total
        entropy_val -= prob * math.log2(prob)
    return entropy_val

# 함수 정의: 정보 획득 계산
def information_gain(current_entropy, partitions):
    total = sum(len(partition) for partition in partitions.values())
    info_gain = current_entropy
    for partition in partitions.values():
        info_gain -= (len(partition) / total) * entropy(partition)
    return info_gain

# 함수 정의: 최적의 속성 선택
def find_best_split(rows):
    attributes = len(rows[0]) - 1
    current_entropy = entropy(rows)
    best_info_gain = 0
    best_attribute = None
    for attribute in range(attributes):
        partitions = partition(rows, attribute)
        info_gain = information_gain(current_entropy, partitions)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute = attribute
    return best_attribute

# 함수 정의: 결정 트리 구축
def build_tree(rows):
    if len(rows) == 0:
        return None
    if len(class_counts(rows)) == 1:
        return list(class_counts(rows).keys())[0]
    best_attribute = find_best_split(rows)
    partitions = partition(rows, best_attribute)
    subtree = {best_attribute: {}}
    for value, partitioned_rows in partitions.items():
        subtree[best_attribute][value] = build_tree(partitioned_rows)
    return subtree

# 함수 정의: 예측 수행
def predict(tree, row):
    if isinstance(tree, str):
        return tree
    attribute = list(tree.keys())[0]
    value = row[attribute]
    if value not in tree[attribute]:
        return "Unknown"
    subtree = tree[attribute][value]
    return predict(subtree, row)

# 함수 정의: 예측 결과 저장
def classify_test_data(train_file, test_file, result_file):
    # 훈련 데이터 불러오기
    with open(train_file, 'r') as file:
        train_data = [line.strip().split('\t') for line in file.readlines()]
    # 테스트 데이터 불러오기
    with open(test_file, 'r') as file:
        test_data = [line.strip().split('\t') for line in file.readlines()]
    # 결정 트리 구축
    tree = build_tree(train_data)
    # 분류 결과 예측
    predictions = [predict(tree, row) for row in test_data]
    # 예측 결과 저장
    with open(result_file, 'w') as file:
        for prediction in predictions:
            file.write(prediction + '\n')

# 메인 함수
def main():
    # 실행 시 인자로 전달된 파일 이름들
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]
    # 테스트 데이터 분류 및 결과 저장
    classify_test_data(train_file, test_file, result_file)

if __name__ == "__main__":
    main()
