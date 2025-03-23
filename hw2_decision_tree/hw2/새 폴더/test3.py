import sys
import csv
from collections import Counter
import math

# 필요한 데이터 읽기
def read_dataset(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)  # 첫 번째 행은 헤더
        dataset = [row for row in reader]
    return headers, dataset

# 정보 획득량 계산
def entropy(data, target_attr):
    val_freq = Counter([record[target_attr] for record in data])
    data_entropy = 0.0
    
    for freq in val_freq.values():
        prob = freq / len(data)
        data_entropy -= prob * math.log2(prob)
    
    return data_entropy

# 정보 획득량으로 최적 속성 선택
def gain(data, attr, target_attr):
    val_freq = Counter([record[attr] for record in data])
    subset_entropy = 0.0
    
    for val, freq in val_freq.items():
        val_prob = freq / len(data)
        subset_data = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(subset_data, target_attr)
    
    total_entropy = entropy(data, target_attr)
    gain = total_entropy - subset_entropy
    return gain

# 의사결정 트리 구축
def build_decision_tree(data, attributes, target_attr, default_class=None):
    if not data:
        return default_class
    
    class_values = [record[target_attr] for record in data]
    default_class = Counter(class_values).most_common(1)[0][0]
    
    if len(set(class_values)) == 1:
        return class_values[0]
    
    if not attributes:
        return default_class
    
    best_attr = max(attributes, key=lambda attr: gain(data, attr, target_attr))
    
    tree = {best_attr: {}}
    
    for val in set(record[best_attr] for record in data):
        subset = [record for record in data if record[best_attr] == val]
        new_attrs = [a for a in attributes if a != best_attr]
        subtree = build_decision_tree(subset, new_attrs, target_attr, default_class)
        tree[best_attr][val] = subtree
    
    return tree

# 의사결정 트리로 예측
def classify(tree, instance):
    if not isinstance(tree, dict):
        return tree
    
    attr = next(iter(tree.keys()))
    subtree = tree[attr][instance[attr]]
    return classify(subtree, instance)

# 테스트 데이터에 대한 예측
def predict(tree, headers, test_data):
    results = []
    for instance in test_data:
        prediction = classify(tree, {headers[i]: instance[i] for i in range(len(instance))})
        results.append(instance + [prediction])
    return results

# 실행 부분
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <train_data> <test_data> <result_file>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]
    
    # 데이터 읽기
    train_headers, train_data = read_dataset(train_file)
    test_headers, test_data = read_dataset(test_file)
    
    # 타겟 속성 및 속성 목록 설정
    target_attr = len(train_headers) - 1
    attributes = list(range(target_attr))
    
    # 결정 트리 구축
    decision_tree = build_decision_tree(train_data, attributes, target_attr)
    
    # 테스트 데이터 예측 및 결과 출력
    predictions = predict(decision_tree, train_headers, test_data)
    
    with open(result_file, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        # 첫 행에 헤더 출력
        writer.writerow(train_headers)
        # 예측 결과 출력
        for result in predictions:
            writer.writerow(result)
