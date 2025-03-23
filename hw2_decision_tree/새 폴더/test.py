import sys
import pandas as pd
import math

# 엔트로피를 계산하는 함수
def entropy(data, target_column):
    label_counts = data[target_column].value_counts()
    entropy_value = 0
    for count in label_counts:
        probability = count / len(data)
        entropy_value -= probability * math.log2(probability)
    return entropy_value

# 정보 이득을 계산하는 함수
def information_gain(data, split_attribute, target_column):
    total_entropy = entropy(data, target_column)
    values = data[split_attribute].unique()
    
    weighted_entropy = 0
    for value in values:
        subset = data[data[split_attribute] == value]
        proportion = len(subset) / len(data)
        weighted_entropy += proportion * entropy(subset, target_column)
        
    return total_entropy - weighted_entropy

# 속성 분할의 효율성을 측정하는 Gain Ratio 계산
def gain_ratio(data, split_attribute, target_column):
    attribute_counts = data[split_attribute].value_counts()
    
    split_info = 0
    for count in attribute_counts:
        probability = count / len(data)
        split_info -= probability * math.log2(probability)
    
    info_gain = information_gain(data, split_attribute, target_column)
    if split_info == 0:
        return 0
    else:
        return info_gain / split_info

# 가장 좋은 속성을 선택하는 함수
def best_attribute(data, attributes, target_column):
    best_gain_ratio = -1
    best_attr = None
    
    for attribute in attributes:
        ratio = gain_ratio(data, attribute, target_column)
        if ratio > best_gain_ratio:
            best_gain_ratio = ratio
            best_attr = attribute
    
    return best_attr

# 결정 트리를 구축하는 함수 (딕셔너리 기반)
def build_decision_tree(data, attributes, target_column):
    # 모든 레이블이 같은 경우, 리프 노드를 반환합니다.
    if len(data[target_column].unique()) == 1:
        return {"is_leaf": True, "label": data[target_column].iloc[0]}
    
    # 더 이상 속성이 없는 경우, 가장 일반적인 레이블을 반환합니다.
    if len(attributes) == 0:
        common_label = data[target_column].mode()[0]
        return {"is_leaf": True, "label": common_label}
    
    # 가장 좋은 속성을 선택하고, 노드를 만듭니다.
    best_attr = best_attribute(data, attributes, target_column)
    tree = {"attribute": best_attr, "children": {}}
    
    # 선택한 속성의 고유한 값을 기반으로 분할합니다.
    unique_values = data[best_attr].unique()
    for value in unique_values:
        subset = data[data[best_attr] == value]
        new_attributes = [attr for attr in attributes if attr != best_attr]
        child_node = build_decision_tree(subset, new_attributes, target_column)
        tree["children"][value] = child_node
    
    return tree

# 트리 구조에서 예측을 수행하는 함수
def predict(tree, data_row):
    if tree["is_leaf"]:
        return tree["label"]
    
    attribute_value = data_row[tree["attribute"]]
    if attribute_value in tree["children"]:
        return predict(tree["children"][attribute_value], data_row)
    else:
        return None

# 결과를 파일에 쓰는 함수
def write_results(test_data, predictions, result_file):
    test_data["Predicted_Class"] = predictions
    test_data.to_csv(result_file, sep="\t", index=False)

# 스크립트 실행 메인 함수
def main():
    if len(sys.argv) != 4:
        print("Usage: python <script_name> <training_file> <test_file> <result_file>")
        sys.exit(1)
    
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]
    
    train_data = pd.read_csv(training_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")
    
    attribute_names = list(train_data.columns)
    target_column = attribute_names[-1]
    attributes = attribute_names[:-1]
    
    # 결정 트리 구축
    decision_tree = build_decision_tree(train_data, attributes, target_column)
    
    # 테스트 데이터에 대한 예측 수행
    predictions = test_data.apply(lambda row: predict(decision_tree, row), axis=1)
    
    # 결과를 파일에 쓰기
    write_results(test_data, predictions, result_file)

# 스크립트 실행
if __name__ == "__main__":
    main()
