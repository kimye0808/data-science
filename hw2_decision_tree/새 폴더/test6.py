
import sys
import numpy as np

class Feature:
    def __init__(self, feature, i):
        self.featureName = feature #해당 feature의 이름
        self.feature_value_list = [] #feature이 value로 가질 수 있는 값들의 리스트
        self.tag = i #디비에서 feature의 순서
    

class Node:
    def __init__(self):
        self.feature = None #feature 종류, nonleaf이면 값이 있고 아니면 None임
        self.children = [] #child 노드
        self.class_label = None #leaf node일 때 class_label
        self.feature_value = None # 부모노드 feature에 의해 나눠질 때의 해당 feature value
    
    # split하기 전의 entropy값
    def calculate_before_entropy(self, class_labels, data_set):
        entropy = 0
        data_labels = data_set.T[-1]
        for label in class_labels.feature_value_list:
            count = np.sum(data_labels==label)
            if count==0:
                continue
            p_i = count / data_labels.size
            entropy += p_i*np.log2(p_i)
        return -entropy
    # split 후의 entropy 값
    def calculate_after_entropy(self, class_labels, data_set, feature):
        entropy = 0
        attr_values = feature.feature_value_list

        for attr in attr_values:
            data_subset = data_set[data_set.T[feature.tag]==attr]
            p_i = data_subset.shape[0]/data_set.shape[0]
            
            entropy += p_i*self.calculate_before_entropy(class_labels, data_subset)

        return entropy
    # splitinfo를 계산
    def calculate_splitInfo(self, data_set, feature):
        entropy = 0
        attr_values = feature.feature_value_list

        for attr in attr_values:
            data_subset = data_set[data_set.T[feature.tag]==attr]
            p_i = data_subset.shape[0]/data_set.shape[0]
            if p_i==0:
                continue
            entropy += p_i*np.log2(p_i)

        return -entropy
    #리프노드에서 majority voting하는 방법
    def majority_voting(self, class_labels, data_set):
        major_label = None
        max_count = 0
        data_labels = data_set.T[-1]
        for label in class_labels.feature_value_list:
            count = np.sum(data_labels==label)
            if count >= max_count:
                max_count = count
                major_label = label
        return major_label
    #decision tree 학습
    def fit(self, data_set, remain_feature_list, class_label):
        before_entropy = self.calculate_before_entropy(class_label, data_set)
        
        #stop 조건
        if before_entropy == 0:
            self.class_label = data_set[0][-1]
            return
        elif remain_feature_list.size == 0 :
            self.class_label = self.majority_voting(class_label, data_set)
            return


        max_feature = remain_feature_list[0]
        max_gain_ratio = sys.float_info.min
        
        for feature in remain_feature_list:
            info_gain = before_entropy - self.calculate_after_entropy(class_label, data_set, feature)
            split_info = self.calculate_splitInfo(data_set, feature)
            gain_ratio = info_gain / split_info
            if gain_ratio >= max_gain_ratio:
                max_gain_ratio = gain_ratio
                max_feature = feature
        
        self.feature = max_feature
        
        for feature_value in max_feature.feature_value_list:
            data_subset = data_set[data_set.T[max_feature.tag]==feature_value]

            new_remain_feature_list = np.delete(remain_feature_list, np.where(remain_feature_list == max_feature))
            new_leaf = Node()
            new_leaf.feature_value = feature_value

            if data_subset.size != 0:
                new_leaf.fit(data_subset, new_remain_feature_list ,class_label)
            else: #해당 노드에 데이터가 없는 경우
                new_leaf.class_label = self.majority_voting(class_label, data_set)
            self.children.append(new_leaf)
    #decision tree으로 예측
    def predict(self, data):
        if self.feature == None:
            return self.class_label
        
        # for i, item in enumerate(self.feature.feature_value_list):
        #     if item == data[self.feature.tag]:
        #         return self.children[i].predict(data)
        for child in self.children:
            if child.feature_value == data[self.feature.tag]:
                return child.predict(data)

    #train file, test file 읽기
def read_db_file(file_name):
    f = open("./" + file_name,'r')

    read_feature_list = f.readline().split()
    feature_list = np.array([])
    for i, feature in enumerate(read_feature_list):
        feature_list = np.append(feature_list, np.array([Feature(feature, i )]))
    
    input_list = f.readlines()
    transaction_list = np.empty((0,feature_list.size))


    for input in input_list:
        item_set = np.array(input.strip().split('\t'))
        transaction_list = np.append(transaction_list,  np.array([item_set]), axis = 0)
    
        for i, item in enumerate(item_set):
            if item not in feature_list[i].feature_value_list:
                feature_list[i].feature_value_list.append(item)
    
    f.close()
    class_label = feature_list[-1]
    feature_list = feature_list[:-1]
    

    return feature_list, class_label, transaction_list
    #output file 생성
def create_output_file(model, output_file_name, test_file_name, feature_list, class_label):
    output_file = open(output_file_name, 'w')
    _, _, test_data = read_db_file(test_file_name)
    result = ""
    for item in feature_list:
        result += item.featureName+"\t"
    
    output_file.write(result + class_label.featureName + "\n")

    for data in test_data:
        result = ""
        for item in data:
            result += item+"\t"
        result += model.predict(data) +"\n"
        output_file.write(result)

    output_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("argv is not correct! please check argv one more time")
        quit()

    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    feature_list, class_label, transaction_list = read_db_file(train_file_name)

    model = Node()
    model.fit( transaction_list, feature_list, class_label)

    create_output_file(model, output_file_name, test_file_name, feature_list, class_label)
    print("done")
    