import sys
import pandas as pd
from collections import defaultdict
import math

# Helper function to calculate entropy
def entropy(data):
    total = len(data)
    class_counts = defaultdict(int)
    for row in data:
        class_label = row[-1]
        class_counts[class_label] += 1
    entropy_value = 0
    for label, count in class_counts.items():
        probability = count / total
        if probability > 0:
            entropy_value -= probability * math.log2(probability)
    return entropy_value

# Function to calculate information gain
def info_gain(data, attribute_index, base_entropy):
    # Get the possible values for the attribute
    attribute_values = set(row[attribute_index] for row in data)
    weighted_entropy = 0
    total_rows = len(data)
    
    # Calculate entropy for each partition based on the attribute value
    for value in attribute_values:
        subset = [row for row in data if row[attribute_index] == value]
        weighted_entropy += (len(subset) / total_rows) * entropy(subset)
    
    # Information gain is the reduction in entropy
    return base_entropy - weighted_entropy

# Function to find the best attribute to split on
def best_attribute(data, attributes):
    base_entropy = entropy(data)
    best_gain = -1
    best_attr = -1
    
    for i, attribute in enumerate(attributes[:-1]):
        gain = info_gain(data, i, base_entropy)
        if gain > best_gain:
            best_gain = gain
            best_attr = i
            
    return best_attr

# Function to create the decision tree
def create_tree(data, attributes):
    # If all classes are the same, return that class
    class_labels = [row[-1] for row in data]
    if class_labels.count(class_labels[0]) == len(class_labels):
        return class_labels[0]
    
    # If there are no more attributes to split on, return the most common class
    if len(attributes) == 1:
        return max(set(class_labels), key=class_labels.count)
    
    # Select the best attribute to split on
    best_attr_index = best_attribute(data, attributes)
    best_attr_name = attributes[best_attr_index]
    
    tree = {best_attr_name: {}}
    
    # Split the data based on the best attribute
    attribute_values = set(row[best_attr_index] for row in data)
    for value in attribute_values:
        subset = [row for row in data if row[best_attr_index] == value]
        subset_attributes = attributes[:best_attr_index] + attributes[best_attr_index + 1:]
        subtree = create_tree(subset, subset_attributes)
        tree[best_attr_name][value] = subtree
    
    return tree

# Function to classify a single instance using the decision tree
def classify(instance, tree):
    if not isinstance(tree, dict):
        return tree  # leaf node
    
    attribute_name = list(tree.keys())[0]
    subtree = tree[attribute_name]
    
    attribute_value = instance[attribute_name]
    if attribute_value in subtree:
        return classify(instance, subtree[attribute_value])
    else:
        return None  # unknown case

# Load the training data
def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip().split('\t') for line in file]
    attributes = lines[0]
    data = lines[1:]
    return attributes, data

# Main function
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python <script> <train_file> <test_file> <output_file>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    # Load training data
    train_attributes, train_data = load_data(train_file)
    test_attributes, test_data = load_data(test_file)

    # Ensure attributes in the train and test data match
    if train_attributes[:-1] != test_attributes:
        print("Train and test attributes do not match")
        sys.exit(1)

    # Build the decision tree
    decision_tree = create_tree(train_data, train_attributes)

    # Classify the test data and save the results
    results = []
    for instance in test_data:
        instance_dict = dict(zip(test_attributes, instance))
        predicted_class = classify(instance_dict, decision_tree)
        results.append('\t'.join(instance + [predicted_class]))

    # Write the results to the output file
    with open(output_file, 'w') as f:
        f.write('\t'.join(test_attributes + [train_attributes[-1]]) + '\n')
        f.write('\n'.join(results))
