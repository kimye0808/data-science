import sys
import pandas as pd
import numpy as np
from collections import Counter
import math

# Read a dataset from a file
def read_dataset(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    headers = lines[0].strip().split('\t')
    data = [line.strip().split('\t') for line in lines[1:]]
    
    return headers, data

# Calculate entropy of a dataset
def entropy(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    total = len(data)
    
    ent = 0
    for label in label_counts:
        prob = label_counts[label] / total
        ent -= prob * math.log2(prob)
    
    return ent

# Split a dataset by a specific attribute and its value
def split_dataset(data, attr_index, value):
    return [row for row in data if row[attr_index] == value]

# Find the best attribute to split the dataset
def best_split(data, attributes):
    base_entropy = entropy(data)
    best_gain = -1
    best_attr = None
    
    for attr in attributes:
        attr_index = attributes.index(attr)
        # Get all unique values for this attribute
        unique_values = set([row[attr_index] for row in data])
        
        # Calculate the weighted entropy after splitting
        weighted_entropy = 0
        for value in unique_values:
            subset = split_dataset(data, attr_index, value)
            prob = len(subset) / len(data)
            weighted_entropy += prob * entropy(subset)
        
        # Information gain
        info_gain = base_entropy - weighted_entropy
        
        if info_gain > best_gain:
            best_gain = info_gain
            best_attr = attr
    
    return best_attr

# Build a decision tree
def build_tree(data, attributes):
    # If all examples are in the same class, return the class
    class_labels = [row[-1] for row in data]
    if len(set(class_labels)) == 1:
        return class_labels[0]
    
    # If no more attributes, return the majority class
    if not attributes:
        return Counter(class_labels).most_common(1)[0][0]
    
    # Find the best attribute to split on
    best_attr = best_split(data, attributes)
    
    # Create a new tree node
    tree = {best_attr: {}}
    attr_index = attributes.index(best_attr)
    unique_values = set([row[attr_index] for row in data])
    
    # Split the dataset and build subtrees
    for value in unique_values:
        subset = split_dataset(data, attr_index, value)
        subtree_attributes = [attr for attr in attributes if attr != best_attr]
        subtree = build_tree(subset, subtree_attributes)
        
        tree[best_attr][value] = subtree
    
    return tree

# Classify a single instance using a decision tree
def classify(tree, instance, attributes):
    if isinstance(tree, str):
        return tree
    
    # Get the current attribute to check
    attr_name = list(tree.keys())[0]
    attr_index = attributes.index(attr_name)
    
    # Get the subtree based on the instance's attribute value
    attr_value = instance[attr_index]
    subtree = tree[attr_name].get(attr_value, None)
    
    if subtree is None:
        # If no subtree is found for the given attribute value, use the majority class
        # This is a fallback for unexpected values in the test set
        subtree = Counter([tree[attr_name][v] for v in tree[attr_name]]).most_common(1)[0][0]
    
    return classify(subtree, instance, attributes)

# Write the classification results to a file
def write_results(file_path, headers, data, predictions):
    with open(file_path, 'w') as f:
        # Write the header
        f.write('\t'.join(headers) + '\n')
        
        # Write each instance with its predicted class
        for i, instance in enumerate(data):
            row = instance + [predictions[i]]
            f.write('\t'.join(row) + '\n')

# Main function
def main():
    # Get the command line arguments
    if len(sys.argv) != 4:
        print("Usage: python <script_name> <training_dataset> <test_dataset> <result_dataset>")
        return
    
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]
    
    # Read the training and test datasets
    train_headers, train_data = read_dataset(training_file)
    test_headers, test_data = read_dataset(test_file)
    
    # Build the decision tree
    attributes = train_headers[:-1]
    tree = build_tree(train_data, attributes)
    
    # Classify the test dataset
    predictions = [classify(tree, test_instance, attributes) for test_instance in test_data]
    
    # Write the results to the output file
    write_results(result_file, train_headers, test_data, predictions)

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
