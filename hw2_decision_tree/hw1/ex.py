import sys
from itertools import combinations

def read_transactions(file_name):
    transactions = []
    with open(file_name, 'r') as file:
        for line in file:
            transaction = list(map(int, line.strip().split('\t')))
            transactions.append(transaction)
    return transactions

def calculate_support(transactions, item_set):
    count = sum(1 for transaction in transactions if all(item in transaction for item in item_set))
    return (count / len(transactions)) * 100

def get_frequent_item_sets(transactions, min_sup):
    items = set(item for transaction in transactions for item in transaction)
    frequent_item_sets = []

    for k in range(1, len(items) + 1):
        candidates = [set(comb) for comb in combinations(items, k)]
        frequent_item_sets.extend([item_set for item_set in candidates if calculate_support(transactions, item_set) >= min_sup])

    return frequent_item_sets

def generate_association_rules(transactions, frequent_itemsets, output_file):
    with open(output_file, 'w') as file:
        for freq_set in frequent_itemsets:
            for k in range(1, len(freq_set)):
                for subset in combinations(freq_set, k):
                    item_set = set(subset)
                    associative_item_set = freq_set - item_set

                    support = calculate_support(transactions, freq_set)
                    confidence = (calculate_support(transactions, freq_set) / calculate_support(transactions, item_set)) * 100

                    string_item_set = '{{{}}}'.format(', '.join(map(str, item_set)))
                    string_associative_item_set = '{{{}}}'.format(', '.join(map(str, associative_item_set)))

                    file.write(f"{string_item_set}\t{string_associative_item_set}\t{support:.2f}\t{confidence:.2f}\n")

if __name__ == "__main__":
    min_sup = float(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    transactions = read_transactions(input_file)
    print("get transaction list done")

    frequent_item_sets = get_frequent_item_sets(transactions, min_sup)
    print("get frequent item set done")

    generate_association_rules(transactions, frequent_item_sets, output_file)
    print("get association rules done")
