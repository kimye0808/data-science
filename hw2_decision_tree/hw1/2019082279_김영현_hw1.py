import sys
from itertools import combinations

def getTransactionsList(fileName):
    transactions = [];
    with open(fileName, 'r') as file:
        for line in file:
            lineContent = line.strip();
            itemId = lineContent.split('\t');
            transaction = list(map(int, itemId));
            transactions.append(transaction);
    return transactions;

def calcSupport(transactions, itemSet):
    count = 0;
    total = len(transactions);
    for transaction in transactions:
        if all(item in transaction for item in itemSet):
            count += 1
    support = (count / total) * 100;
    return support;



def getFrequentItemSets(transactions, minSup):
    items = set();
    frequentItemSets = [];
    candidateItemSetSize= 2;

    for transaction in transactions:
        for item in transaction:
            items.add(frozenset([item]));


    while True:
        candidates = set();
        frequentCandidates = [];

        for candidate in items:
            for item in items:
                newCandidate = candidate.union(item);
                if len(newCandidate) == candidateItemSetSize:
                    candidates.add(newCandidate);
        
        if len(candidates) == 0:
            break;
        for candidate in candidates:
            support = calcSupport(transactions, candidate);

            if support >= minSup:
                frequentItemSets.append(candidate);
                frequentCandidates.append(candidate);
        items = frequentCandidates;
        candidateItemSetSize += 1;
    
    return frequentItemSets;




def writeAssociationRules(transactions, frequentItemsets, outputFile):

    with open(outputFile, 'w') as file:
        for freqSet in frequentItemsets:
            for k in range(1, len(freqSet)):

                for subset in combinations(freqSet, k):
                
                    itemSet = set(subset);
                    associativeItemSet = freqSet - itemSet;

                    support = calcSupport(transactions, freqSet);
                    confidence = (calcSupport(transactions, freqSet) / calcSupport(transactions, itemSet)) * 100;
                    
                    stringItemSet = f"{{{', '.join(map(str, itemSet))}}}";
                    stringAssociativeItemSet = f"{{{', '.join(map(str, associativeItemSet))}}}";

                    file.write(f"{stringItemSet}\t{stringAssociativeItemSet}\t{support:.2f}\t{confidence:.2f}\n");


def main():
    minSup = float(sys.argv[1]);
    inputFile = sys.argv[2];
    outputFile = sys.argv[3];

    transactions = getTransactionsList(inputFile);
    frequentItemSets = getFrequentItemSets(transactions, minSup);
    writeAssociationRules(transactions, frequentItemSets, outputFile);


if __name__ == '__main__':
    main();