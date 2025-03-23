import sys;
import math;
import os;

def getClusterSize(cluster):
    return len(cluster[1]);

def eucDistanceFunc(a, b):
    distanceResult = math.sqrt(math.pow((a[1] - b[1]), 2) + math.pow((a[2] - b[2]), 2));

    return distanceResult;


def rangeQuery(db, p, eps):
    neighborsN = [];

    for i in range(len(db)):
        if eucDistanceFunc(db[p], db[i]) <= eps:
            neighborsN.append(i);
    
    return neighborsN;




def dbscan(db, eps, minPts):

    labels = len(db) * [None];
    c = 0;

    for p in range(len(db)):

        if labels[p] is not None:
            continue;

        neighborsN = rangeQuery(db,  p, eps);

        if len(neighborsN) < minPts:
            labels[p] = -1;
            continue;

        c += 1
        labels[p] = c;

        seedSet = set(neighborsN) - {p};

        while seedSet:
            q = seedSet.pop();

            if labels[q] == -1:
                labels[q] = c;

            if labels[q] is not None:
                continue;

            labels[q] = c;
            neighborsQ = rangeQuery(db, q, eps);
            if len(neighborsQ) < minPts:
                continue;
            seedSet.update(set(neighborsQ));

    return labels;





if __name__ == "__main__":

    inputFile = sys.argv[1];
    n = int(sys.argv[2]);
    eps = float(sys.argv[3]);
    minPts = int(sys.argv[4]);

    db = [];
    with open(inputFile, 'r') as file:
        for line in file:

            id, a, b = line.strip().split('\t');
            db.append((id, float(a), float(b)));

    
    
    labels = dbscan(db, eps, minPts);




    clusters = {};
    i = 0;
    for label in labels:
        if label > 0:
            if label not in clusters:
                clusters[label] = [];
            clusters[label].append(db[i][0]);
        i += 1


    sortedClusters = list(clusters.items());
    sortedClusters.sort(key=lambda x: len(x[1]), reverse=True);
    
    fileName, _ = os.path.splitext(inputFile);

    for i in range(min(n, len(sortedClusters))):
        with open(f'{fileName}_cluster_{i}.txt', 'w') as file:
            for id in sortedClusters[i][1]:
                file.write(f'{id}\n')
