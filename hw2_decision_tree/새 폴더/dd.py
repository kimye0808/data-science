import math;
import sys;
import pandas;



# entropy 계산 Info(D)
def calcEntropy(classLabels):
  length = len(classLabels);
  entropy = 0;

  if length == 0:
    return 0;
    
  classCounts = {};
  for label in classLabels:
    if label in classCounts:
      classCounts[label] += 1;
    else:
      classCounts[label] = 1;
        
  for count in classCounts.values():
    p = count / length;
    entropy += p * math.log2(p);
    
  entropy = -entropy;

  return entropy;

# entropy 계산 Info(Dj)
def calcEntropyAfterSplit( data, feature, classLabel ):
  featureValues = data[feature].unique();
  entropy = 0;
    
  for featureValue in featureValues:
    partition = data[data[feature] == featureValue];
    partitionRatio = len(partition) / len(data);

    entropy += partitionRatio * calcEntropy( partition[classLabel] );
  
  return entropy;

# Information Gain 계산 Gain(A) 
def calcInformationGain( data, feature, classLabel ):
  infoD = calcEntropy( data[classLabel] );
  infoDj = calcEntropyAfterSplit( data, feature, classLabel );

  informationGain = infoD - infoDj;
  return informationGain;


# split info 계산 SplitInfoA(D)
def calcSplitInfo( data, feature ):
  length = len(data);
  featureValues = data[feature].unique();
    
  splitInfo = 0;
  for featureValue in featureValues:
    partitionRatio = len( data[data[feature] == featureValue] ) / length;
    splitInfo += partitionRatio * math.log2(partitionRatio);
    
  splitInfo = -splitInfo;

  return splitInfo;

# gain ratio 계산 GainRatio(A)
def calcGainRatio( data, feature, classLabel ):
  splitInfo = calcSplitInfo( data, feature );
  if splitInfo == 0:
    return 0;

  informationGain = calcInformationGain( data, feature, classLabel );
    
  return informationGain / splitInfo;


# split용 feature 선택
def getBestFeature( data, features, classLabel ):
  max = 0;
  bestFeature = None;
    
  for feature in features:
    if feature == classLabel:
      continue;
        
    tmp = calcGainRatio( data, feature, classLabel );
        
    if max < tmp :
      max = tmp;
      bestFeature = feature;
    
  return bestFeature;



def buildDecisionTree( data, features, classLabel ):
  if len(data[classLabel].unique()) == 1:
    return data[classLabel].unique()[0];
  
  feature = getBestFeature( data, features, classLabel );
    
  if feature is None:
    return data[classLabel].mode()[0];
    
  tree = {
    feature: {}
  };
    
  for value in data[feature].unique():
    featureData = data[data[feature] == value];
    restFeatures = [];
    for feat in features:
      if feat != feature:
        restFeatures.append(feat);
  
    subTree = buildDecisionTree( featureData, restFeatures, classLabel);
    if subTree is None:
      continue;
            
    tree[feature][value] = subTree
      
  return tree


# 조건이 없을 경우 가장 많이 선택된 클래스로 정함
def getMajorityClass( tree):
  if type(tree) is dict:
    classesGroup = [];
    for value in tree.values():
      if type(value)is dict:
        classesGroup.append(getMajorityClass(value));
      else:
        classesGroup.append(value);
        
    return max(set(classesGroup), key=classesGroup.count);
  else:
    return tree;


def getTestResult( tree, rowData, major ):
  if type(tree) is dict:
    root = list(tree.keys())[0];
    value = rowData[root];
    if value in tree[root]:
      return getTestResult(tree[root][value], rowData, major );
    else:
      return major;

  else:
    return tree;
    


def main():
  trainingFile = sys.argv[1];
  testFile = sys.argv[2];
  resultFile = sys.argv[3];

  trainingData = pandas.read_csv(trainingFile, sep="\t", engine='python');
  testData = pandas.read_csv(testFile, sep="\t", engine='python');

  classLabel = trainingData.columns[-1];
  features = list( trainingData.columns[ :-1 ]);



  tree = buildDecisionTree(trainingData, features, classLabel);

  majorityClass = getMajorityClass(tree);

  testResult = [];
  for _, row in testData.iterrows():
    result = getTestResult(tree, row, majorityClass );
    testResult.append(result);

  resultData = testData.copy();
  resultData[classLabel] = testResult;

  resultData.to_csv( resultFile, sep="\t", index=False );

if __name__ == '__main__':
  main();