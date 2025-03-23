import math;
import sys;
import pandas;

def getFilteredFeatures(data, feature, value):
  filteredFeatures = data[data[feature] == value]
    
  return filteredFeatures;


def getClassLabelCount(classLabels):
  classCounts = {};
  for label in classLabels:
    if label in classCounts:
      classCounts[label] += 1;
    else:
      classCounts[label] = 1;
  return classCounts;


# entropy 계산 Info(D)
def calcEntropy(classLabels):
  length = len(classLabels);
  entropy = 0;

  if length == 0:
    return 0;
    
  classCounts = getClassLabelCount(classLabels);
  if classCounts == {}:
    return 0;
    
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
    partition = getFilteredFeatures(data, feature, featureValue);

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

  if(length == 0):
    return 0;

  featureValues = data[feature].unique();
    
  splitInfo = 0;
  for featureValue in featureValues:
    partitionRatio = len( getFilteredFeatures(data, feature, featureValue)) / length;
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
  maxGainRatio = 0;
  bestFeature = None;
    
  for feature in features:
    if feature == classLabel:
      continue;
        
    tmpGainRatio = calcGainRatio( data, feature, classLabel );
        
    
    # maxGainRatio = max(maxGainRatio, tmpGainRatio);

    if maxGainRatio < tmpGainRatio :

      maxGainRatio = tmpGainRatio;
      bestFeature = feature;
    
  return bestFeature;



def buildDecisionTree( data, features, classLabel ):
  uniqVal = None;
  uniqVals = None;
  modVals = None;

  uniqVals = data[classLabel].unique()
  uniqVal = uniqVals[0] if uniqVals.size > 0 else None

  if len(uniqVals) == 1:
    return uniqVal;

  modVal = data[classLabel].mode()[0] if not data[classLabel].mode().empty else None
  
  # print("class label: ", classLabel);

  # if data is not None and classLabel in data.columns and data[classLabel].notna().any():
  #   uniqVal = data[classLabel].unique()[0];
  #   uniqVals = data[classLabel].unique();
  #   modVals = data[classLabel].mode()[0];
  
  # if len(uniqVals) == 1:
  #   return uniqVal;
  
  feature = getBestFeature( data, features, classLabel );
    
  if feature is None:
    return modVal;
    
  tree = {feature: {}};
  
  uniqFeats = data[feature].unique();

  for value in uniqFeats:
    featureData = getFilteredFeatures(data, feature, value);
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


def getCommonClassTrainingData( traingingData, classLabel):
  classLabels = traingingData[classLabel];

  commonClassLabel = classLabels.mode().iloc[0];
    
  return commonClassLabel;


def getNearNodeClass(tree, rowData, root):
  values = list(tree[root].keys());
  if len(values) > 0:
    closeValue = values[0]
    return getNearNodeClass(tree[root][closeValue], rowData, root);
  else:
    return None;


def getTestResult( tree, rowData, major ):
  if type(tree) is dict:
    root = list(tree.keys())[0];
    value = rowData[root];
    if value in tree[root]:
      return getTestResult(tree[root][value], rowData, major );
    else:
      return major;
      # return getMajorityClass(tree);
      # return getNearNodeClass(tree, rowData, root);


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
  
  commonClass = getCommonClassTrainingData(trainingData, classLabel);

  testResult = [];
  for _, row in testData.iterrows():
    result = getTestResult(tree, row, majorityClass );
    # result = getTestResult(tree, row, commonClass);
    testResult.append(result);

  resultData = testData.copy();
  resultData[classLabel] = testResult;

  resultData.to_csv( resultFile, sep="\t", index=False );

if __name__ == '__main__':
  main();