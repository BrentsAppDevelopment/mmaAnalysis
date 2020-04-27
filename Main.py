import tensorflow
import keras
import copy
import sklearn
import pandas as pd
import numpy as np

from sklearn import linear_model, preprocessing
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import svm, metrics
from sklearn.cluster import KMeans


sherdogData = pd.read_csv("mma_data_sherdog.csv")
fightMetricData = pd.read_csv("mma_data_fightmetric.csv")

def CleanFightMetric():
    global fightMetricData

    for method in fightMetricData.Method:
        if 'KO/TKO' in method:      # if fight ended by Knockout or Technical Knockout
            method = 0
        elif 'SUB' in method:       # if fight ended by Submission
            method = 1
        elif 'DEC' in method:       # if fight ended by Decision
            method = 2
        elif 'DQ' in method:        # if fight ended by Disqualification
            method = 3
        elif 'CNC' in method:       # if fight ended by No Contest
            method = 4
        # print(method)

    for time in range (0, len(fightMetricData.Time)):
        (m, s) = fightMetricData.Time[time].split(':')
        tempTime = int(m)+(int(s)/60)
        fightMetricData.Time[time] = ((((fightMetricData.Round[time])-1)*5)+tempTime)


def CleanSherdog():
    numRows = 0
    for winLoss in sherdogData.Result:
        numRows += 1
        if(winLoss == 'win'):
            winLoss = 1
        elif(winLoss == 'loss'):
            winLoss = 0
        else:
            winLoss = 0
        # print(winLoss)
    # print(numRows)


def CleanFrames():
    CleanFightMetric()
    CleanSherdog()


def GenerateFightMetric():
    global fightMetricData
    listFightMetrics = ["Sig Strikes", "Takedowns", "Sub Attempts", "Guard Passes"]

    newDF = {
        "Sig Strikes Rate": [],
        "Takedowns Rate": [],
        "Sub Attempts Rate": [],
        "Guard Passes Rate": []
    }
    for newCol in newDF:
        my_data = copy.deepcopy(fightMetricData.Time)   #      np.random.random((210, 8))  # recfromcsv('LIAB.ST.csv', delimiter='\t')
        newDF.get(newCol).extend(my_data)

    newDF2 = {
        "Sig Strikes Differential": [],
        "Takedowns Differential": [],
        "Sub Attempts Differential": [],
        "Guard Passes Differential": []
    }

    my_data = copy.deepcopy(fightMetricData["Sig Strikes"])
    newDF2["Sig Strikes Differential"].extend(my_data)
    my_data = copy.deepcopy(fightMetricData["Takedowns"])
    newDF2.get("Takedowns Differential").extend(my_data)
    my_data = copy.deepcopy(fightMetricData["Sub Attempts"])
    newDF2.get("Sub Attempts Differential").extend(my_data)
    my_data = copy.deepcopy(fightMetricData["Guard Passes"])
    newDF2.get("Guard Passes Differential").extend(my_data)
    #counter = 0
    #print(fightMetricData[listFightMetrics[0]])
    # for newCol in newDF2:
    #     print(newDF2.get(newCol))
    #     my_data = copy.deepcopy(fightMetricData[listFightMetrics[counter]])   #      np.random.random((210, 8))  # recfromcsv('LIAB.ST.csv', delimiter='\t')
    #     print(newDF2.get(newCol))
    #     newDF2.get(newCol).extend(my_data)
    #     counter += 1

    counter = 0
    newDF3 = {
        "Sig Strikes Rate Differential": [],
        "Takedowns Rate Differential": [],
        "Sub Attempts Rate Differential": [],
        "Guard Passes Rate Differential": []
    }

    newDF4 = {
        "Fight Stoppage": [],
        "KO/TKO": [],
        "Submission": [],
        "Win Type": []
    }
    for newCol in newDF4:
        my_data = copy.deepcopy(fightMetricData.Method)
        newDF4.get(newCol).extend(my_data)


    fightMetricsIndex = 0   # listFightMetrics

    for col in newDF:
        numRows = 0
        tempArray = []
        for tempTime in newDF.get(col):
            if(float(tempTime) > 0.0):
                asdfTime = float(fightMetricData._get_value(numRows, listFightMetrics[fightMetricsIndex]))/float(tempTime)
                tempArray.append(asdfTime)
            else:
                asdfTime = 0.0
                tempArray.append(asdfTime)
            numRows += 1
        newDF[col] = tempArray
        fightMetricData = fightMetricData.assign(tempColName=newDF.get(col))     #fightMetricData.insert(15, col, newDF.get(col), True)
        fightMetricData.rename(columns={'tempColName': col}, inplace=True)
        fightMetricsIndex += 1

    #define df2: differentials
    for col in newDF2:
        numRows = 0
        tempArray = []
        for metric in newDF2.get(col):
            if (numRows%2 == 0):
                asdfMetric = (metric - newDF2.get(col)[numRows+1])# metric = float(fightMetricData._get_value(numRows, listFightMetrics[fightMetricsIndex])) / float(metric)
                tempArray.append(asdfMetric)
            else:
                asdfMetric = (metric - newDF2.get(col)[numRows - 1])
                tempArray.append(asdfMetric)
            numRows += 1
            #print(metric)
        newDF2[col] = tempArray
        fightMetricData = fightMetricData.assign(tempColName=newDF2.get(col))  # fightMetricData.insert(15, col, newDF2.get(col), True)
        fightMetricData.rename(columns={'tempColName': col}, inplace=True)

        fightMetricsIndex += 1

    listFightMetricsRate = ["Sig Strikes Rate", "Takedowns Rate", "Sub Attempts Rate", "Guard Passes Rate"]
    counter = 0
    for newCol in newDF3:
        # print(fightMetricData[str(listFightMetrics[counter])])
        my_data = copy.deepcopy(fightMetricData[str(listFightMetricsRate[counter])])  # np.random.random((210, 8))  # recfromcsv('LIAB.ST.csv', delimiter='\t')
        newDF3.get(newCol).extend(my_data)
        counter += 1

    for col in newDF3:
        numRows = 0
        tempArray = []
        for rateMetric in newDF3.get(col):
            if (numRows%2 == 0):
                asdfMetric = (rateMetric - newDF3.get(col)[numRows+1])
                tempArray.append(asdfMetric)
            else:
                asdfMetric = (rateMetric - newDF3.get(col)[numRows - 1])
                tempArray.append(asdfMetric)
            numRows += 1
        newDF3[col] = tempArray
        fightMetricData = fightMetricData.assign(tempColName=newDF3.get(col))  # fightMetricData.insert(15, col, newDF3.get(col), True)
        fightMetricData.rename(columns={'tempColName': col}, inplace=True)

        fightMetricsIndex += 1

    #define df4: logistic classifiers
    # for col in newDF4:
    numRows = 0
    tempArray_StoppageWin = []
    tempArray_KOTKOWin = []
    tempArray_SubmissionWin = []

    tempArray_WinType = []

    rowNum = 0
    for method in newDF4.get("Fight Stoppage"):     # These count victories by stoppage
        if fightMetricData.Result[rowNum] == 1:
            if 'KO/TKO' in method:  # if fight ended by Knockout or Technical Knockout
                tempArray_StoppageWin.append('1')
                tempArray_KOTKOWin.append('1')
                tempArray_SubmissionWin.append('0')
                tempArray_WinType.append('1')
            elif 'SUB' in method:  # if fight ended by Submission
                tempArray_StoppageWin.append('1')
                tempArray_KOTKOWin.append('0')
                tempArray_SubmissionWin.append('1')
                tempArray_WinType.append('2')
            else:
                if 'U-DEC' in method:
                    tempArray_WinType.append('3')
                elif 'S-DEC' in method:
                    tempArray_WinType.append('4')
                else:
                    tempArray_WinType.append('-1')
                tempArray_StoppageWin.append('0')
                tempArray_KOTKOWin.append('0')
                tempArray_SubmissionWin.append('0')
        else:
            tempArray_StoppageWin.append('0')
            tempArray_KOTKOWin.append('0')
            tempArray_SubmissionWin.append('0')
            tempArray_WinType.append('0')

        rowNum += 1
    newDF4["Fight Stoppage"] = tempArray_StoppageWin
    newDF4["KO/TKO"] = tempArray_KOTKOWin
    newDF4["Submission"] = tempArray_SubmissionWin
    newDF4["Win Type"] = tempArray_WinType

    for col in newDF4:
        fightMetricData = fightMetricData.assign(tempColName=newDF4.get(col))
        fightMetricData.rename(columns={'tempColName': col}, inplace=True)
        print(newDF4.get(col))

    print('Column Generation Breakpoint\n')
    # fightMetricData.append()


def GenerateColumns():
    GenerateFightMetric()


def LinearRegression(dataSet, dataFrame, dependent, labels):
    print('We are predicting: ' + str(dependent) + '\nUsing: ' + '\n' + str(labels))
    X = dataFrame
    Y = np.array(dataSet[dependent])

    # for time in dependent:
    #     print(time)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)
    accuracy = lm.score(x_test, y_test)
    print(accuracy, "\n")
    # print('breakpoint')


def _LogisticRegression(dataSet, dataFrame, dependent, labels):
    print('We are predicting: ' + str(dependent) + '\nUsing: ' + '\n' + str(labels))
    X = dataFrame
    Y = np.array(dataSet[dependent])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    lm = LogisticRegression()
    lm.fit(x_train, y_train)
    accuracy = lm.score(x_test, y_test)
    print(accuracy, "\n")
    # print('breakpoint')


def KNN(dataSet, dataFrame, dependent, labels):
    print('We are predicting: ' + str(dependent) + ' via KNN\nUsing: ' + '\n' + str(labels))
    X = dataFrame#list(zip(dataFrame))
    # print(X)
    Y = np.array(dataSet[dependent])
    # print(Y)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print(accuracy, "\n")


def SVM(dataSet, dataFrame, dependent, labels):
    print('We are predicting: ' + str(dependent) + ' via SVM\nUsing: ' + '\n' + str(labels))
    X = dataFrame#list(zip(dataFrame))
    # print(X)
    Y = np.array(dataSet[dependent])
    # print(Y)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

    tempClassNotation = ("NOT " + str(dependent))
    classes = [str(dependent), tempClassNotation]

    classifier = svm.SVC()
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print(accuracy, "\n")



def bench_k_means(estimator, name, data, y):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

def _KMeans(dataSet, dataFrame, dependent, labels):
    print('We are predicting: ' + str(dependent) + ' via SVM\nUsing: ' + '\n' + str(labels))
    data = scale(dataFrame)# scale(dataFrame.data)#list(zip(dataFrame))

    Y = np.array(dataSet[dependent])

    k = 10
    samples, feature = data.shape
    classifier = KMeans(n_clusters=k, init="random", n_init=10)

    bench_k_means(classifier, (str(dependent) + " via: " + str(labels)), data, Y)



def HelloWorld():
    print('hello world')


def main():
    global fightMetricData
    HelloWorld()
    CleanFrames()
    GenerateColumns()
    #** print(fightMetricData.head())
    #** print(sherdogData.head())
    # print('breakpoint')


    # dependentVar1 = "Time"
    # dependentVar2 = "Fight Number"
    # dependentVar3 = "Weightclass"
    dependentVars = ["Time", "Fight Number", "Weightclass", "Result"]
    dependentVars_Logistic = ["Fight Stoppage","KO/TKO", "Submission"]
    dependentVars_KNN = ["Fight Stoppage","KO/TKO", "Submission", "Win Type", "Round", "Fight Number", "Weightclass", "Result"]
    dependentVars_SVM = ["Fight Stoppage", "KO/TKO", "Submission", "Result"]
    dependentVars_KMeans = ["Result", "Fight Stoppage","KO/TKO", "Submission", "Win Type", "Round"]

    dataFrame = fightMetricData[["Sig Strikes",	"Takedowns", 	"Sub Attempts", "Guard Passes"]]
    dataFrame_Rate = fightMetricData[["Sig Strikes Rate",	"Takedowns Rate", 	"Sub Attempts Rate", "Guard Passes Rate"]]
    dataFrame_Metric_Differential = fightMetricData[["Sig Strikes Differential", "Takedowns Differential", "Sub Attempts Differential", "Guard Passes Differential"]]
    dataFrame_Rate_Differential = fightMetricData[["Sig Strikes Rate Differential", "Takedowns Rate Differential", "Sub Attempts Rate Differential", "Guard Passes Rate Differential"]]

    dataFrame_Labels = ["Sig Strikes",	"Takedowns", 	"Sub Attempts", "Guard Passes"]
    dataFrame_Rate_Labels = ["Sig Strikes Rate", "Takedowns Rate", "Sub Attempts Rate", "Guard Passes Rate"]
    dataFrame_Metric_Differential_Labels = ["Sig Strikes Differential", "Takedowns Differential", "Sub Attempts Differential", "Guard Passes Differential"]
    dataFrame_Rate_Differential_Labels = ["Sig Strikes Rate Differential", "Takedowns Rate Differential", "Sub Attempts Rate Differential", "Guard Passes Rate Differential"]

    #dataFrameMetricsRate = ["Sig Strikes Rate", "Takedowns Rate", "Sub Attempts Rate", "Guard Passes Rate"]

    listOf_Dataframes = [dataFrame, dataFrame_Rate, dataFrame_Metric_Differential, dataFrame_Rate_Differential]
    listOf_LabelLists = [dataFrame_Labels, dataFrame_Rate_Labels, dataFrame_Metric_Differential_Labels, dataFrame_Rate_Differential_Labels]

    dataFrame_Total = []
    dataFrame_Total.extend(dataFrame)
    dataFrame_Total.extend(dataFrame_Rate)
    dataFrame_Total.extend(dataFrame_Metric_Differential)
    dataFrame_Total.extend(dataFrame_Rate_Differential)
    # np.array(fightMetricData.drop('Url', 1))
    # print('breakpoint 2')
    #LinearRegression(fightMetricData, dataFrame, dependentVars, dataFrame_Labels)

    labelIndex = 0

    #** print('DF: ', dataFrame_Rate)
    #** print('Cols:', fightMetricData.columns)


    for dv in dependentVars:
        dfIndex = 0
        for df in listOf_Dataframes:
            labelIndex = 0
            for metric in df:
                #* print("Metric: ", metric)
                LinearRegression(fightMetricData, np.array(fightMetricData[[metric]]), dv, metric)#dataFrame_Labels[labelIndex])
                labelIndex += 1
            LinearRegression(fightMetricData, dataFrame, dv, listOf_LabelLists[dfIndex])#dataFrame_Labels)
            dfIndex += 1

    print("\n\n\n\n\n\n\n\nLog Assessment\n\n\n\n")

    for dv in dependentVars_Logistic:
        dfIndex = 0
        for df in listOf_Dataframes:
            labelIndex = 0
            for metric in df:
                #* print("Metric: ", metric)
                _LogisticRegression(fightMetricData, np.array(fightMetricData[[metric]]), dv, metric)#dataFrame_Labels[labelIndex])
                labelIndex += 1
            _LogisticRegression(fightMetricData, dataFrame, dv, listOf_LabelLists[dfIndex])#dataFrame_Labels)
            dfIndex += 1

    print("\n\n\n\n\n\n\n\nKNN Assessment\n\n\n\n")
    le = preprocessing.LabelEncoder()

    for dv in dependentVars_KNN:
        dfIndex = 0
        for df in listOf_Dataframes:
            labelIndex = 0
            for metric in df:
                #* print("Metric: ", metric)
                KNN(fightMetricData, np.array(fightMetricData[[metric]]), dv, metric)
                labelIndex += 1
            KNN(fightMetricData, dataFrame, dv, listOf_LabelLists[dfIndex])  # dataFrame_Labels)
            dfIndex += 1

    print("\n\n\n\n\n\n\n\nSVM Assessment\n\n\n\n")

    for dv in dependentVars_SVM:
        dfIndex = 0
        for df in listOf_Dataframes:
            labelIndex = 0
            for metric in df:
                #* print("Metric: ", metric)
                SVM(fightMetricData, np.array(fightMetricData[[metric]]), dv, metric)
                labelIndex += 1
            SVM(fightMetricData, dataFrame, dv, listOf_LabelLists[dfIndex])  # dataFrame_Labels)
            dfIndex += 1

    print("\n\n\n\n\n\n\n\nK-Means Assessment\n\n\n\n")

    for dv in dependentVars_SVM:
        dfIndex = 0
        for df in listOf_Dataframes:
            labelIndex = 0
            for metric in df:
                #* print("Metric: ", metric)
                _KMeans(fightMetricData, np.array(fightMetricData[[metric]]), dv, metric)
                labelIndex += 1
            _KMeans(fightMetricData, dataFrame, dv, listOf_LabelLists[dfIndex])  # dataFrame_Labels)
            dfIndex += 1





main()