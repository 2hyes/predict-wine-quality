import sys
import numpy as np
from sklearn import tree, svm, linear_model, neighbors
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import AgglomerativeClustering, KMeans

#예측함수
def prediction (model,input_figures):
    if(model == 1): #dt
        classifier = tree.DecisionTreeClassifier(random_state = 0)

    elif(model == 2): #svm
        classifier = svm.SVC(random_state = 0)

    elif(model == 3): #LR
        classifier = linear_model.LogisticRegression(random_state = 0)

    elif(model == 4): #k-NN
        classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)

    classifier.fit(X,Y)
    predicted_class = classifier.predict(input_figures)

    return predicted_class

#evaluate#
def true_pred(model):
    y_true = Y
    y_pred = prediction(model, X)

    confusionmatrix(model, y_true, y_pred)
    accuracy(model, y_true, y_pred)
    precision(model, y_true, y_pred)
    recall(model, y_true, y_pred)
    fmeasure(model, y_true, y_pred)
    
def confusionmatrix (model, y_true, y_pred):
    print("1. Confusion matrix")
    print(confusion_matrix(y_true, y_pred))

def accuracy(model, y_true, y_pred):
    print("\n2. Accuracy: ")
    print(accuracy_score(y_true, y_pred))

def precision(model, y_true, y_pred):
    print("\n3. Precision: ")
    print(precision_score(y_true, y_pred, average = None))

def recall(model, y_true, y_pred):
    print("\n4. Recall :\n")
    print(recall_score(y_true, y_pred, average = None))
    
def fmeasure(model, y_true, y_pred):
    print("\n5. F-measure :\n")
    print(f1_score(y_true, y_pred, average = None))
    
#cluster#    
def cluster(model, cluster_num):
    if model == 1:
        model = AgglomerativeClustering(n_clusters = cluster_num)
    elif model == 2:
        model = KMeans(n_clusters = cluster_num, random_state=0)
    model.fit(X)
    cluster_inf = model.labels_

    #방법1: 오름차순으로정렬후 카운트
##    cluster_inf.sort() 
##    count = []
##    c = 0
##
##    for i in range(0, len(cluster_inf)-1):
##        c += 1
##        if (cluster_inf[i] != cluster_inf[i+1]):
##            count.append(c)
##            c = 0
##    count.append(c+1)
##
##    for x in range(0,cluster_num):
##        print("Cluster %d: %d" % (x,count[x]))
##    print()

    #방법2: 카운트함수 사용
    for i in range(0,cluster_num):    
        a = np.count_nonzero( cluster_inf == i )
        print("Cluster %d: %d" % (i,a))
    print()
    

while(1):
    print("[ Student ID: 1715237 ]\n[ Name: 이혜승( LEE HYE SEUNG ) ]\n")
    print('''1. Predict wine quality
2. Evaluate wine prediction models
3. Cluster wines
4. Quit\n''')

    choice = int(input())
    if choice == 4:
        print("EXIT")
        sys.exit()

    #data loading
    source = open("C:\winequality-red.csv")

    data = np.genfromtxt(source, dtype=np.float32, delimiter = ";",
                     skip_header = 1, usecols = range(0,12))

    X = data[:, 0:11]
    Y = data[:, 11]

    if choice == 1:
        #input the values
        print("\nInput the values of a wine:")
        
        fixed_acid = float(input("1. fixed acidity: "))
        volatile_acid = float(input("2. volatile acidity: "))
        citric_acid = float(input("3. citric acid: "))
        residual_sugar = float(input("4. residual sugar: "))
        chol = float(input("5. chlorides: "))
        free_sulfur_dio = float(input("6. free sulfur dioxide: "))
        total_sulfur_dio = float(input("7. total sulfur dioxide: "))
        dens = float(input("8. density: "))
        pH = float(input("9. pH: "))
        sul = float(input("10. sulphates: "))
        alc = float(input("11. alcohol: "))

        input_figures = (np.array([fixed_acid, volatile_acid, citric_acid,
                                   residual_sugar,chol, free_sulfur_dio,
                                   total_sulfur_dio,dens, pH, sul, alc])).reshape(1, -1)

        print("Predicted wine quality:\n")
        print("1. Decision tree: %d" % prediction(1, input_figures) )
        print("2. Support vector machine: %d" % prediction(2, input_figures) )
        print("3. Logistic regression: %d" % prediction(3, input_figures) )
        print("4. k-NN clssifier: %d\n" % prediction(4, input_figures) )

    elif choice == 2:
        print("Decision tree:")
        true_pred(1)
        
        print("\nSupport vector machine:")
        true_pred(2)
        
        print("\nLogistic regression:")
        true_pred(3)
   
        print("\nk-NN classifier:")
        true_pred(4)
        print()

    elif choice == 3:
        cluster_num = int(input(("Input the number of clusters: ")))
        print("\nThe number of wines in each cluster:\n")
                                    
        print("<hierarchical clustering>")
        cluster(1,cluster_num)
        print("<k-means clustering>")
        cluster(2,cluster_num)
