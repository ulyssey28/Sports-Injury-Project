

def train_test(df, columns):
    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.model_selection import train_test_split, GridSearchCV

    from sklearn.ensemble import RandomForestClassifier
   
    
    import numpy as np
    y = np.array(df['Injury Status_Injured'])
    
    X = df[columns]
    New_X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(New_X, y, train_size=0.75)
    
    return X_train, X_test, y_train, y_test
    
    
    
def logistic_model(X_tn, X_tt, y_tn, y_tt):
    
    X_train = X_tn 
    X_test = X_tt
    y_train = y_tn
    y_test = y_tt 
    
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    print(classifier)
    print('---------------------------------')
    
    
    classifier.fit(X_train, y_train)
    
    print('X_test Prediction: ')
    print('---------------------------------')
    
    print(classifier.predict(X_test))
    print('---------------------------------')
    
    print(f"Training Data Score: {classifier.score(X_train, y_train)}")
    print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_test, y_test)
    
    return train_score, test_score 
    
    
    
    
    
def knn_model(X_tn, X_tt, y_tn, y_tt, k_value=41):
    
    X_train = X_tn 
    X_test = X_tt
    y_train = y_tn
    y_test = y_tt 
    
    import matplotlib.pyplot as plt
    import math
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    
    
    X_scaler = StandardScaler().fit(X_train.reshape(-1, 1))
    
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Loop through different k values to see which has the highest accuracy
    # Note: We only use odd numbers because we don't want any ties
    train_scores = []
    test_scores = []
    for k in range(1, 60, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_score = knn.score(X_train_scaled, y_train)
        test_score = knn.score(X_test_scaled, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")


    plt.plot(range(1, 60, 2), train_scores, marker='o')
    plt.plot(range(1, 60, 2), test_scores, marker="x")
    plt.xlabel("k neighbors")
    plt.ylabel("Testing accuracy Score")
    plt.show()
    
    print('------------------------------------')
    
    train_scores = []
    test_scores = []

    k = int( math.sqrt(len(X_train)))

    knn = KNeighborsClassifier(n_neighbors= k_value)
    knn.fit(X_train_scaled, y_train)
    train_score = knn.score(X_train_scaled, y_train)
    test_score = knn.score(X_test_scaled, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"k: {k_value}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
    
    y_pred = knn.predict(X_test)
    
    print('-------------------------------------')
    print('Confusion Matrix: ')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
          
    print('-------------------------------------')  
          
    print('F1 Score: ')
    print(f1_score(y_test, y_pred))
    print('-------------------------------------')
    
    print('Accuracy Score: ')
    print(accuracy_score(y_test, y_pred))
    
    
def random_for(X_tn, X_tt, y_tn, y_tt, df):
    import pandas as pd
    X_train = X_tn 
    X_test = X_tt
    y_train = y_tn
    y_test = y_tt 
    
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(X_train, y_train.ravel())
    
    preds=clf.predict(X_test)
    print(f'Number of Predictions: {len(preds)}')
    
    print('-------------------------------------')
    
    #Print summary information from running prediction on X_test set
    newdf = pd.DataFrame(X_test)
    newdf['predicted']=preds
    
    odf=df.loc[newdf.index]
    odf['predicted']=preds
    print("Predicted as injured:")
    print(newdf.loc[newdf.predicted==1].shape)
    print("Predicted as not injured")
    print(newdf.loc[newdf.predicted==0].shape)
    
    