import numpy as np
import pandas as pd
import sys    
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
    
OUTPUT_TEMPLATE = (
    'Bayesian classifier:               {bayes:.3f}  {bayes_convert:.3f}\n'
    'kNN classifier:                    {knn:.3f}  {knn_convert:.3f}\n'
    'Rand forest classifier:            {rf:.3f}  {rf_convert:.3f}\n'
    'Decision tree classifier:          {dt_t:.3f}  {dt_v:.3f}\n' 
    'Gradient boosting classsifier      {gb_t:.3f}  {gb_v:.3f}\n'  
    'Support-vector machine classifier  {scv_t:.3f} {scv_v:.3f}\n' 
)
    
"""
Reads the labelled data and trains and validates a machine learning model for the best possible results.
It then predicts the cities where the unlabelled 2016 weather came from.
"""
def main():
    # read the labelled data and train and validate a machine learning model
    labelled_data = pd.read_csv(sys.argv[1])
    y = labelled_data['city']
    X = labelled_data.drop(['city','year'], axis=1)  
    # X = StandardScaler().fit(X).transform(X) # normalize the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    # create some models
    bayes_model = GaussianNB()
    bayes_convert_model = make_pipeline(StandardScaler(), GaussianNB())
    knn_model = KNeighborsClassifier(n_neighbors=20)
    knn_convert_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=20))
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20)
    rf_convert_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20))
    dt_model = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=15))
    gb_model = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=10, max_depth=2, min_samples_leaf=0.1))
    scv_model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))

    # train each model
    models = [bayes_model,bayes_convert_model,knn_model,knn_convert_model,rf_model,rf_convert_model,dt_model,gb_model,scv_model]
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
    
    # print("kNN score:", knn_convert_model.score(X_valid, y_valid))
    
    # output results
    # print(OUTPUT_TEMPLATE.format(
    #     bayes=bayes_model.score(X_valid, y_valid),
    #     bayes_convert=bayes_convert_model.score(X_valid, y_valid),
    #     knn=knn_model.score(X_valid, y_valid),
    #     knn_convert=knn_convert_model.score(X_valid, y_valid),
    #     rf=rf_model.score(X_valid, y_valid),
    #     rf_convert=rf_convert_model.score(X_valid, y_valid),
    #     dt_t = dt_model.score(X_train, y_train),
    #     dt_v = dt_model.score(X_valid, y_valid),
    #     gb_t = gb_model.score(X_train, y_train),
    #     gb_v = gb_model.score(X_valid, y_valid),
    #     scv_t = scv_model.score(X_train, y_train),
    #     scv_v = scv_model.score(X_valid, y_valid)
    # ))
    
    # Predict the cities where the unlabelled 2016 weather came from
    # From some testing, SCV seems to work okay, so let's use that
    unlabelled_data = pd.read_csv(sys.argv[2])
    X_unlabelled = unlabelled_data.drop(['city','year'], axis=1)  
    predictions = scv_model.predict(X_unlabelled)
    pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)
    
    # Print the score
    print('SVC score: ', scv_model.score(X_valid, y_valid))
    
    # Check where the weather model makes the wrong prediciton
    # df = pd.DataFrame({'truth': y_valid, 'prediction': scv_model.predict(X_valid)})
    # print(df[df['truth'] != df['prediction']])          
    
if __name__ == '__main__':
    main()

    