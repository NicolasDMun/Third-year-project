import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer

#imputer libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#SMOTE library
from imblearn.over_sampling import SVMSMOTE
from imblearn.metrics import specificity_score, sensitivity_score 


#, sensitivity_specificity_support

#cross validation, random forest and F1 scoring libraries
from sklearn.model_selection import cross_val_score, GridSearchCV,KFold, train_test_split
from sklearn.metrics import f1_score, make_scorer, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv(r"C:\Users\Izahe\OneDrive\Desktop\Year 3\Third Year Project\Code\Reproducible paper 1\dataset paper 1 not processed.csv", encoding = 'utf8')

#delete rows where there is 0 data:
X = pd.DataFrame(df.loc[:,df.columns != 'SARS-Cov-2 exam result'].values)
Y = df.loc[:,'SARS-Cov-2 exam result']
mystery_indices = X.index[X.isna().all(axis = 1)]
Y.drop(index=mystery_indices, inplace=True)
X.drop(index=mystery_indices, inplace=True)

print(Y.shape)
print(X.shape)



#define diagnosis through binary encoder
binary_exam_result = LabelBinarizer()
binary_exam_result.fit_transform(Y)
Y = binary_exam_result.fit_transform(Y)
print('Binary encoder   check')

nan_indices = np.where(np.isnan(X.iloc[:,2]))

#perform iterative imputation
imputer = IterativeImputer()
imputed = imputer.fit_transform(X)
X = pd.DataFrame(imputed, columns=X.columns)
print('Iterative imputation   check')


#set SVM-SMOTE parameters (k = 5)
sm = SVMSMOTE(sampling_strategy = 'auto',random_state=None ,k_neighbors=5)
x_res,y_res = sm.fit_resample(X,Y)
print('SVM-SMOTE   check')


#performing nested cross validation:
def cross_validation():

    #def hyperparameters
    hyperparameter_grid = {'n_estimators' : [10,20,30,45,50,55,60,65,70,75,80,85,90,95,100], 'max_depth' : [2,4,8,16,32,64]}

    # def inner_loop
    inner_loop = KFold(n_splits = 5, shuffle = True , random_state = 0)
    
    # def outer_loop
    outer_loop = KFold(n_splits = 5, shuffle = True , random_state = 0)

    #initialize f1 as a scoring metric
    f1_scorer = make_scorer(f1_score, average = 'macro')
    
    #initialize RFC model, run cross-validation and fit to balanced dataset
    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(rfc, hyperparameter_grid,cv=inner_loop,scoring=f1_scorer)
    grid_search.fit(x_res,y_res)
    nested_scores = cross_val_score(grid_search, x_res,y_res, cv = outer_loop, scoring = f1_scorer)
    
    #find the best hyperparameters for the model
    best_estimator = grid_search.best_params_

    print('nested_scores   check')
    print("f1 score: (at %0.2f) (+/- %0.2f)" % (nested_scores.mean(), nested_scores.std()*2))
    return best_estimator

def repeat_train (best_estimator):

    #best_parameters = {'best_estimator': 85,'best_depth' : 16}

    accuracy = []
    f1_result = []
    sensitivity = []
    specificity = []
    AUROC = []


    #retrain best model with hyperparameters
    rfc = RandomForestClassifier(n_estimators = best_estimator['n_estimators'],max_depth = best_estimator['max_depth'])
    
    #fit for 10 iterations
    for i in range(10):
        x_testing,x_training,y_testing, y_training = train_test_split(x_res,y_res,test_size = 0.2, random_state = i)
        rfc.fit(x_training, y_training)
    
        y_pred = rfc.predict(x_testing)
        
        #scoring
        accuracy.append(accuracy_score(y_testing, y_pred))
        f1_result.append(f1_score(y_testing, y_pred, average = 'macro'))
        sensitivity.append(sensitivity_score(y_testing, y_pred))
        specificity.append(specificity_score(y_testing, y_pred))
        AUROC.append(balanced_accuracy_score(y_testing, y_pred))


    print('Accuracy: %f (+/- %0.2f) '%(np.mean(accuracy),np.std(accuracy)*2))
    print('F1-score: %f (+/- %0.2f) '%(np.mean(f1_result),np.std(f1_result)*2))
    print('Sensitivity: %f (+/- %0.2f) '%(np.mean(sensitivity),np.std(sensitivity)*2))
    print('Specificity: %f (+/- %0.2f) '%(np.mean(specificity),np.std(specificity)*2))
    print('AUROC: %f (+/- %0.2f) '%(np.mean(AUROC),np.std(AUROC)*2))


repeat_train(cross_validation())