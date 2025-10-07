
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import random
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os


table2 = pd.read_excel('i-spy2-tables2.xlsx')
table2.set_index('Patient Identifier', inplace = True)
table2 = table2.iloc[:,:7]
gexp = pd.read_csv('data_gexp4000.csv')
gexp.set_index('Unnamed: 0', inplace = True)
table2 = table2.loc[gexp.index]

arm_short_name = ['Ctr', 'AMG386', 'N', 'Ganitumab', 'MK2206', 'Ganetespib', 'VC', 'Pembro', 'TDM1/P', 'Pertuzumab']
#arm_short_name = ['Ctr', 'VC', 'TDM1/P'] 

# Set the search space of hyperparameters 
param_grids = {
    'n_estimators': [100, 200, 400],
    'max_depth': [2,3,4],
    'learning_rate': [0.01, 0.1, 0.3, 0.5],
    'colsample_bynode': [0.2,0.1,0.06,0.04,0.02],
    'reg_lambda':[0.1,1,10,20]
}


#train and test XGBoost models for each arm 
random_seed = list(range(100,2001,100))
n_repeat = len(random_seed)
for a in arm_short_name:
    table2_1arm = table2.loc[table2['Arm (short name)'] == a]

    for n_run in range(n_repeat):
        rseed = random_seed[n_run] #rseed will be used in 1) train_test_split
        random.seed(rseed)         

        #seperate data into four categories accournding to the HR and HER2 status
        temp1 = table2_1arm.loc[table2_1arm['HR'] == 1]
        temp2 = temp1.loc[temp1['HER2'] == 1]
        pp_pCR = temp2['pCR']

        temp2 = temp1.loc[temp1['HER2'] == 0]
        pn_pCR = temp2['pCR']
    
        temp1 = table2_1arm.loc[table2_1arm['HR'] == 0]
        temp2 = temp1.loc[temp1['HER2'] == 1]
        np_pCR = temp2['pCR']
    
        temp2 = temp1.loc[temp1['HER2'] == 0]
        nn_pCR = temp2['pCR']

        #split data into a train set and a test set, 
        #split four categories into training and testing proportionally 
        d_train = pd.DataFrame({ 'pCR': []},dtype=int)
        d_test = pd.DataFrame({ 'pCR': []},dtype=int)       
        for i in range(4):
            if i == 0:
                data = pp_pCR
            elif i ==1:
                data = pn_pCR
            elif i==2:
                data = np_pCR
            else:
                data = nn_pCR 
            
            if data.shape[0] > 9:
                temp1, temp2 = train_test_split(data, test_size=0.2, shuffle = True, stratify = data.values,  random_state=rseed)
                if d_train.shape[0] == 0:
                    d_train = temp1
                    d_test = temp2
                else:
                    d_train = pd.concat([d_train, temp1])
                    d_test  = pd.concat([d_test,temp2])
       
        X_train_gexp = gexp.loc[d_train.index]
        y_train = d_train.values
        y_train = y_train.astype('int8')
        
        X_test_gexp  = gexp.loc[d_test.index]
        y_test  = d_test.values
        y_test = y_test.astype('int8')
        
        
        hh=1 #hh ==1 add HR and HER2 status as features,  hh=0 do not add them as features
        if hh == 1:
            temp = table2.loc[d_train.index,['HR','HER2']]
            temp.columns = ['HRc', 'HER2c']  #there is a HR gene in gexp data, rename HR here to avoid duplicate
            X_train = pd.concat([X_train_gexp, temp], axis = 1)
            temp = table2.loc[d_test.index,['HR','HER2']]
            temp.columns = ['HRc', 'HER2c']
            X_test  = pd.concat([X_test_gexp, temp], axis = 1)
        else:
            X_train = X_train_gexp
            X_test  = X_test_gexp
        feature_names = X_train.columns.values

        norm = 1 # norm ==1 do featrue normalization, norm ==0 no feature normalization
        if norm == 1:
            sc = StandardScaler()
            sc.fit(X_train)
            X_train.values[:] = sc.transform(X_train)
            X_test.values[:]  = sc.transform(X_test)
        
        #model training, GridSearchCV to search optimal hyper-parameters
        temp = y_train.sum()
        temp = (y_train.shape[0]-temp)/temp
        xgb_clf = XGBClassifier(
            random_state = 100,
            objective = 'binary:logistic',
            eval_metric = 'auc',
            scale_pos_weight = temp
            #nthread = 3,
            #device = 'cuda'
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rseed)
        gs = GridSearchCV(estimator = xgb_clf, param_grid = param_grids, cv=cv, scoring='roc_auc', n_jobs=30,)
        gs.fit(X_train.to_numpy(), y_train)
        best_model = gs.best_estimator_

        #determine optimal cutoff probabililty for prediciting class label using Youden’s J statistic
        #best_params = gs.best_params_
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
        optimal_cutoffs = []
        for tr_idx, val_idx in kf.split(X_train, y_train):
            X_train1, X_val1 = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_train1, y_val1 = y_train[tr_idx], y_train[val_idx]
            clf = clone(best_model).fit(X_train1.to_numpy(), y_train1)
            y_pred_proba = clf.predict_proba(X_val1.to_numpy())

            y_pred_proba=y_pred_proba[:,1]
            fpr, tpr, thr = roc_curve(y_val1, y_pred_proba)
            if len(thr) > 0 and np.isinf(thr[0]):
                fpr = fpr[1:]
                tpr = tpr[1:]
                thr = thr[1:]
            best_idx = np.argmax(tpr - fpr)
            temp  = thr[best_idx]
            optimal_cutoffs.append(temp)
        optimal_thr = np.mean(optimal_cutoffs)

        #save optimal probability threshoold and best model parameters
        if n_run == 0:
            temp = [gs.best_params_['n_estimators'], gs.best_params_['max_depth'], 
                    gs.best_params_['learning_rate'], gs.best_params_['colsample_bynode'], 
                    gs.best_params_['reg_lambda']]
            best_params = [temp]
            optimal_proba_thr = np.array([optimal_thr]) 
        else:
            temp = [gs.best_params_['n_estimators'], gs.best_params_['max_depth'], 
                    gs.best_params_['learning_rate'], gs.best_params_['colsample_bynode'], 
                    gs.best_params_['reg_lambda']]
            best_params.append(temp)
            optimal_proba_thr = np.append(optimal_proba_thr,optimal_thr)

        #model testing
        y_pred_proba = best_model.predict_proba(X_test.to_numpy())
        y_pred_proba1=y_pred_proba[:,1]

        #save confusion matrix
        y_pred = (y_pred_proba1 >= optimal_thr).astype(int) 
        cm = confusion_matrix(y_test, y_pred)
        if n_run == 0:
            cfm = cm
        else:
            cfm = np.concatenate((cfm,cm),axis = 0)

        #save confusion matrix for four categories seperately based on HR and HER2 status    
        phh = table2_1arm.loc[d_test.index.values,['pCR','HR','HER2']]
        df_ypred = pd.DataFrame(y_pred,index = phh.index.values) 

        cm =  np.zeros((2, 8), dtype=int)
        for hr in [0,1]:
            temp1 = phh[phh['HR']==hr]
            for her2 in [0,1]:
                temp2 = temp1[temp1['HER2']==her2]
                if temp2.shape[0] != 0:
                    y=temp2['pCR'].values
                    y_pred = df_ypred.loc[temp2.index.values,0].values
                    c = confusion_matrix(y, y_pred)
                    i = (hr*2+her2)*2
                    cm[:,i:i+2] = c
        if n_run == 0:
            cfm_hh = cm
        else:
            cfm_hh = np.concatenate((cfm_hh,cm),axis = 0)

        #save ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba1)
        temp = np.concatenate((np.reshape(fpr,(1,-1)), np.reshape(tpr,(1,-1)), np.reshape(thresholds,(1,-1))),axis = 0)
        #roc_auc = auc(fpr, tpr)
        if n_run == 0:
            ftpr = temp.T
        else:
            ftpr = np.concatenate((ftpr, np.array([100,100,100]).reshape(1,-1)), axis = 0)
            ftpr = np.concatenate((ftpr,temp.T), axis = 0 )
            
        #save PRC
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba1)
        th = np.reshape(thresholds,(1,-1))
        th = np.append(th,[1])
        th = np.reshape(th, (1,-1))
        temp = np.concatenate((np.reshape(precision,(1,-1)), np.reshape(recall,(1,-1)), th),axis = 0)
        if n_run == 0:
            precr = temp.T
        else:
            precr = np.concatenate((precr, np.array([100,100,100]).reshape(1,-1)), axis = 0)
            precr = np.concatenate((precr,temp.T), axis = 0 )
            
        #save true and predicted labels of the test data
        if n_run == 0:
            y_true_pred = np.concatenate((np.reshape(y_test,(-1,1)), 
                                      np.reshape(y_pred_proba1,(-1,1))),axis = 1) 
        else:
            temp = np.concatenate((np.reshape(y_test,(-1,1)), 
                               np.reshape(y_pred_proba1,(-1,1))),axis = 1) 
            y_true_pred = np.concatenate((y_true_pred,temp), axis = 1 )
            

           
        #determine feature importaince 
        importance = pd.DataFrame({"fimportance": best_model.feature_importances_},
                                      index=feature_names)
            
        importance.sort_values(by='fimportance', ascending = False, inplace = True)
        importance['rank'] = list(range(importance.index.shape[0]))  
        if n_run == 0:
            feature_rank = pd.DataFrame([0]*X_train.columns.shape[0],index=feature_names)
        for i in feature_rank.index:
            feature_rank.loc[i,n_run] = importance.loc[i,'rank']        


        # save the running status in a file for monitoring
        filename = 'run_xgb35_hh' + str(hh) + 'norm' +str(norm) +'.csv'
        temp = pd.DataFrame([a, n_run])
        temp.to_csv(filename, index = False, header = False)

        #print(a)
        #print(n_run)

    #save results of each arm to files
    if a == 'TDM1/P':
        b = 'TDM1P'
    else:
        b = a

    file_directory  = 'res_xgb41_hhcmf_hh' + str(hh) + 'norm' +str(norm)  
    try:
        os.mkdir(file_directory)
    except FileExistsError:
        pass

    fprefix = file_directory + '/results_xgb35_hh' + str(hh) + 'norm' +str(norm) + '_' + b

    feature_rank['rank'] = feature_rank.mean(axis=1)
    feature_rank['std'] = feature_rank.std(axis=1)
    feature_rank.sort_values(by='rank', inplace = True)    
    filename = fprefix + '_frank.csv'
    feature_rank.to_csv(filename) 

    filename =  fprefix + '_cfm.csv'
    np.savetxt(filename, cfm.astype(int), fmt='%i', delimiter = ',')

    filename =  fprefix + '_cfm_hh.csv'
    np.savetxt(filename, cfm_hh.astype(int), fmt='%i', delimiter = ',')
    
    filename = fprefix + '_ftpr.csv'
    np.savetxt(filename, ftpr, fmt='%.3e', delimiter = ',')
    
    filename = fprefix + '_precr.csv'
    np.savetxt(filename, precr, fmt='%.3e', delimiter = ',')

    filename = fprefix + '_yproba.csv'
    np.savetxt(filename, y_true_pred, fmt='%.4e', delimiter = ',')

    temp = pd.DataFrame(best_params, columns = ['n_estimators', 'max_depth', 'learning_rate', 
                                                'colsample_bynode','reg_lambda'])
    filename = fprefix + '_best_params.csv'
    temp.to_csv(filename)
    
    filename = fprefix + '_optimal_proba_thr.csv'
    np.savetxt(filename, optimal_proba_thr, delimiter=',', fmt='%f')


    filename =  fprefix + '_RandomSeed.csv'
    np.savetxt(filename, np.array(random_seed), fmt='%i', delimiter = ',')

