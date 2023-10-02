#!/usr/bin/env python
# coding: utf-8

# In[1]:


#sklearn.utils.class_weight.compute_class_weight(class_weight, classes, y)
##本程式之Logistic regression自變數使用標準化處理(可改程式 std = StandardScaler())
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import random
import datetime
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold,learning_curve,validation_curve,GridSearchCV
import joblib
from sklearn.metrics import classification_report,confusion_matrix,roc_curve, auc,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
#from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import calibration_curve, CalibrationDisplay,CalibratedClassifierCV
import time
#!pip install imblearn
#!pip install eli5
#import eli5
#from eli5.sklearn import PermutationImportance
from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#重要參數！outcome labels加權
IF_MM_SMOTE =True #ML是否需Smote 處理不平衡資料？

IF_NN_SMOTE =False #NN是否需Smote 處理不平衡資料？
IF_HIGH_DIM =False #是否模型隱藏層需高維度(1024 neuron神經元且多一層隱藏層)？
IF_ML_WEIGHT=False # 是否 傳統 ML要加權？
IF_NN_WEIGHT = True # 是否 神經網路法NN要加權？
IF_DYNAMIC_WEIGHT = True # 是否 NN加權要採用動態計算而來？CalculateClassWeights(n_neg, n_pos)
IF_LOG_NORMALIZED = False # 羅吉斯回歸X要不要先標準化？ 有標準化 feature importance比較正確，但實做很難，不能直接輸入預測
data_path = 'C:\\Users\\1PC072073\\OneDrive\\AI\\CMH\\專案\\AI-沈郁婷\\experiment\\data\\'
plot_path ='C:\\Users\\1PC072073\\OneDrive\\AI\\CMH\\專案\\AI-沈郁婷\\experiment\\plot\\'
pkl_path = 'C:\\Users\\1PC072073\\OneDrive\\AI\\CMH\\專案\\AI-沈郁婷\\experiment\\pkl\\'
performance_path = 'C:\\Users\\1PC072073\\OneDrive\\AI\\CMH\\專案\\AI-沈郁婷\\experiment\\performance\\'
DATA_FILE=  data_path + 'data_train_40.csv'
OUTFILE_HEADER='CMHAI030_'
MLP_DROP_RATE = 0.2
NUM_EPOCHS=200 #200, 不要太大
BATCH_SIZE=128 #1024愈高愈好? 越高-->train_history後面越不會亂跳(loss) Mortality:200

#SHOW_HISTORY_TRAIN_ACC="acc" # Fred's home, 小主機
#SHOW_HISTORY_VALIDATE_ACC="val_acc" # Fred's home PC, 小主機
SHOW_HISTORY_TRAIN_ACC="accuracy" # CMH PC
SHOW_HISTORY_VALIDATE_ACC="val_accuracy" # CMH PC

# Algorithms=['LR','RF','SVM','MLPClassifier', 'XGBoost','voting' ,'Stacking']
Algorithms=['LR'] 


FirstHiddenLayerDim =64
#HiddenLayerDim =[512, 512, 128]
HiddenLayerDim =[128, 64, 32]

FirstHiddenLayerDim1024 =1024
HiddenLayerDim1024 =[1024, 512, 128, 64]
OutputLayerDim=2

Test_split_test_Size=0.3 # 切第三塊來測試驗證,即: X_test, y_test。第三塊測試可省略但程式需另寫，目前回報之模型指標是這個dataset的結果...
Validate_split_test_Size=0.3 # MLP 內定training/validation 比率，mlp.fit

Full_features = ['age', 'Gender','BMI', 'live alone', 'Smoking history','family history_COPD ',
                 'SBP', 'DBP', 'HR', 'SpO2','RR', 
                 'CAT score', 'mMRC score', 'Episode of AECOPD',
                 'Asthma', 'TB','Hypertension', 'Diabetes', 'CVD', 'CLD', 
                 'Pre-BD-FEV1(L)', 'Post-BD-FEV1(L)', 'Post-BD-FEV1/FVC(%)',
                 'LABA', 'LAMA', 'ICS']

Fe = ['age', 'Gender','BMI', 'live alone', 'Smoking history','family history_COPD ',
                 'SBP', 'DBP', 'HR', 'SpO2','RR', 
                 'CAT score', 'mMRC score', 'Episode of AECOPD',
                 'Asthma', 'TB','Hypertension', 'Diabetes', 'CVD', 'CLD', 
                 'Pre-BD-FEV1(L)', 'Post-BD-FEV1(L)', 'Post-BD-FEV1/FVC(%)',
                 'LABA', 'LAMA', 'ICS','AECOPD within 3 months']

# Fe1 = ['age', 'Gender','BMI', 'live alone', 'Smoking history','family history_COPD ',
#                  'SBP', 'DBP', 'HR', 'SpO2','RR', 
#                  'CAT score', 'mMRC score', 'Episode of AECOPD',
#                  'Asthma', 'TB','Hypertension', 'Diabetes', 'CVD', 'CLD', 
#                  'Pre-BD-FEV1(L)', 'Post-BD-FEV1(L)', 'Post-BD-FEV1/FVC(%)',
#                  'LABA', 'LAMA', 'ICS']
#outcome類別加權 class_weight = {0:0.4, 1: 1.6} 
#Class_weight = {0:0.4, 1: 1.4}  # 死亡未併入 sepsis, RF呼吸衰竭
#Class_weight = {0:0.4, 1: 1.4} # 死亡併入 sepsis, RF呼吸衰竭


# Full_features =['BISAP']
              
# NUM_FEATURE=40
#第一版 Full_features = ['Age', 'Sex', 'BMI', 'Smoking','x_ASA','CVP_A_line', 'DM', 'HT', 'liver','kidney', 'cancer','Dementia','Schizophrenia','stroke','COPD', 'Respiratory']
# features_ASA = ['x_ASA']
#NUM_FEATURE=14
#Outcome= ['死亡	術後新疾病_AMI	術後新疾病_Sepsis 敗血症	術後註記_ICU_Admission	術後註記_On_Ventilators	術後註記_prolonged_hospital_stay0	prolonged_hospital_stay	出院疾病_DM 糖尿病	出院疾病_Critical_limb_ischemia 截肢/嚴重肢體缺血	出院疾病_Hypertension 高血壓	出院疾病_Congestive_heart_failure 缺血性心衰竭	出院疾病_Ischemic_heart_disease 冠狀動脈疾病	出院疾病_Valvular_heart_disease 心瓣膜	出院疾病_Atrial_fibrillation '心房顫動	出院疾病_Cirrhosis_of_liver 肝硬化	出院疾病_Renal_failure_on_dialysis 透析腎功能衰竭	出院疾病_Malignancy 惡性腫瘤	出院疾病_Dementia 失智	出院疾病_Schizophrenia 精神分裂	出院疾病_Stroke 中風	出院疾病_Ischemic 缺血	出院疾病_Hemorrhagic 出血	出院疾病_Sepsis (infection + organ failure) 敗血症	出院疾病_COPD 慢性阻塞性肺病	出院疾病_Asthma 哮喘	出院疾病_Respiratory_failure 呼吸衰竭	出院疾病_SLE 紅斑性狼瘡	出院疾病_Rheumatic_arthritis 類風濕關節炎	出院疾病_Intracapsular 開放性	出院疾病_Trochanteric	出院疾病_Intertrochanteric	出院疾病_Subtrochanteric	出院疾病_AMI 急性心肌梗死	Hosp Stay時間(天數)	Hosp Stay_grade	麻醉總時間	恢復室總院時間	CVP_ALINE	術中出血量']


# In[3]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import binarize 
#https://github.com/cmsalgado/workshops/blob/master/ML_taipei.ipynb?fbclid=IwAR0DYmilOfxBztxayE-2RIqcFwaooLAqELidNW37Ehi4xYiMV7o1q_ucP-I
#顯示最佳threshold(sensitivity 與 specificity最相近); 由給定之threshold往下遞減0.0001, 直到sensitivity>=給定值。並回傳新的：threshold、AUC、sensitivity、specificity
def BestThresholdPerformance(y, y_pred_proba, print_ = 1, *args):   
    """ Calculate performance measures for a given ground truth classification y and predicted 
    probabilities y_pred_proba. If *args is provided a predifined threshold is used to calculate the performance.
    If not, the threshold giving the best mean sensitivity and specificity is selected. The AUC is calculated
    for a range of thresholds using the metrics package from sklearn.
    ---輸入參數說明---
    y為真正結果(0,1)
    y_pred_proba是據測結果(機率)，是單維array, 呼叫此函式前，y_pred_proba可能要先裁定陣列維度(取陽性的機率)，ex: y_pred_rnd_proba[:, 1]。
    print_：是否印出AUC圖
    args[0]：起始thresdhold，每次遞減0.0001直到滿足sensitivity>=args[1]
    args[1]自訂敏感度最小值
    ---輸出(return)說明---
    
    使用範例：
    New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_log_proba[:, 1],print_ = 1)# print_ = 1 表示會印出AUC圖
    New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_log_proba[:, 1], 1)# 同上
    New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_log_proba[:, 1],1, 0.4, 0.8) # 1表會印出AUC圖，threshold由0.4往下找到符合sensitivity= 0.8之門檻值之各結果
    
    """
    
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_proba, pos_label=1) # y_pred_proba is probability, not class  
    # area under the ROC curve
    AUC = metrics.auc(fpr, tpr) #TPR又稱為敏感度(Sensitivity), FPR偽陽性率 = 1-特異度, 特異度= 1-FPR   
    difference = abs(tpr-(1-fpr)) # tpr = sensitivity
    best_threshold = thresholds[difference.argmin()]     
    
    y_pred_class=np.empty((len(y_pred_proba),1))    
    for i in range(len(y_pred_proba)):
        if y_pred_proba[i]>= best_threshold:
            y_pred_class[i] =1
        else:
            y_pred_class[i] =0
               
    tn, fp, fn, tp = confusion_matrix(y, y_pred_class).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print('performance Threshold(best): ' + str(round(best_threshold,4)))
    print('performance TP(best): ' + str(tp))
    print('performance TN(best): ' + str(tn))
    print('performance FP(best): ' + str(fp))
    print('performance FN(best): ' + str(fn))
    print("performance Accuracy(best): " + str( round(accuracy, 3)))
    print('performance Sensitivity(best): ' + str(round(sensitivity,3)))
    print('performance Specificity(best): ' + str(round(specificity,3)))
    print('performance PPV (best):%.3f' % (tp / (tp+fp)))
    print('performance NPV (best):%.3f' % (tn / (fn+tn)))
    print('performance LR+ (best):%.3f' % (tp/((tp+fn) / (fp/(fp+tn))))) 
    print('performance LR- (best):%.3f' % (fn/((tp+fn) / (tn/(fp+tn)))))
    print('performance AUC: ' + str(round(AUC,3)))    
    print("===========自動算出最佳threshold(敏感度、特異度最相近處)值：",best_threshold," >=該值視為1, 否則視為0")    
    
    expected_sensi = 0.8
    if args:
        threshold = args[0]
        if args[1]: expected_sensi = args[1]        
    else:
        threshold = best_threshold
        # we will choose the threshold that gives the best balance between sensitivity and specificity       
    print("===========給定threshold開始值",round(threshold,4),"，遞減0.0001 ...，預期敏感度>=",expected_sensi)
    
    AUC08 = False
    while not AUC08 :
        y_pred_class=np.empty((len(y_pred_proba),1))    
        for i in range(len(y_pred_proba)):
            if y_pred_proba[i]>= threshold:
                y_pred_class[i] =1
            else:
                y_pred_class[i] =0

        tn, fp, fn, tp = confusion_matrix(y, y_pred_class).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if round(sensitivity, 3)>=expected_sensi:
            AUC08 = True
        else:
            threshold = threshold - 0.0001
            if threshold > 1:
                AUC08 = True
            #else:
                #print("Threshold ==== ", threshold)
                #print("AUC08 ==== ", AUC08)
    # print the performance and plot the ROC curve    
    if print_ == 1:
        print('performance Threshold: ' + str(round(threshold,4)))
        print('performance TP: ' + str(tp))
        print('performance TN: ' + str(tn))
        print('performance FP: ' + str(fp))
        print('performance FN: ' + str(fn))
        print("performance Accuracy: " + str( round(accuracy, 3)))
        print('performance Sensitivity: ' + str(round(sensitivity,3)))
        print('performance Specificity: ' + str(round(specificity,3)))
        print('performance PPV :%.3f' % (tp / (tp+fp)))
        print('performance NPV :%.3f' % (tn / (fn+tn)))
        print('performance LR+ :%.3f' % (tp/((tp+fn) / (fp/(fp+tn))))) 
        print('performance LR- :%.3f' % (fn/((tp+fn) / (tn/(fp+tn)))))
        print('performance AUC: ' + str(round(AUC,3)))
    
        plt.figure(figsize = (4,3))
        plt.scatter(x = fpr, y = tpr, label = None)
        plt.plot(fpr, tpr, label = 'Classifier', zorder = 1)
        plt.plot([0, 1], [0, 1], 'k--', label = 'Random classifier')
        plt.scatter(x = 1 - specificity, y = sensitivity, c = 'black', label = 'Operating point', zorder = 2)
        plt.legend()
        plt.xlabel('1 - specificity')
        plt.ylabel('sensitivity')
        plt.show()        
    return round(threshold,4), round(AUC,3), round(sensitivity,3), round(specificity,3)


# In[4]:


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
def CalculateClassWeights(n_neg, n_pos):
    total = n_neg + n_pos
    weight_for_0 = (1 / n_neg)*(total)/2.0 
    weight_for_1 = (1 / n_pos)*(total)/2.0
    class_weight = {0: round(weight_for_0,3), 1: round(weight_for_1,3)}
    return class_weight


# In[5]:


from sklearn.preprocessing import Normalizer
def PlotFeatureImportantRnd(base_fea,algorithm_name, clf, X, y, outcome, save_plot):
    #--------------變數重要性判斷---------------------- 
    print("<<Feature importance of ",algorithm_name,">>")
    
    importance_vals = clf.feature_importances_
    #print("importance rnd_clf=",importance_vals)
    #劃標準差
   # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
   #         axis=0)
    indices = np.argsort(importance_vals)
    #indices = indices[::-1]
    sorted_names = [base_fea[i] for i in indices]
    # Plot the feature importances of the forest
    
    plt.figure(figsize=(10,10))
   # plt.grid(b=True, which='major', color='#666666', linestyle='-')
#    plt.title('Random forest-Feature importance'+'('+ outcome +')')
    plt.title('Random forest-Feature importance')
    #plt.bar(range(X.shape[1]), importance_vals[indices],yerr=std[indices], align="center")
    plt.barh(range(X.shape[1]), importance_vals[indices], align="center",height=0.5)
    plt.yticks(range(X.shape[1]), sorted_names)
    plt.ylim([-1, X.shape[1]])
    plt.xticks(np.arange(0.00, 0.26, 0.05))
    #plt.xlabel(base_features14)
#     maxY = round(importance_vals[indices[0]],3)*1.11
#     plt.xlim([0, maxY])
    plt.ylabel('Features')
    plt.xlabel('Feature Important')
    ###importance 值
    for x, y in enumerate(importance_vals[indices]):
        plt.text(y+.0055, x-.15,'{:.3f}'.format(y),ha='center') # '%.2f'% y  
    
    
    if save_plot==True:
        plt.savefig(plot_path + 'RF-Feature importance'+'('+ outcome +')'+'.png', bbox_inches = 'tight',dpi=300)


# In[6]:


def PlotFeatureImportantLog(base_fea,algorithm_name, clf, X, y,outcome,save_plot):
    #--------------變數重要性判斷---------------------- 
    print("<<Feature importance (coefficients) of ",algorithm_name,">>")
    
#     coef=clf.coef_
    coef=clf.steps[1][1].coef_
    importance_vals0 = coef
   # print("original coef:", coef)
    importance_vals=abs(importance_vals0)
    indices0 = np.argsort(importance_vals) #遞減
    indices = indices0[0] #[::-1]
    importance_vals =importance_vals[0]
    sorted_names = [base_fea[i] for i in indices]
#     plt.figure(figsize=(10,10))
#     plt.title(algorithm_name  + ' feature importance(coefficients)'+'('+outcome+')')
#     plt.bar(range(X.shape[1]), importance_vals[indices], align="center")
#     plt.xticks(range(X.shape[1]), sorted_names, rotation=90)
#     plt.xlim([-1, X.shape[1]])
#     maxY = importance_vals[indices[0]]*1.1
#     plt.ylim([0, maxY])
#     plt.show()    
    
    plt.figure(figsize=(10,10))
#     plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.title('Logistic regression-Feature importance'+'('+ outcome +')')
    #plt.bar(range(X.shape[1]), importance_vals[indices],yerr=std[indices], align="center")
    plt.barh(range(X.shape[1]), importance_vals[indices], align="center",height=0.5)
    plt.yticks(range(X.shape[1]), sorted_names)
    plt.ylim([-1, X.shape[1]])
    plt.xticks(np.arange(0.0, 1.05, 0.1))
    #plt.xlabel(base_features14)
#     maxY = round(importance_vals[indices[0]],3)*1.11
#     plt.xlim([0, maxY])
    plt.ylabel('Features')
    plt.xlabel('Feature Important')
    for x, y in enumerate(importance_vals[indices]):
        plt.text(y+.035, x-.05,'{:.3f}'.format(y),ha='center') # '%.2f'% y 
    
    for f in range(X.shape[1]):
        print("%d. feature %d (%s) = (%f)" % (f + 1, indices[f], base_fea[indices[f]],importance_vals[indices[f]]))
            #1. feature 16 (x_malignancy) = (1.304678)
    if save_plot==True:
        plt.savefig(plot_path + 'LR-Feature importance'+'('+ outcome +')'+'.png', bbox_inches = 'tight',dpi=300)


# In[7]:


def PlotFeatureImportantLog2(base_fea,algorithm_name, clf, X, y):
    #--------------變數重要性判斷---------------------- 
    print("<<Feature importance (coefficients) of ",algorithm_name,">>")
    from sklearn.feature_selection import RFE,SelectFromModel
    #selector = SelectFromModel(estimator=clf).fit(X, y)
    selector1 = RFE(clf, n_features_to_select=1)
    selector1 = selector1.fit(X,y)
    ranking=selector1.ranking_ -1
    print("RFE: ranking=",ranking)
    feature_ranks=[]
    for i in ranking:
        feature_ranks.append(f"{i}. {base_fea[i]}")
    print("RFE: ranking feature=",feature_ranks)
    
    selector2 = SelectFromModel(clf, threshold=-np.inf, max_features=23)# 跟我原來寫的一樣PlotFeatureImportantLog
    selector2.fit(X,y)     
    coef=selector2.estimator_.coef_
    importance_vals0 = coef
    print("original coef:", coef)
    importance_vals=abs(importance_vals0)
    indices0 = np.argsort(importance_vals) #遞減
    indices = indices0[0][::-1]
    importance_vals =importance_vals[0]
    sorted_names = [base_fea[i] for i in indices]
    
    print("2 indice0", indices0)
    print("2 indice", indices)
    print("2 sorted=",sorted_names)
    print("2 imp value", importance_vals[indices])    
    #print("SelectFromModel: coef=",coef) # 跟我原來寫的一樣PlotFeatureImportantLog
    feature_ranks=[]   
    for i in indices:
        feature_ranks.append(f"{i}. {base_fea[i]}")
    print("SelectFromModel: ranking feature=",feature_ranks)


# In[8]:


def PlotFeatureImportantLgb(base_fea,algorithm_name, clf, X, y):
    #--------------變數重要性判斷---------------------- 
    # Plot the feature importances of the Lgb
    
    print("<<Feature importance of ",algorithm_name,">>")
   
    #importance_vals = clf.best_estimator_.feature_importance_
    importance_vals = clf.feature_importances_
    #optimized_GBM.best_estimator_.feature_importances_
    #print("importance lgb_clf=",importance_vals)
    
    #std = np.std([lgb.feature_importances_ for lgb in lgb_clf.estimators_],axis=0)
    indices = np.argsort(importance_vals)
    indices = indices[::-1]
    sorted_names = [base_fea[i] for i in indices]
    # Plot the feature importances of the forest
    plt.figure(figsize=(10,10))
    plt.title(algorithm_name + " feature importance")
    #plt.bar(range(X.shape[1]), importance_vals[indices],yerr=std[indices], align="center")
    plt.bar(range(X.shape[1]), importance_vals[indices], align="center")
    plt.xticks(range(X.shape[1]), sorted_names, rotation=90)
    plt.xlim([-1, X.shape[1]])
    #plt.xlabel(base_features14)
    maxY = importance_vals[indices[0]]*1.1
    plt.ylim([0, maxY])
    plt.show()


# In[9]:


def GetModelPerformance(alg_name, clf, X_train, y_train, X_test, y_test):    
    print()
    print("<<<<< Model Performance Indexes: ",alg_name, " >>>>>")
    print('======================',alg_name,'======================')
    print('--------------------Train Performance--------------------')
    trian_y_pred = clf.predict(X_train)
    trian_y_pred_proba = clf.predict_proba(X_train)

    train_fpr, train_tpr, train_threshold = roc_curve(y_train, trian_y_pred_proba[:, 1])   #計算TPR(真陽性率),FPR(偽陽性率)
    
    #混淆矩陣
    trian_confusion = metrics.confusion_matrix(y_train, trian_y_pred)
    trian_TP = trian_confusion[1, 1]
    trian_TN = trian_confusion[0, 0]
    trian_FP = trian_confusion[0, 1]
    trian_FN = trian_confusion[1, 0]
    
    #AUC
    trian_roc_auc = round(auc(train_fpr, train_tpr),3)   #計算auc的值(ROC曲線下的面積)
    #accuracy
    train_acc = round((np.mean(trian_y_pred == y_train)),3)
    #Sensitivity敏感度 TP/TP+FN
    trian_SE = round((trian_TP / (trian_TP+trian_FN)),3)
    #Specificity特異性 TN/FP+TN
    train_SPE = round((trian_TN / (trian_FP+trian_TN)),3)
    #陽性預測力
    train_PPV = round((trian_TP / (trian_TP+trian_FP)),3)
    #陰性預測力
    train_NPV = round((trian_TN / (trian_FN+trian_TN)),3)    
    #LR+陽性概似比
    train_LR1 = round((trian_TP /((trian_TP+trian_FN) / (trian_FP/(trian_FP+trian_TN)))),3)    
    #LR-陰性概似比
    train_LR0 = round((trian_FN/((trian_TP+trian_FN) / (trian_TN/(trian_FP+trian_TN)))),3)
    # F1(Recall and Precision的調和平均)=2TP/(P+P′)=2TP/(2TP+FP+FN)
    train_f1 = round(((2*trian_TP) / (((2*trian_TP)+trian_FP+trian_FN))),3)
    train_f1_binary = round(f1_score(y_train, trian_y_pred, average='binary'),3)
    train_f1_micro = round(f1_score(y_train, trian_y_pred, average='micro'),3)


    train_Performance = pd.Series({'Algorithm':alg_name, 'train/test':'Train',
                                   'Accuracy':train_acc, 'Sensitivity':trian_SE, 'Specificity':train_SPE, 'AUC':trian_roc_auc,
                                   'PPV':train_PPV, 'NPV':train_NPV, 'LR+':train_LR1, 'LR-':train_LR0, 
                                   'F1-score':train_f1, 'F1-score binary':train_f1_binary, 'F1-score micro':train_f1_micro})
    print('TP:', trian_TP)
    print('TN:', trian_TN)
    print('FP:', trian_FP)
    print('FN:', trian_FN)
    print('Accuracy('+alg_name+'): %.3f' % train_acc)
    print('Sensitivity('+alg_name+'): %.3f' % trian_SE)
    print('Specificity('+alg_name+'): %.3f' % train_SPE)
    print('Train AUC('+alg_name+'): %.3f' % trian_roc_auc)
    print('PPV('+alg_name+'): %.3f' % train_PPV)
    print('NPV('+alg_name+'): %.3f' % train_NPV)
    print('LR+('+alg_name+'): %.3f' % train_LR1)
    print('LR-('+alg_name+'): %.3f' % train_LR0)
    print('F1-score('+alg_name+'): %.3f' % train_f1)
    print('F1-score binary('+alg_name+'): %.3f' % train_f1_binary)
    print('F1-score micro('+alg_name+'): %.3f' % train_f1_micro)

    print()
    print('--------------------Test Performance--------------------')
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    #log的ROC曲線     
    #fpr_log,tpr_log,threshold_log = roc_curve(y_test, y_pred_proba[:, 1])   #計算TPR(真陽性率),FPR(偽陽性率)
    test_fpr,test_tpr,test_threshold = roc_curve(y_test, y_pred_proba[:, 1])   #計算TPR(真陽性率),FPR(偽陽性率)
       
    #print(alg_name, ': Accuracy: %.3f' % ((y_test == y_pred).sum() / y_test.shape[0]))       
    
    #混淆矩陣
    confusion = metrics.confusion_matrix(y_test, y_pred)
    test_TP = confusion[1, 1]
    test_TN = confusion[0, 0]
    test_FP = confusion[0, 1]
    test_FN = confusion[1, 0]
    
    #AUC
    test_roc_auc = round(auc(test_fpr, test_tpr),3)   #計算auc的值(ROC曲線下的面積)
    #accuracy
    test_acc = round((np.mean(y_pred == y_test)),3)
    #Sensitivity敏感度 TP/TP+FN
    test_SE = round((test_TP / (test_TP+test_FN)),3)
    #Specificity特異性 TN/FP+TN
    test_SPE = round((test_TN / (test_FP+test_TN)),3)
    #陽性預測力
    test_PPV = round((test_TP / (test_TP+test_FP)),3)
    #陰性預測力
    test_NPV = round((test_TN / (test_FN+test_TN)),3)    
    #LR+陽性概似比
    test_LR1 = round((test_TP /((test_TP+test_FN) / (test_FP/(test_FP+test_TN)))),3)    
    #LR-陰性概似比
    test_LR0 = round((test_FN/((test_TP+test_FN) / (test_TN/(test_FP+test_TN)))),3)
    # F1(Recall and Precision的調和平均)=2TP/(P+P′)=2TP/(2TP+FP+FN)
    test_f1 = round(((2*test_TP) / (((2*test_TP)+test_FP+test_FN))), 3)
    test_f1_binary = round(f1_score(y_test, y_pred, average='binary'), 3)
    test_f1_micro = round(f1_score(y_test, y_pred, average='micro'), 3)
    
    test_Performance = pd.Series({'Algorithm':alg_name, 'train/test':'Test',
                                   'Accuracy':test_acc, 'Sensitivity':test_SE, 'Specificity':test_SPE, 'AUC':test_roc_auc,
                                   'PPV':test_PPV, 'NPV':test_NPV, 'LR+':test_LR1, 'LR-':test_LR0, 
                                   'F1-score':test_f1, 'F1-score binary':test_f1_binary, 'F1-score micro':test_f1_micro})
    
    print('TP:', test_TP)
    print('TN:', test_TN)
    print('FP:', test_FP)
    print('FN:', test_FN)
    print('Accuracy('+alg_name+'): %.3f' % test_acc)
    print('Sensitivity('+alg_name+'): %.3f' % test_SE)
    print('Specificity('+alg_name+'): %.3f' % test_SPE)
    print('Train AUC('+alg_name+'): %.3f' % test_roc_auc)
    print('PPV('+alg_name+'): %.3f' % test_PPV)
    print('NPV('+alg_name+'): %.3f' % test_NPV)
    print('LR+('+alg_name+'): %.3f' % test_LR1)
    print('LR-('+alg_name+'): %.3f' % test_LR0)
    print('F1-score('+alg_name+'): %.3f' % test_f1)
    print('F1-score binary('+alg_name+'): %.3f' % test_f1_binary)
    print('F1-score micro('+alg_name+'): %.3f' % test_f1_micro)
    print(alg_name, ': Test classification:')
    print(classification_report(y_test, y_pred, digits=3))   
    
    model_Performance = pd.concat([train_Performance, test_Performance],  axis = 1)
    
    return y_pred, y_pred_proba, test_fpr, test_tpr, test_roc_auc, model_Performance


# In[10]:


##暫沒用到 ###
def show_train_history_auc(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history.history[SHOW_HISTORY_TRAIN_ACC])
    plt.plot(train_history.history[SHOW_HISTORY_VALIDATE_ACC])    
    #plt.plot(train_history.history["auc"])
    #plt.plot(train_history.history["val_auc"])

    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# In[11]:


def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history.history[SHOW_HISTORY_TRAIN_ACC]) ###
    plt.plot(train_history.history[SHOW_HISTORY_VALIDATE_ACC])
    #plt.plot(train_history.history["acc"]) ### Fred home PC
    #plt.plot(train_history.history["val_acc"])

    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# In[12]:


def DataPreprocessing(outcome,if_oversampling):
    ### parameters: outcome: 分析的outcome字串, features:選擇之feature變數s, y0_meaning: y=0的意義，ex:存活、成功, y1_meaning:y=1的意義，ex:死亡、失敗
    #def CnnBuildModel(outcome,features, y0_meaning, y1_meaning, if_multi_class):

    y0_meaning="沒"+outcome
    y1_meaning="有"+outcome
    print("===================",outcome, " Modeling=============================")

    X = df[Full_features]
    y = df[outcome]
    n_neg = (y==0).sum()
    n_pos = (y==1).sum()
    print("Raw total count y=", y0_meaning, "/",n_neg)
    print("Raw total count y=", y1_meaning, "/",n_pos)
    print("Raw total count (all)/",n_neg+n_pos)
    
    dynamic_class_weight = CalculateClassWeights(n_neg, n_pos)
    print('dynamic_class_weight=',dynamic_class_weight)
#X_train, X_test, y_train, y_test = train_test_split(X.values, y.values,test_size=0.3, stratify=y,random_state=19)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values,test_size=Test_split_test_Size, stratify=y, 
                                                        random_state=22)
    if if_oversampling:
        sm = BorderlineSMOTE(random_state=22) #random_state=7
        X_train, y_train = sm.fit_resample(X_train, y_train)     
        #         X_test, y_test = sm.fit_resample(X_test, y_test) 
        print("Train-smote count y=",y0_meaning, "/",(y_train==0).sum())
        print("Train-smote count y=",y1_meaning, "/",(y_train==1).sum())
        print("Train-smote total count(all)/",(y_train==0).sum()+(y_train==1).sum())

    else:
        print("Train-nosmote count y=",y0_meaning, "/",(y_train==0).sum())
        print("Train-nosmote count y=",y1_meaning, "/",(y_train==1).sum())
        print("Train-nosmote total count(all)/",(y_train==0).sum()+(y_train==1).sum())
        
    print("Test count y=",y0_meaning, "/",(y_test==0).sum())
    print("Test count y=",y1_meaning, "/",(y_test==1).sum())
    total_test_y1=(y_test==1).sum()
    print("Test total count(all)/",total_test_y1+(y_test==0).sum())    
    #print('dynamic Class Weight',dynamic_class_weight)        
    y_train_onehot=np_utils.to_categorical(y_train)
    y_test_onehot=np_utils.to_categorical(y_test)
    ## train_test_split 會將數據幀轉換為不再具有列的numpy數組
    X_train = pd.DataFrame(data=X_train, columns=Full_features)
    X_test = pd.DataFrame(data=X_test, columns=Full_features)
    return X_train, X_test, y_train, y_train_onehot, y_test, y_test_onehot, dynamic_class_weight


# In[13]:


#畫 confusion matrix
#    plot_confusion_matrix(cm,range(0,2),title='Confusion matrix_MLP'+'('+ outcome +')')
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=10)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[14]:


def ExportModelPKL(fileheader, alg_name, outcome, clf):
    now = datetime.datetime.now()
    #轉換為指定的格式:
    Daystyle = now.strftime("%Y%m%d%H%M%S")

    print("-------output PKL: ",alg_name," -------------------")
    #rndno = str(random.randint(1,1000))
    pkl= pkl_path + fileheader+outcome+"_"+alg_name+'_'+ Daystyle + '.pkl'
    print(Daystyle,'PKL file=',pkl)
    joblib.dump(clf, pkl,compress=3)      
#    if alg_name == 'mlp':
#        h5=fileheader+"_"+alg_name+"_"+outcome+"_" + Daystyle +".h5"
#        print(Daystyle," H5 file=",h5)
#        clf.save(h5)


# In[15]:


### function description ########
### parameters: outcome: 分析的outcome字串, features:選擇之feature變數s, y0_meaning: y=0的意義，ex:存活、成功, y1_meaning:y=1的意義，ex:死亡、失敗
def BuildModel2(algorithms, X_train, X_test, y_train_onehot, y_test, y_test_onehot, outcome):
    y_pred_log= y_pred_rnd= y_pred_svm=y_pred_knn =y_pred_lgb= y_pred_mlp=y_pred_xgb=y_pred_voting=[]
    y_pred_log_proba= y_pred_rnd_proba= y_pred_svm_proba=y_pred_knn_proba =y_pred_lgb_proba= y_pred_mlp_proba=y_pred_xgb_proba=y_pred_voting_proba=[]

    classes = np.unique(y_test)    

    for alg in algorithms:
        if alg=='LR':
            # When use pipeline class
            LR_pipeline = Pipeline([('scaler', StandardScaler()), ('LR', LogisticRegression(max_iter=1000))])
            LR_parameters={'LR__C': [1], 'LR__max_iter': [1000], 'LR__penalty': ['l2']} #20, 50
#             LR_parameters={'LR__penalty': ['l1', 'l2'],'LR__C':np.logspace(-3,3,7),'LR__max_iter':[100,1000,2000]} 
#             LR_parameters={'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,1.2,5,15,30,100],'max_iter':[10,30,50,100,1000]}

            gsearch_LR=GridSearchCV(LR_pipeline,LR_parameters, scoring='roc_auc', n_jobs=-1)
            gsearch_LR.fit(X_train, y_train)
            print(gsearch_LR.best_params_)
            print(gsearch_LR.score(X_test, y_test))
            log_clf = gsearch_LR.best_estimator_
            stac = log_clf
            print()
            print("----- LogisticRegression best_params_", gsearch_LR.best_estimator_)
            y_pred_log, y_pred_log_proba, fpr_log,tpr_log, roc_auc_log,Performance_LR  = GetModelPerformance('Logistic Regression', log_clf, X_train, y_train, X_test, y_test)  # 未標準化
           
            y_ANS = pd.Series(y_test)
            ROC_per_LR =  pd.Series(y_pred_log_proba[:, 1])
            confusion = metrics.confusion_matrix(y_test, y_pred_log)   
            np.set_printoptions(precision=2)
            plot_confusion_matrix(confusion,range(0,2),title='Confusion matrix_Logistic Regression'+'('+ outcome +')')
            #### 畫coef圖 , 使用正規化的X值                
            if IF_LOG_NORMALIZED==True:
                PlotFeatureImportantLog(Full_features,"Logistic Regression",log_clf,X_train,y_train, outcome,True)                                     
            else:
                PlotFeatureImportantLog(Full_features,"Logistic Regression",log_clf,X_train,y_train, outcome,True)    

            
        if alg=='RF':
            #rnd_clf = RandomForestClassifier(criterion = "gini",n_estimators=1000)
            #rnd_clf.fit(X_train, y_train)
            rnd_parameters = {'class_weight': ['balanced'], 'criterion': ['gini'], 
                              'max_depth':[7], 'max_features': ['auto'], 
                              'n_estimators':[500] ,'n_jobs':[-1]}   #'oob_score':[True],
            
            rfc=RandomForestClassifier(random_state=22) #random_state=42
            CV_rfc = GridSearchCV(estimator=rfc, param_grid=rnd_parameters, cv= 5)
            CV_rfc.fit(X_train, y_train) 
            rnd_clf = CV_rfc.best_estimator_                 
            #rnd_clf = RandomForestClassifier(criterion = "gini",n_estimators=1000,class_weight = class_weights)
            #rnd_clf.fit(X_train, y_train)
            print("----- Random forest best_params_",CV_rfc.best_params_)
            y_pred_rnd, y_pred_rnd_proba, fpr_rnd,tpr_rnd, roc_auc_rnd, Performance_RF = GetModelPerformance('Random Forest', rnd_clf, X_train, y_train, X_test, y_test)  # 未標準化
            ROC_per_RF =  pd.Series(y_pred_rnd_proba[:, 1])
            confusion = metrics.confusion_matrix(y_test, y_pred_rnd)
            np.set_printoptions(precision=2)
            plot_confusion_matrix(confusion,range(0,2),title='Confusion matrix_Random Forest'+'('+ outcome +')')
            #base_features=features                        
            PlotFeatureImportantRnd(Full_features,"RandomForest",rnd_clf,X_train,y_train, outcome,True)#False    
                               
        if alg=='SVM':
            svm_clf = SVC(probability=True)
            svm_clf.fit(X_train, y_train)                                                    
            y_pred_svm, y_pred_svm_proba, fpr_svm,tpr_svm, roc_auc_svm, Performance_svm = GetModelPerformance('SVM', svm_clf, X_train, y_train, X_test, y_test)  # 未標準化
            ROC_per_SVM =  pd.Series(y_pred_svm_proba[:, 1])
            confusion = metrics.confusion_matrix(y_test, y_pred_svm)
            np.set_printoptions(precision=2)
            plot_confusion_matrix(confusion,range(0,2),title='Confusion matrix_SVM'+'('+ outcome +')')    
            
        if alg=='lightGBM':
            lgb_parameters = {'learning_rate': [0.01],'num_iterations': [500]}
            
            gbm = lgb.LGBMClassifier(objective = 'binary', is_unbalance = True,
                                     metric = 'binary_logloss,auc', max_depth = 6,
                                     num_leaves = 40,learning_rate = 0.1,feature_fraction = 0.7,
                                     min_child_samples=15,min_child_weight=0.001,
                                     bagging_fraction = 1,bagging_freq = 2,
                                     reg_alpha = 0.01,cat_smooth = 0,num_iterations = 200)
            lgb_tmp = GridSearchCV(gbm, param_grid=lgb_parameters, scoring='roc_auc', cv=5)
            lgb_tmp.fit(X_train,y_train)

            lgb_clf = lgb_tmp.best_estimator_  
            print("----- lightGBM best_params_",lgb_tmp.best_params_)
            #print("lgb_clf.best_score_ = ", lgb_clf.best_score_)
            y_pred_lgb, y_pred_lgb_proba, fpr_lgb,tpr_lgb, roc_auc_lgb,  Performance_lgb = GetModelPerformance('LightGBM', lgb_clf, X_train, y_train, X_test, y_test)  # 未標準化
            ROC_per_lgb =  pd.Series(y_pred_lgb_proba[:, 1])
            confusion = metrics.confusion_matrix(y_test, y_pred_lgb)
            np.set_printoptions(precision=2)
            plot_confusion_matrix(confusion,range(0,2),title='Confusion matrix_lightGBM'+'('+ outcome +')')                                                     
#            PlotFeatureImportantLgb(Full_features,"lightGBM",lgb_clf,X_train,y_train)

            fig,ax = plt.subplots(figsize=(10,10))
            lgb.plot_importance(lgb_clf,height=0.5,ax=ax,max_num_features=64, 
                                importance_type='split',title='lightGBM-Feature importance(type:split)', ignore_zero=False,
                                grid=False)
            plt.savefig(plot_path + 'lightGBM-Feature importance(split)'+'('+ outcome +')'+'.png', bbox_inches = 'tight',dpi=300)
            
            fig,ax = plt.subplots(figsize=(10,10))
            lgb.plot_importance(lgb_clf,height=0.5,ax=ax,max_num_features=64, 
                                importance_type='gain',title='lightGBM-Feature importance(type:gain)',ignore_zero=False,
                                grid=False)
            plt.savefig(plot_path + 'lightGBM-Feature importance(gain)'+'('+ outcome +')'+'.png', bbox_inches = 'tight',dpi=300)
            
                        ########## #shap圖
#             explainer = shap.TreeExplainer(lgb_clf)
#             lgb_shap_values = explainer.shap_values(X_test)
#             plt.figure(figsize=(10,10))
#             shap.summary_plot(lgb_shap_values[1], X_test, show=False, max_display=30, title=None, sort=True)
#             plt.savefig(plot_path +'SHAP_lightGBM'+'('+outcome+')'+'.png', bbox_inches='tight', dpi=300)
#             plt.savefig(plot_path +'SHAP_lightGBM'+'('+outcome+')'+'.tif', bbox_inches='tight', dpi=300)
           
#             # Feature importance
#             plt.figure(figsize=(10,10))
#             shap.summary_plot(lgb_shap_values[1],X_test, show=False, max_display=30, title=None, sort=True,plot_type='bar')
# #             shap.plots.bar(lgb_shap_values_f, max_display=30)
#             plt.savefig(plot_path +'SHAP_lightGBM_bar'+'('+outcome+')'+'.png', bbox_inches='tight', dpi=300)
#             plt.savefig(plot_path +'SHAP_lightGBM_bar'+'('+outcome+')'+'.tif', bbox_inches='tight', dpi=300)
            
        if alg=='MLPClassifier': #(90,50),
            mlp_parameters = {'hidden_layer_sizes':[(100)],
                              'max_iter':[15],'learning_rate': ['constant'],
                              'learning_rate_init':[0.001],'shuffle':[True],'random_state':[15]} #15
            mlp_clf = MLPClassifier(max_iter=1000,batch_size=16,early_stopping=True) 
            gsearch_mlp = GridSearchCV(mlp_clf, param_grid=mlp_parameters, scoring='roc_auc', cv=5)
    
            mlp_clf = gsearch_mlp.fit(X_train,y_train)
            mlp_clf = gsearch_mlp.best_estimator_
            print('Mlp-参数的最佳取值:{0}'.format(gsearch_mlp.best_params_))
            print('Mlp-最佳模型得分:{0}'.format(gsearch_mlp.best_score_))
            print('Mlp-mean_score:', gsearch_mlp.cv_results_['mean_test_score'])
            print('Mlp-papameters:', gsearch_mlp.cv_results_['params'])

            y_pred_mlp, y_pred_mlp_proba, fpr_mlp,tpr_mlp, roc_auc_mlp , Performance_mlp= GetModelPerformance('MLPClassifier', mlp_clf, X_train, y_train, X_test, y_test)  
            ROC_per_mlp =  pd.Series(y_pred_mlp_proba[:, 1])
            confusion = metrics.confusion_matrix(y_test, y_pred_mlp)
            #class_names=y
            np.set_printoptions(precision=2)
            plot_confusion_matrix(confusion,range(0,2),title='Confusion matrix_MLP'+'('+ outcome +')')            
            
        if alg=='XGBoost':
#             xgb = XGBClassifier()             
            xgb = XGBClassifier(objective='binary:logistic',learning_rate=0.001,tree_method='exact',
                                scale_pos_weight=4,gamma=0.2,max_delta_step=16,colsample_bytree=0.25) #調參數 ,scale_pos_weight=4
            xgb.fit(X_train, y_train)
            y_pred_xgb, y_pred_xgb_proba, fpr_xgb,tpr_xgb, roc_auc_xgb ,Performance_xgb= GetModelPerformance('XGBoost', xgb, X_train, y_train, X_test, y_test)  # 未標準化
            ROC_per_xgb =  pd.Series(y_pred_xgb_proba[:, 1])
            confusion = metrics.confusion_matrix(y_test, y_pred_xgb)
            np.set_printoptions(precision=2)
            plot_confusion_matrix(confusion,range(0,2),title='Confusion matrix_xgboost'+'('+ outcome +')')
            
#             fig,ax = plt.subplots(figsize=(10,10))
#             plot_importance(xgb,height=0.5,ax=ax,max_num_features=64, 
#             importance_type='weight',title='XGBoost-Feature importance(type:weight)'+'('+ outcome +')',show_values=False)
    
#             fig,ax = plt.subplots(figsize=(10,10))
#             plot_importance(xgb,height=0.5,ax=ax,max_num_features=64, 
#             importance_type='gain',title='XGBoost-Feature importance(type:gain)'+'('+ outcome +')', show_values=False)
                        
            importance = xgb.get_booster().get_score(importance_type="gain")  # Get the importance dictionary (by gain) from the booster
            # make your changes
            for key in importance.keys():
                importance[key] = round(importance[key],3)
            fig,ax = plt.subplots(figsize=(10,10))  
            plot_importance(importance,height=0.5,ax=ax,max_num_features=64, xlabel='Feature Important',
                            importance_type='gain',title='XGBoost-Feature importance(type:gain)', 
                            show_values=True, grid=False)
            plt.savefig(plot_path +'XGBoost-Feature importance(gain)'+'('+ outcome +')'+'.png', bbox_inches = 'tight',dpi=300)
#             ########### Plot SHAP ########## 
#             plt.figure(figsize=(10,10))
#             explainer = shap.TreeExplainer(xgb)
#             xgb_shap_values = explainer.shap_values(X_test)
#             shap.summary_plot(xgb_shap_values, X_test, show=False, max_display=30, title=None, sort=True)
#             plt.savefig(plot_path + 'SHAP_XGboost'+'('+outcome+')'+'.png', bbox_inches='tight', dpi=300)     
#             plt.savefig(plot_path + 'SHAP_XGboost'+'('+outcome+')'+'.tif', bbox_inches='tight', dpi=300)  
            
#             # Feature importance
#             plt.figure(figsize=(10,10))
#             shap.summary_plot(xgb_shap_values, X_test, show=False, max_display=30, title=None, sort=True,plot_type='bar')
#             plt.savefig(plot_path + 'SHAP_XGBoost_bar'+'('+outcome+')'+'.png', bbox_inches='tight', dpi=300)     
#             plt.savefig(plot_path + 'SHAP_XGBoost_bar'+'('+outcome+')'+'.tif', bbox_inches='tight', dpi=300)  

        
        if alg=='voting':
            voting_clf = VotingClassifier(estimators = [('LR', log_clf),('svm',svm_clf),('mlp',mlp_clf),('RF',rnd_clf),('xgb',xgb)], voting = 'soft')                
            voting_clf.fit(X_train, y_train)
            y_pred_voting, y_pred_voting_proba, fpr_voting,tpr_voting, roc_auc_voting,  Performance_vot = GetModelPerformance('Voting', voting_clf, X_train, y_train, X_test, y_test)
            ROC_per_voting =  pd.Series(y_pred_voting_proba[:, 1])
                    
        if alg == 'Stacking':
            estimators = [('LR', log_clf),('svm',svm_clf),('mlp',mlp_clf),('RF',rnd_clf),('xgb',xgb)]
            sta_clf = StackingClassifier(estimators=estimators, final_estimator=stac)
#             sta_clf = StackingClassifier(estimators=estimators, final_estimator= MLPClassifier(max_iter=1000,batch_size=16,early_stopping=True))
                                         
            sta_clf.fit(X_train, y_train)
            y_pred_stacking, y_pred_stacking_proba, fpr_stacking,tpr_stacking, roc_auc_stacking,  Performance_stat = GetModelPerformance('Stacking', sta_clf, X_train, y_train, X_test, y_test)
            ROC_per_stacking =  pd.Series(y_pred_stacking_proba[:, 1])
                       
            
    ##########------- Export Model to PKL and/or H5 files   ###########   
#     print('>>>>>>>>>Export Models to PKL and/or H5 files <<<<<<<<<<<<<<<')
#     for alg in algorithms:
#         if alg=='LR':
#             ExportModelPKL(OUTFILE_HEADER, 'LR', outcome, log_clf)
#         if alg=='knn':                                                                            
#             ExportModelPKL(OUTFILE_HEADER, 'knn', outcome, knn_clf)
#         if alg=='SVM':                                                                               
#             ExportModelPKL(OUTFILE_HEADER, 'SVM', outcome, svm_clf)
#         if alg=='RF':                                                                               
#             ExportModelPKL(OUTFILE_HEADER, 'RF', outcome, rnd_clf)
#         if alg=='lightGBM':                                                                               
#             ExportModelPKL(OUTFILE_HEADER, 'lightGBM', outcome, lgb_clf)
#         if alg=='MLPClassifier':                                                                               
#             ExportModelPKL(OUTFILE_HEADER, 'MLPClassifier', outcome, mlp_clf)
#         if alg=='XGBoost':
#             ExportModelPKL(OUTFILE_HEADER, 'XGBoost', outcome, xgb)
    ##########------- ROC curve    #####################################   

    print('>>>>>>>>>ROC curve<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    lw = 2
    plt.figure(1)
    plt.figure(figsize=(8,8))
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')  #對角線
    for alg in algorithms:
        if alg=='LR':
            plt.plot(fpr_log, tpr_log, color='darkorange', lw=lw, label='Logistic Regression(AUC = %0.3f)' % roc_auc_log)
        if alg=='knn':                                                                            
            plt.plot(fpr_knn, tpr_knn, color='goldenrod',lw=lw,label='KNN(AUC = %0.3f)' % roc_auc_knn)
        if alg=='SVM':                                                                               
            plt.plot(fpr_svm, tpr_svm, color='cyan',lw=lw,label='SVM(AUC = %0.3f)' % roc_auc_svm)
        if alg=='RF':                                                                               
            plt.plot(fpr_rnd, tpr_rnd, color='black',lw=lw, label='Random Forest(AUC = %0.3f)' % roc_auc_rnd)
        if alg=='MLPClassifier':                                                                               
            plt.plot(fpr_mlp, tpr_mlp, color='blue',lw=lw,  linestyle=':', label='MLP(AUC = %0.3f)' % roc_auc_mlp)
        if alg=='lightGBM': 
            plt.plot(fpr_lgb, tpr_lgb, color='green',lw=lw,label='LightGBM(AUC = %0.3f)' % roc_auc_lgb)
        if alg=='XGBoost': 
            plt.plot(fpr_xgb, tpr_xgb, color='deeppink',lw=lw, label='XGBoost(AUC = %0.3f)' % roc_auc_xgb)
        if alg=='voting':
            plt.plot(fpr_voting, tpr_voting, color='saddlebrown',lw=lw,label='Voting(AUC = %0.3f)' % roc_auc_voting)
        if alg=='Stacking':
            plt.plot(fpr_stacking, tpr_stacking, color='pink',lw=lw,label='Stacking(AUC = %0.3f)' % roc_auc_stacking)
    
    plt.ylabel('True positive rate(Sensitivity)')
    plt.xlabel('False positive rate (1-Specificity)')
#    plt.title('ROC curve'+'('+ outcome +')')
    plt.title('ROC curve')
    plt.legend(loc='lower right')

###  plot ROC 
    if roc_save==True:
        plt.savefig(plot_path + 'ROC Curve'+'('+ outcome +')'+'.png', dpi=300)
        plt.savefig(plot_path + 'ROC Curve'+'('+ outcome +')'+'.tif', dpi=300)
    
    if report_save==True:
        report = pd.DataFrame((pd.concat([Performance_LR, Performance_RF,Performance_svm,
                                          Performance_mlp, Performance_xgb,Performance_vot,Performance_stat],
                                         axis = 1))).T.round(decimals = 3)
        report_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        report.to_csv(performance_path+ 'Model Performance_'+outcome +'_'+report_time+'.csv', index=0, encoding='utf_8_sig')   
    
    if prob_save==True:
        roc_prob = pd.DataFrame(pd.concat([y_ANS,ROC_per_LR,ROC_per_RF,ROC_per_SVM,
                                           ROC_per_mlp,ROC_per_xgb,ROC_per_voting,ROC_per_stacking],axis=1)) 
        roc_prob.columns = ['y_test','LR_prob','RF_prob','SVM_prob','mlp_prob','xgb_prob','voting_prob','Stacking_prob']
        roc_prob.to_csv( performance_path +'roc_problity_' + outcome +'_.csv',index=False)
    
     ########校準圖
# Create classifiers
    cv_K = KFold(n_splits=5, shuffle=True, random_state=22)#n_splits=5, shuffle=True, random_state=4
    clf_list  = [('LR', CalibratedClassifierCV(log_clf, method='sigmoid'))]
#                  ('RF', CalibratedClassifierCV(rnd_clf, method='sigmoid', cv=cv_K)), 
#                  ('SVM', CalibratedClassifierCV(svm_clf, method='sigmoid', cv=cv_K)),
#                  ('MLP', CalibratedClassifierCV(mlp_clf, method='isotonic', cv=cv_K)), 
#                  ('XGBoost', CalibratedClassifierCV(xgb, method='sigmoid', cv=cv_K)),]
    
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    markers = ["^"] #, "v", "s", "o","a"
    for i, (name,clf) in enumerate(clf_list):
        clf.fit(X_train, y_train)
        display = CalibrationDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
            marker=markers[i],
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (name,_) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])
                
        ax.hist(
            calibration_displays[name].y_prob,  #.y_prob[150:950]
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i))
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()
        
    return y_pred_log_proba, y_pred_rnd_proba, y_pred_svm_proba, y_pred_knn_proba,y_pred_lgb_proba, y_pred_mlp_proba,y_pred_xgb_proba,y_pred_voting_proba# all probability


# In[16]:


begin_time=datetime.datetime.now()
print("Starting time...", begin_time)


# In[17]:


df = pd.read_csv(DATA_FILE,encoding='utf-8')
df = df[Fe]
algs = Algorithms


# In[18]:


algs = Algorithms
roc_save = False
report_save = False #False #存
prob_save = False #False

print("[====== Start Modeling ICU (傳統ML法，IF_MM_SMOTE=True)=================================]")
X_train, X_test, y_train, y_train_onehot, y_test, y_test_onehot, dynamic_class_weight= DataPreprocessing('AECOPD within 3 months', IF_MM_SMOTE)
for i in range(1):
    print('第',i,'次建模(20).......................')
    y_pred_log_proba, y_pred_rnd_proba, y_pred_svm_proba, y_pred_knn_proba,y_pred_lgb_proba, y_pred_mlp_proba, y_pred_xgb_proba,y_pred_voting_proba = BuildModel2(algs,  X_train, X_test, y_train_onehot, y_test, y_test_onehot,'AECOPD within 3 months') #in_hosp_mortality


# In[19]:


New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_log_proba[:, 1],1, 0.55, 0.8)
print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[20]:


New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_rnd_proba[:, 1],1, 0.5, 0.71)
print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[ ]:


New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_svm_proba[:, 1],1, 0.55, 0.72)
print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[ ]:


# New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_knn_proba[:, 1],1, 0.55, 0.7)
# print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[ ]:


New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_lgb_proba[:, 1],1, 0.5, 0.73)
print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[ ]:


New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_mlp_proba[:, 1],1, 0.5, 0.7)
print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[ ]:


New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_xgb_proba[:, 1],1, 0.5, 0.732)
print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[ ]:


# New_threshold, New_AUC, New_sensitivity, New_specificity = BestThresholdPerformance(y_test, y_pred_svm_proba[:, 1],1, 0.55, 0.7)
# print("New_threshold=",New_threshold, ", New_AUC=",New_AUC, ", New_sensitivity=,", New_sensitivity, ", New_specificity=",New_specificity)


# In[ ]:





# In[ ]:





# In[ ]:




