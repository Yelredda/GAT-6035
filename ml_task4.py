import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.feature_selection import RFE

class ModelMetrics:
    def __init__(self, model_type:str,train_metrics:dict,test_metrics:dict,feature_importance_df:pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"
    
    def add_train_metric(self,metric_name:str,metric_val:float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self,metric_name:str,metric_val:float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self): 
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str


# ------------ Beginning of tasks - DO NOT MODIFY ABOVE ------------ 

# ---------- TESTS PASSED  ---------- 
def calculate_naive_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, naive_assumption:int) -> ModelMetrics:
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    
    # ---------------- Training  ----------------  
    # The naive model is just the same value repeated.
    naive_train = [naive_assumption] * len(train_features)
    accuracy = round( accuracy_score(train_targets, naive_train), 4 )
    recall = round( recall_score(train_targets, naive_train), 4 )
    precision = round( precision_score(train_targets, naive_train), 4 )
    fscore = round( f1_score(train_targets, naive_train), 4)  
 
    train_metrics["accuracy"] = accuracy
    train_metrics["recall"] = recall
    train_metrics["precision"] = precision
    train_metrics["fscore"] = fscore

    # ---------------- Testing  ----------------  
    # The naive model is just the same value repeated.
    naive_test = [naive_assumption] * len(test_features)
    accuracy = round( accuracy_score(test_targets, naive_test), 4 )
    recall = round( recall_score(test_targets, naive_test), 4 )
    precision = round( precision_score(test_targets, naive_test), 4 )
    fscore = round( f1_score(test_targets, naive_test), 4)  
 
    test_metrics["accuracy"] = accuracy
    test_metrics["recall"] = recall
    test_metrics["precision"] = precision
    test_metrics["fscore"] = fscore

    # Returns, unchanged
    naive_metrics = ModelMetrics("Naive",train_metrics,test_metrics,None)
    return naive_metrics

def calculate_logistic_regression_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, logreg_kwargs) -> tuple[ModelMetrics,LogisticRegression]:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html and implement the function as described
    model = LogisticRegression()
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    
    # ---------------- Training  ----------------  
    # Train logistic regression model
    logistic_model = LogisticRegression(**logreg_kwargs)
    logistic_model.fit( train_features, train_targets )

    train_predict = logistic_model.predict(train_features)
    train_probability = logistic_model.predict_proba(train_features)[:,1]

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # In the binary case, we can extract true positives, etc. as follows:
    # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    tn, fp, fn, tp = confusion_matrix(train_targets, train_predict).ravel()
    
    accuracy = round( accuracy_score(train_targets, train_predict), 4 )
    recall = round( recall_score(train_targets, train_predict), 4 )
    precision = round( precision_score(train_targets, train_predict), 4 )
    fscore = round( f1_score(train_targets, train_predict), 4)
    fpr = round( fp / (fp + tn), 4)
    fnr = round( fn / (fn + tp), 4)
    roc_auc = round( roc_auc_score(train_targets, train_probability ), 4 )

    train_metrics["accuracy"] = accuracy
    train_metrics["recall"] = recall
    train_metrics["precision"] = precision
    train_metrics["fscore"] = fscore
    train_metrics["fpr"] = fpr
    train_metrics["fnr"] = fnr
    train_metrics["roc_auc"] = roc_auc

    #  ---------------- Testing  ----------------  
    test_predict = logistic_model.predict(test_features)
    test_probability = logistic_model.predict_proba(test_features)[:,1]

    tn, fp, fn, tp = confusion_matrix(test_targets, test_predict).ravel()
    
    accuracy = round( accuracy_score(test_targets, test_predict), 4 )
    recall = round( recall_score(test_targets, test_predict), 4 )
    precision = round( precision_score(test_targets, test_predict), 4 )
    fscore = round( f1_score(test_targets, test_predict), 4)
    fpr = round( fp / (fp + tn), 4)
    fnr = round( fn / (fn + tp), 4)
    roc_auc = round( roc_auc_score(test_targets, test_probability ), 4 )

    test_metrics["accuracy"] = accuracy
    test_metrics["recall"] = recall
    test_metrics["precision"] = precision
    test_metrics["fscore"] = fscore
    test_metrics["fpr"] = fpr
    test_metrics["fnr"] = fnr
    test_metrics["roc_auc"] = roc_auc

    # RFE feature training 
    rfe_model = RFE(logistic_model, n_features_to_select=10 )
    rfe_model.fit( train_features, train_targets )

    top_train_features = train_features.iloc[:, rfe_model.support_]

    top_logistic_model = LogisticRegression(**logreg_kwargs)
    top_logistic_model.fit(top_train_features, train_targets)

    # TODO: Can't get the last test test_importance to pass.
    feature_importance = pd.DataFrame( {"Feature": top_train_features.columns, "Importance": top_logistic_model.coef_[0] }  )
    
    # Convert to absolute value THEN sort by importance (desc) THEN round THEN reset index
    #feature_importance["Importance"] = feature_importance["Importance"].abs()
    #feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    #feature_importance["Importance"] = feature_importance["Importance"].round(4)
    #feature_importance = feature_importance.head(10) 
    #feature_importance = feature_importance.reset_index(drop=True)

    # Importance, top 10, rounded 4, sorted in descending order, then reordered 
    feature_importance = pd.DataFrame( { 'Feature': top_train_features.columns, 'Importance': top_logistic_model.coef_[0] })
    feature_importance['Importance'] = feature_importance['Importance'].abs()
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    feature_importance = feature_importance.head(10)
    feature_importance['Importance'] = feature_importance["Importance"].round(4)
    feature_importance.reset_index(drop=True, inplace=True)

    #model = top_train
    model = top_logistic_model
    log_reg_importance = feature_importance

    # Returns, unchanged
    #log_reg_importance = pd.DataFrame()

    log_reg_metrics = ModelMetrics("Logistic Regression",train_metrics,test_metrics,log_reg_importance)

    return log_reg_metrics,model

# ---------- TESTS PASSED  ---------- 
def calculate_random_forest_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, rf_kwargs) -> tuple[ModelMetrics,RandomForestClassifier]:
    model = RandomForestClassifier(**rf_kwargs)
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }

    # Model fiting, test/train predictions/probabilities
    model.fit(train_features, train_targets) 

    train_predict = model.predict(train_features)
    test_predict = model.predict(test_features)

    train_prob = model.predict_proba(train_features)[:,1]
    test_prob = model.predict_proba(test_features)[:,1]

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # In the binary case, we can extract true positives, etc. as follows:
    # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

    #  ---------------- Training  ----------------
    tn, fp, fn, tp = confusion_matrix(train_targets, train_predict).ravel()
    
    accuracy = round( accuracy_score(train_targets, train_predict), 4 )
    recall = round( recall_score(train_targets, train_predict), 4 )
    precision = round( precision_score(train_targets, train_predict), 4 )
    fscore = round( f1_score(train_targets, train_predict), 4)
    fpr = round( fp / (fp + tn), 4)
    fnr = round( fn / (fn + tp), 4)
    roc_auc = round( roc_auc_score(train_targets, train_prob ), 4 )

    train_metrics["accuracy"] = accuracy
    train_metrics["recall"] = recall
    train_metrics["precision"] = precision
    train_metrics["fscore"] = fscore
    train_metrics["fpr"] = fpr
    train_metrics["fnr"] = fnr
    train_metrics["roc_auc"] = roc_auc    

    #  ---------------- Testing  ----------------  
    tn, fp, fn, tp = confusion_matrix(test_targets, test_predict).ravel()
    
    accuracy = round( accuracy_score(test_targets, test_predict), 4 )
    recall = round( recall_score(test_targets, test_predict), 4 )
    precision = round( precision_score(test_targets, test_predict), 4 )
    fscore = round( f1_score(test_targets, test_predict), 4)
    fpr = round( fp / (fp + tn), 4)
    fnr = round( fn / (fn + tp), 4)
    roc_auc = round( roc_auc_score(test_targets, test_prob ), 4 )

    test_metrics["accuracy"] = accuracy
    test_metrics["recall"] = recall
    test_metrics["precision"] = precision
    test_metrics["fscore"] = fscore
    test_metrics["fpr"] = fpr
    test_metrics["fnr"] = fnr
    test_metrics["roc_auc"] = roc_auc

    # Importance, top 10, rounded 4, sorted in descending order, then reordered 
    rf_importance = pd.DataFrame( { 'Feature': train_features.columns, 'Importance': model.feature_importances_ })
    rf_importance = rf_importance.sort_values(by='Importance', ascending=False)
    rf_importance = rf_importance.head(10)
    rf_importance['Importance'] = rf_importance["Importance"].round(4)
    rf_importance.reset_index(drop=True, inplace=True)


    # Untouched
    rf_metrics = ModelMetrics("Random Forest",train_metrics,test_metrics,rf_importance)
    return rf_metrics,model

def calculate_gradient_boosting_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, gb_kwargs) -> tuple[ModelMetrics,GradientBoostingClassifier]:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html and implement the function as described
    model = GradientBoostingClassifier(**gb_kwargs)
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }

    # Model fiting, test/train predictions/probabilities
    model.fit(train_features, train_targets) 

    train_predict = model.predict(train_features)
    test_predict = model.predict(test_features)

    train_prob = model.predict_proba(train_features)[:,1]
    test_prob = model.predict_proba(test_features)[:,1]

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # In the binary case, we can extract true positives, etc. as follows:
    # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

    #  ---------------- Training  ----------------
    tn, fp, fn, tp = confusion_matrix(train_targets, train_predict).ravel()
    
    accuracy = round( accuracy_score(train_targets, train_predict), 4 )
    recall = round( recall_score(train_targets, train_predict), 4 )
    precision = round( precision_score(train_targets, train_predict), 4 )
    fscore = round( f1_score(train_targets, train_predict), 4)
    fpr = round( fp / (fp + tn), 4)
    fnr = round( fn / (fn + tp), 4)
    roc_auc = round( roc_auc_score(train_targets, train_prob ), 4 )

    train_metrics["accuracy"] = accuracy
    train_metrics["recall"] = recall
    train_metrics["precision"] = precision
    train_metrics["fscore"] = fscore
    train_metrics["fpr"] = fpr
    train_metrics["fnr"] = fnr
    train_metrics["roc_auc"] = roc_auc    

    #  ---------------- Testing  ----------------  
    tn, fp, fn, tp = confusion_matrix(test_targets, test_predict).ravel()
    
    accuracy = round( accuracy_score(test_targets, test_predict), 4 )
    recall = round( recall_score(test_targets, test_predict), 4 )
    precision = round( precision_score(test_targets, test_predict), 4 )
    fscore = round( f1_score(test_targets, test_predict), 4)
    fpr = round( fp / (fp + tn), 4)
    fnr = round( fn / (fn + tp), 4)
    roc_auc = round( roc_auc_score(test_targets, test_prob ), 4 )

    test_metrics["accuracy"] = accuracy
    test_metrics["recall"] = recall
    test_metrics["precision"] = precision
    test_metrics["fscore"] = fscore
    test_metrics["fpr"] = fpr
    test_metrics["fnr"] = fnr
    test_metrics["roc_auc"] = roc_auc

    # Importance, top 10, rounded 4, sorted in descending order, then reordered 
    gb_importance = pd.DataFrame( { 'Feature': train_features.columns, 'Importance': model.feature_importances_ })
    gb_importance = gb_importance.sort_values(by='Importance', ascending=False)
    gb_importance = gb_importance.head(10)
    gb_importance['Importance'] = gb_importance["Importance"].round(4)
    gb_importance.reset_index(drop=True, inplace=True)

    gb_metrics = ModelMetrics("Gradient Boosting",train_metrics,test_metrics,gb_importance)

    return gb_metrics,model