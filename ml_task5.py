import numpy as np
import pandas as pd
from sklearn.ensemble import *

def train_model_return_scores(train_df_path,test_df_path) -> pd.DataFrame:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described

    # Feels like I'm missing something... my functions are oddly simple?
    
    # Create dataframes based on loaded CSV
    training_data = pd.read_csv(train_df_path)
    testing_data = pd.read_csv(test_df_path)

    # Create training data, target is class
    training_features = training_data.drop(columns=['class'])
    training_targets = training_data['class']

    # Is there a better model? Depends on accuracy score...
    model = RandomForestClassifier(random_state=0)
    model.fit(training_features, training_targets)

    test_prob = model.predict_proba(testing_data)[:,1]

    test_scores = pd.DataFrame({'index': testing_data.index, 'malware_score': test_prob})

    return test_scores 

def train_model_unsw_return_scores(train_df_path,test_df_path) -> pd.DataFrame:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described

    # Create dataframes based on loaded CSV
    training_data = pd.read_csv(train_df_path)
    testing_data = pd.read_csv(test_df_path)

    # Create training data, target is class
    training_features = training_data.drop(columns=['class'])
    training_targets = training_data['class'] # Not used

    # Using gradient boosting.... mainly because I used it in 4!
    # Tweaked settings to (hopefully) increase roc_auc
    # Commented values got me to 0.759390493692452
    model = GradientBoostingClassifier(
        n_estimators=300, # 500
        learning_rate=0.05, # 0.05
        max_depth=7, # 7
        min_samples_leaf=20, # 20
        random_state=0, # 0 
        subsample=0.8 # 0.8
    )

    model.fit(training_features, training_targets)    

    test_prob = model.predict_proba(testing_data)[:,1]
 
    test_scores = pd.DataFrame({'index':testing_data.index, 'prob_class_1':test_prob})
    return test_scores 
