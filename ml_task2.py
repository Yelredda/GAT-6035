import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection

# ---------- TESTS PASSED  ---------- 
def tts(  dataset: pd.DataFrame,
                       label_col: str, 
                       test_size: float,
                       stratify: bool,
                       random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:

    # Split into features (independent variables) and labels (dependent variable) 
    features = dataset.drop(columns=label_col)
    labels = dataset[label_col]

    # Call train_test_split to separate the data into test/train data sets
    if (stratify) : # Bool argument, stratify on label_col
        train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split( 
            features, labels, test_size=test_size, random_state=random_state, stratify=labels )
    else: 
        train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split( 
            features, labels, test_size=test_size, random_state=random_state )

    return train_features,test_features,train_labels,test_labels

class PreprocessDataset:
    def __init__(self, 
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        
        self.n_components = n_components

        # Better way to generate column names? No autonaming switch in docs
        self.column_names = []
        for i in range( self.n_components):
            self.column_names.append( 'component_' + str(i+1) ) 
        
        self.feature_engineering_functions = feature_engineering_functions

        # Created in init for sharing between train/test data functions
        self.encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False,handle_unknown='infrequent_if_exist') # Needed for PURPLE
        self.scaler = sklearn.preprocessing.MinMaxScaler()
        self.pca = sklearn.decomposition.PCA(n_components=self.n_components, random_state=0)

        return

    # ---------- TESTS PASSED  ---------- 
    def one_hot_encode_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        
        # Split into encode and other datasets
        to_hot_encode_columns = train_features[self.one_hot_encode_cols]
        other_columns = train_features.drop(columns=self.one_hot_encode_cols)

        # Call encoder to fit and transform columns
        to_hot_encode_array = self.encoder.fit_transform(to_hot_encode_columns)

        # Transform array into dataframe
        encoded_dataset = pd.DataFrame(to_hot_encode_array, 
                                       columns=self.encoder.get_feature_names_out(),
                                       index=train_features.index
                                       )
        
        # Final concat to combine one hot and original datasets
        one_hot_encoded_dataset = pd.concat([encoded_dataset, other_columns], axis=1)

        return one_hot_encoded_dataset

    # ---------- TESTS PASSED  ---------- 
    def one_hot_encode_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        
        # Split into encode and other datasets
        to_hot_encode_columns = test_features[self.one_hot_encode_cols]
        other_columns = test_features.drop(columns=self.one_hot_encode_cols)

        # Call encoder to fit and transform columns
        to_hot_encode_array = self.encoder.transform(to_hot_encode_columns)

        # Transform array into dataframe
        encoded_dataset = pd.DataFrame(to_hot_encode_array, 
                                       columns=self.encoder.get_feature_names_out(),
                                       index=test_features.index
                                       )

        # Final concat to combine one hot and original datasets        
        one_hot_encoded_dataset = pd.concat([encoded_dataset, other_columns], axis=1)

        return one_hot_encoded_dataset

    # ---------- TESTS PASSED  ---------- 
    def min_max_scaled_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:

        train_features[self.min_max_scale_cols] = self.scaler.fit_transform( train_features[self.min_max_scale_cols] )
        min_max_scaled_dataset = train_features 

        return min_max_scaled_dataset

    # ---------- TESTS PASSED  ---------- 
    def min_max_scaled_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:

        test_features[self.min_max_scale_cols] = self.scaler.transform( test_features[self.min_max_scale_cols] )
        min_max_scaled_dataset = test_features 

        return min_max_scaled_dataset

    # ---------- TESTS PASSED  ---------- 
    def pca_train(self,train_features:pd.DataFrame) -> pd.DataFrame:

        # Drop N/A columns just in case they exist
        dropped_dataset = train_features.dropna(axis='columns')

        # Call PCA on cleaned up dataframe
        transformed_dataset = self.pca.fit_transform(dropped_dataset)

        # Create dataframe w/ custom columns
        pca_dataset = pd.DataFrame( transformed_dataset, columns=self.column_names, index=train_features.index)

        return pca_dataset

    # ---------- TESTS PASSED  ---------- 
    def pca_test(self,test_features:pd.DataFrame) -> pd.DataFrame:

        # Drop N/A columns just in case they exist
        dropped_dataset = test_features.dropna(axis='columns')

        # Call PCA on cleaned up dataframe
        transformed_dataset = self.pca.transform(dropped_dataset)

        # Create dataframe w/ custom columns
        pca_dataset = pd.DataFrame( transformed_dataset, columns=self.column_names, index=test_features.index)

        return pca_dataset

    # ---------- TESTS PASSED  ---------- 
    def feature_engineering_train(self,train_features:pd.DataFrame) -> pd.DataFrame:

        # You can store functions in variables!! 
        for column, function in self.feature_engineering_functions.items():
            train_features[column] = function(train_features)

        feature_engineered_dataset = train_features

        return feature_engineered_dataset
    
    # ---------- TESTS PASSED  ---------- 
    def feature_engineering_test(self,test_features:pd.DataFrame) -> pd.DataFrame:

        # You can store functions in variables!! 
        for column, function in self.feature_engineering_functions.items():
            test_features[column] = function(test_features)

        feature_engineered_dataset = test_features

        return feature_engineered_dataset

    # ---------- TESTS PASSED  ---------- 
    def preprocess_train(self,train_features:pd.DataFrame) -> pd.DataFrame:

        train_features = self.one_hot_encode_columns_train(train_features)
        train_features = self.min_max_scaled_columns_train(train_features)
        train_features = self.feature_engineering_train(train_features)

        preprocessed_dataset = train_features
        return preprocessed_dataset

    # ---------- TESTS PASSED  ---------- 
    def preprocess_test(self,test_features:pd.DataFrame) -> pd.DataFrame:

        test_features = self.one_hot_encode_columns_test(test_features)
        test_features = self.min_max_scaled_columns_test(test_features)
        test_features = self.feature_engineering_test(test_features)

        preprocessed_dataset = test_features

        return preprocessed_dataset