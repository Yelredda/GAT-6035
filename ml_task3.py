import numpy as np
import pandas as pd
import sklearn.cluster
import yellowbrick.cluster

class KmeansClustering:
    def __init__(self, 
                 random_state: int
                ):

        self.random_state = random_state

        # Will be shared between test and train
        self.kmeans_model = sklearn.cluster.KMeans(n_init=10, random_state=self.random_state) 
        self.visualizer = yellowbrick.cluster.KElbowVisualizer(self.kmeans_model, k=(1,10) )

        pass

    # ---------- TESTS PASSED  ----------     
    def kmeans_train(self,train_features:pd.DataFrame) -> list:

        # Create kmeans model and KElbow visualizer pushed out to __init__ for sharing
        
        # Perform initial visualizer fit
        self.visualizer.fit(train_features)
        
        # Reinit the kmeans_model using the optimal n_clusters (is it better to create a new separate model? I don't see an issue w/ the reinit)
        self.kmeans_model = sklearn.cluster.KMeans(n_init=10, random_state=self.random_state, n_clusters=self.visualizer.elbow_value_)
        
        # Train to the data
        self.kmeans_model.fit(train_features)

        # Output cluster IDs 
        cluster_ids = self.kmeans_model.labels_.tolist()
        return cluster_ids

    # ---------- TESTS PASSED  ----------  
    def kmeans_test(self,test_features:pd.DataFrame) -> list:
                
        # Everything is already done in train; predict and report
        test_clusters = self.kmeans_model.predict(test_features)

        # Output cluster IDs 
        cluster_ids = test_clusters.tolist()
        return cluster_ids
    
    # ---------- TESTS PASSED  ----------  
    def train_add_kmeans_cluster_id_feature(self,train_features:pd.DataFrame) -> pd.DataFrame:

        # Call previous function, add to 'kmeans_cluster_id'
        train_features['kmeans_cluster_id'] = self.kmeans_train(train_features)

        # Output dataframe with new column
        output_df = train_features
        return output_df

    # ---------- TESTS PASSED  ----------  
    def test_add_kmeans_cluster_id_feature(self,test_features:pd.DataFrame) -> pd.DataFrame:

        # Call previous function, add to 'kmeans_cluster_id'
        test_features['kmeans_cluster_id'] = self.kmeans_test(test_features)

        # Output dataframe with new column
        output_df = test_features
        return output_df