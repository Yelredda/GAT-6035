import numpy as np
import pandas as pd

# ---------- TESTS PASSED  ---------- 
def find_data_type(dataset:pd.DataFrame,column_name:str) -> np.dtype:
    return np.dtype( dataset[column_name].dtype ) 

# ---------- TESTS PASSED  ---------- 
# Set/unset column as index 
def set_index_col(dataset:pd.DataFrame,index:pd.Series) -> pd.DataFrame:
    return pd.DataFrame( dataset.set_index(index) )

# ---------- TESTS PASSED  ---------- 
def reset_index_col(dataset:pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame( dataset.reset_index(drop=True) )

# ---------- TESTS PASSED  ---------- 
# Set astype (string, int, datetime)
def set_col_type(dataset:pd.DataFrame,column_name:str,new_col_type:type) -> pd.DataFrame:
    # Original code passed tests even though it literally just returned the one column based on the error?
    dataset[column_name] = dataset[column_name].astype(new_col_type)
    return pd.DataFrame( dataset )

# ---------- TESTS PASSED  ---------- 
# Take Matrix of numbers and make it into a dataframe with column name and index numbering
def make_DF_from_2d_array(array_2d:np.array,column_name_list:list[str],index:pd.Series) -> pd.DataFrame:
    return pd.DataFrame( data=array_2d, columns=column_name_list, index=index ) # index=index... UGLY, cleanup?

# ---------- TESTS PASSED  ---------- 
# Sort Dataframe by values
def sort_DF_by_column(dataset:pd.DataFrame,column_name:str,descending:bool) -> pd.DataFrame:
    if (descending):    
        return pd.DataFrame( dataset.sort_values(by=column_name,ascending=False) )
    else:
        return pd.DataFrame( dataset.sort_values(by=column_name,ascending=True) )
 
# ---------- TESTS PASSED  ---------- 
# Drop NA values in dataframe Columns 
def drop_NA_cols(dataset:pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame( dataset.dropna(axis='columns') )

# ---------- TESTS PASSED  ---------- 
# Drop NA values in dataframe Rows 
def drop_NA_rows(dataset:pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame( dataset.dropna(axis='rows') )

# ---------- TESTS PASSED  ---------- 
def make_new_column(dataset:pd.DataFrame,new_column_name:str,new_column_value:list) -> pd.DataFrame:    
    dataset[new_column_name] = new_column_value # Cannot create and assign in the return statement, has to be separate
    return pd.DataFrame( dataset )

# ---------- TESTS PASSED  ---------- 
def left_merge_DFs_by_column(left_dataset:pd.DataFrame,right_dataset:pd.DataFrame,join_col_name:str) -> pd.DataFrame:
    return pd.DataFrame( left_dataset.merge(right_dataset, left_on=join_col_name, right_on=join_col_name) )

# ---------- TESTS PASSED  ---------- 
class simpleClass():

    # Nothing to do, just assign the parameters to the instance vars 
    def __init__(self, length:int, width:int, height:int):
        self.length = length
        self.width = width
        self.height = height
        pass


def find_dataset_statistics(dataset:pd.DataFrame,label_col:str) -> tuple[int,int,int,int,int]:

    # Only perc_positive should need to be casted as int... explicit casts for all just in case    
    n_records = int( dataset.shape[0] )
    n_columns = int( dataset.shape[1] )

    values = dataset[label_col].value_counts()

    n_negative = int( values[0] )
    n_positive = int( values[1] )
    perc_positive = int( n_positive / (n_positive + n_negative) * 100 )

    return n_records,n_columns,n_negative,n_positive,perc_positive