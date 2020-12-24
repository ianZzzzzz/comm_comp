# preprocess of "log file to time series"
import pandas as pd
# import cudf as pd # nvidia GPU only # !pip install cudf 
from typing import List, Dict
from numpy import ndarray
from pandas import DataFrame

def load(
    log_path: str,
    read_mode: str,
    return_mode: str,
    encoding='utf-8',
    columns=None
    )-> ndarray or DataFrame:
    #if read_mode == 'cudf':import cudf as pd
    if read_mode == 'pandas' :import pandas as pd
    
    log = pd.read_csv(log_path,encoding=encoding,names=columns)
    
    if return_mode == 'df':return log
    if return_mode == 'values':return log.values
def cut_nan(
    log: ndarray or DataFrame,
    key_col: List['col_number_1','col_number_2']
    )->ndarray:
    import numpy as np

    pandas_ = type(pd.DataFrame([]))
    numpy_  = type(pd.DataFrame([]).values)
        # type(type(pd.DataFrame([]))) == <class 'type'> not a str :)

    if type(log)== pandas_ :
        log = log.values
    if type(log)== numpy_:
        pass

    # nan row filter
    for i in key_col:
        mask = np.isnan(log[:,i])
        log = log[~mask]
    
    return log
def time_convert(
    log: ndarray or DataFrame,
    columns:List['date','time'], 
    merge: bool
    )->ndarray: 
    import numpy as np

    for col in columns:
     for row in range(len(log)):
            log[row,col]=

            log[row,col]= np.datetime64(log[row,col])
            start = log[row,2]
            course_id = log[row,3]
            end = c_dict[course_id][0]
            log[row,2] = int((start - end).item().total_seconds())
    pass
