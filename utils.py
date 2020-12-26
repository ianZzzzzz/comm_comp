# preprocess of "log file to time series"
import pandas as pd
# import cudf as pd # nvidia GPU only # !pip install cudf 
from typing import List, Dict
from numpy import ndarray
from pandas import DataFrame
log_path = 'm'

log_np = load(
    log_path =log_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding = 'gbk',
    columns = ['date','time','area','upload_quantity_GB','download_quantity_gb' ]
    )
log_np = cut_nan(log_np,key_col = [0,1])
log_np = time_convert(log_np,merge = True,date_time_col = [0,1])
log_dict = to_dict(log_np)

def load(
    log_path: str,
    read_mode: str,
    return_mode: str,
    encoding='utf-8',
    columns=None
    )-> ndarray or DataFrame:
    '''读取csv文件 返回numpy数组'''
    #if read_mode == 'cudf':import cudf as pd
    if read_mode == 'pandas' :import pandas as pd
    
    log = pd.read_csv(log_path,encoding=encoding,names=columns)
    
    if return_mode == 'df':return log
    if return_mode == 'values':return log.values
def cut_nan(
    log: ndarray or DataFrame,
    key_col: List['col_number_1','col_number_2']
    )->ndarray:
    '''删除给定列中存在空值的日志行'''
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
    merge: bool,
    date_time_col:List['datetime'], 
    start_time = '2018/3/1T00:00:00'
    )->ndarray: 
    '''将日志数据的时间进行拼接转秒 秒到坐标部分未完成'''
    import numpy as np
    
    pandas_ = type(pd.DataFrame([]))
        # type(type(pd.DataFrame([]))) == <class 'type'> not a str :)
    if type(log)== pandas_ :
        log = log.values
    
    date_time_array = np.zeros(len(log),dtype=np.int32)
    # 存储一个月的秒数 需要2^22 存储一年的秒数需要2^25
    for row in range(len(log)):
        #1拼接
        if merge == True: 
            def merge_col(log: ndarray)-> str:
                date = log[0]
                time = log[1]
                date__time = log[date]+'T'+'0'+log[time] # '0' especially for mathor cup data
                return date__time
            date_time_a16 = merge_col(log[row,date_time_col]) 
            real_time = date_time_a16
            # 此处用一个type=a16变量暂存date_time字符串
            # WARNING 警告 两个字符串相加会因为原先的字符串类型位数不够 导致相加失败 但是不报错
            # date_time covered the date column
        else:   
            real_time = log[row,date_time]
        
        #2转化
        real_time = np.datetime64(real_time) 
        start_time = np.datetime64(start_time)
        #3转秒
            #(1)
        delta = int((real_time - start_time).item().total_seconds())
        date_time_array[row] = delta
            #(2)
        

        #4定位
    
    # 合并秒数列和log 清除原来date time 列
    new_log = np.concatenate(
         (log[:,[2,3,4]],date_time_array)
         ,axis = 1)
    return new_log

def to_dict(
    log: ndarray
)-> Dict[str,ndarray]:
    '''将日志转化为以各关键id为索引的字典'''
    log_dict = {}
    area_set = set(log[:,0])
    for area in area_set:
        mask = log[:,0]==area
        log_dict[area] = log[mask][:,[1,2,3]]
        
    return log_dict









