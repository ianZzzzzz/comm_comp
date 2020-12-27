'''
    程序功能：

        读取通信流量数据日志文件
        对时间与日期列进行拼接 
        以第一行数据为起始时间 
        以秒为单位计算距离起始时间的距离 作为时间序列索引
        对以GB为单位的数据*1024转为MB单位
        以字典方式存储各个区域下的日志

'''

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
    encoding_='utf-8',
    columns=None
    )-> ndarray or DataFrame:
    '''读取csv文件 返回numpy数组'''
    #if read_mode == 'cudf':import cudf as pd
    if read_mode == 'pandas' :
        import pandas as pd
        log = pd.read_csv(log_path,encoding=encoding_,names=columns,)
        print('load running!')
    if return_mode == 'df':return log
    if return_mode == 'values':return log.values
def cut_nan(
    log,
    key_col:list
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
        mask = np.isnan(log[:,i]) # bug
        log = log[~mask]
    
    return log
def time_convert(
    log: ndarray ,
    merge,
    date_time_col:List['datetime'], 
    start_time = '2018-03-01T00:00:00'
    )->ndarray: 
    '''将日志数据的时间进行拼接转秒 秒到坐标部分未完成'''
    import numpy as np
    pandas_ = type(pd.DataFrame([]))
        # type(type(pd.DataFrame([]))) == <class 'type'> not a str :)
    if type(log)== pandas_ :
        log = log.values
    
    date_time_array = np.zeros((len(log),1),dtype=np.int32)
    # 存储一个月的秒数 需要2^22 存储一年的秒数需要2^25
    print('time convert running!')
    
    for row in range(len(log)):
        if (row%10000)==0:print('row : ',row)
        #1拼接
        if merge == True: 
            def merge_col(log: ndarray)-> str:
                
                date = log[0].replace('/','-')
                # yyyy-m-d -> yyyy-mm-dd 以适应numpy严格的格式要求
                if len(date)<10:
                    if len(date) == 8:
                        date = date[0:5]+'0'+date[5:7]+'0'+date[7]
                    else:
                       # print('date:',date)
                        if date[6]=='-':
                            date = date[0:5]+'0'+date[5:]
                        else:
                            date = date[0:8]+'0'+date[8]
                time = log[1]
                if time[0]=='0':time = '0'+time
                date__time = date+'T'+time
                return date__time
            date_time_a16 = merge_col(log[row,date_time_col]) 
            real_time = date_time_a16
           # print('real_time:',real_time)
            # 此处用一个type=a16变量暂存date_time字符串
            # WARNING 警告 两个字符串相加会因为原先的字符串类型位数不够 导致相加失败 但是不报错
            # date_time covered the date column
        else:   
            real_time = log[row,date_time_col]
        
        #2转化
        
        real_time = np.datetime64(real_time) 
        start_time = np.datetime64(start_time)
        #print('时间转化完成')
        #3转秒
            #(1)
        delta = int((real_time - start_time).item().total_seconds())
        date_time_array[row] = delta
        #print('转秒完成')
            #(2)
        #4定位
    
    # 合并秒数列和log 清除原来date time 列
    print(
        'len date_time_array: ',len(date_time_array)
        ,'len log:',len(log[:,[2,3,4]]))
    new_log = np.concatenate(
         (log[:,[2,3,4]],date_time_array)
         ,axis = 1)
    print('concat finish!')
    # 类型转换
    new_log[:,0] = new_log[:,0].astype('u4')
    new_log[:,1] = ((new_log[:,1].astype('f2'))*1024).astype('u2')
    new_log[:,2] = ((new_log[:,2].astype('f2'))*1024).astype('u2')
    new_log[:,3] = new_log[:,3].astype('u4')
    print('astype finish')
    return new_log
def to_dict(
    log: ndarray
)-> Dict[str,ndarray]:
    '''将日志转化为以各关键id为索引的字典'''
    print('to_dict running!')
    log_dict = {}
    area_set = set(log[:,0])
    i = 0
    for area in area_set:
        i+=1
        mask = log[:,0]==area
        log_dict[area] = log[mask][:,[1,2,3]]
        if (i%10000)==0:print('already dict  ',i,' areas')
        
    return log_dict

log_path = 'D:\\zyh\\data\\com_comp\\train\\test2.csv'
log_np = load(
    log_path =log_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns = ['date','time','area','upload_quantity_GB','download_quantity_gb' ]
    )
#log_np = cut_nan(log_np[1:5,:],key_col = [0,1])
log_np = time_convert(log_np[1:,:],merge = True,date_time_col = [0,1])
log_dict = to_dict(log_np)




