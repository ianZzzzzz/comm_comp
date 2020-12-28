'''
    程序功能：

        接收numpy arry .tolist()得到的list
        使用list中的数据列的头尾两行的时间值作为序列起始和终止时间点
        如果无序 则先排序
        根据起始和终止时间点以及通过EDA确定的采样频率
        确定时间序列坐标轴->time_list
        遍历输入的数据的时间列 
        将其填充到time_list
        对于缺失值填充有几种方式
            1 置0
            2 均值
        

'''
import numpy as np 
from numpy import ndarray
#log = [[332, 3646, 0],[33, 3146, 1],[33, 3146, 2],[33, 3146, 5]]
def time_map(log:ndarray)->ndarray:
   # log = np.array(log)
    time_col = log[:,2]

    start = time_col[0]
    end = time_col[-1]

    length = end-start+1
    data_width = 2

    ts = np.zeros((length,data_width),dtype = np.int32)

    for row in log:
        time = row[2] # 时间值 也就是ts中的位置
        ts[time,[0]] = row[0]
        ts[time,[1]] = row[1]
    return ts

for k,v in log_dict.items():
    v = time_map(v)
