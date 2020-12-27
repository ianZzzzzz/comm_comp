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
import pandas as pd
