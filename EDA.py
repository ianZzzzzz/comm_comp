import pandas as pd
import numpy as np
from typing import List, Dict
from pandas import DataFrame
from numpy import ndarray

raw = pd.read_csv('D:\\zyh\\data\\com_comp\\train\\train.csv',encoding='gbk')
raw = raw.rename(columns={
    '日期':'date',
    '时间':'time',
    '小区编号':'area',
    '上行业务量GB':'upload_quantity_GB',
    '下行业务量GB':'download_quantity_gb'})
mask = raw['area']==2
area_set = set(raw['area'])
#len(area_set) == 132279
len(raw[mask])
raw['datetime'] = raw['date']+raw['time'] # bug 时间和日期之间要有空格 
raw.head(5)
gb = raw.groupby('area')['datetime'].count()
gb.sort_values()