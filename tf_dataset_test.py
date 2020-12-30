
#train_examples 对minst数据集来说 内部是三位数组 也就是 一个样本数据是一个二维数组
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_examples, test_labels))


train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)

def func_timer(function):
    from functools import wraps
    import time
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print ('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer
@func_timer
def p(i):
    for x in range(0,i):
        print(x*10)
