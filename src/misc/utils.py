import pandas as pd

def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  


def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])


def moving_average(data, window=5):
    """
    Computes a moving average used for reward trace smoothing.
    """
    data = pd.Series(data)
    mov_data = data.rolling(window=window).mean()
    return list(mov_data)


def moving_std(data, window=5):
    """
    Computes a moving standard deviation used for reward trace smoothing.
    """
    data = pd.Series(data)
    mov_data = data.rolling(window=window).std()
    return list(mov_data)
