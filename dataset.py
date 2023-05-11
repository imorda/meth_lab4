import numpy as np
import pandas as pd

from regression import loss_funcs, loss_func


def get_data():
    dataset = pd.read_csv('https://idas.expert/dataset_stud.csv')
    X = dataset['time_study'].to_numpy(dtype=np.float64)
    Y = dataset['Marks'].to_numpy(dtype=np.float64)
    assert len(X) == len(Y)
    return X, Y


def get_linear_loss_func(X, Y):
    students_loss_chunk = loss_funcs(X, Y)
    students_loss = loss_func(X, Y)
    return students_loss, students_loss_chunk
