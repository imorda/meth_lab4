import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from regression import loss_funcs, loss_func, loss_funcs_sinus


def get_data():
    dataset = pd.read_csv("https://idas.expert/dataset_stud.csv")
    X = dataset["time_study"].to_numpy(dtype=np.float64)
    Y = dataset["Marks"].to_numpy(dtype=np.float64)
    assert len(X) == len(Y)
    return X, Y


def get_linear_loss_func(X, Y):
    students_loss_chunk = loss_funcs(X, Y)
    students_loss = loss_func(students_loss_chunk)
    return students_loss, students_loss_chunk


def get_nonlinear_loss_func(X, Y):
    weather_loss_chunk = loss_funcs_sinus(X, Y)
    weather_loss = loss_func(weather_loss_chunk)
    return weather_loss, weather_loss_chunk


def get_data_cgi():
    dataset = pd.read_csv("https://idas.expert/dataset_cgi.csv", delim_whitespace=True)
    # dataset.groupby(['GlucoseValue']).count()
    dataset = dataset.query("GlucoseValue == '1636-69-123'")
    dataset["InternalTime"] = pd.to_datetime(
        dataset["subjectId"].astype(str) + " " + dataset["InternalTime"]
    ).map(pd.Timestamp.timestamp)
    X = dataset["InternalTime"].to_numpy(dtype=np.float64)
    Y = dataset["DisplayTime"].to_numpy(dtype=np.float64)
    assert len(X) == len(Y)
    return X, Y


def get_data_weather():
    dataset = pd.read_csv("https://idas.expert/dataset_weather.csv", sep=";")
    dataset["DateTime"] = pd.to_datetime(dataset["DateTime"]).map(
        pd.Timestamp.timestamp
    )
    X = dataset["DateTime"].to_numpy(dtype=np.float64)
    Y = dataset["Средняя температура, С"].to_numpy(dtype=np.float64)
    assert len(X) == len(Y)
    return X, Y


class TorchDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.len = self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.len
