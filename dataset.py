import numpy as np
import pandas as pd
import torch

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


def torch_linear_loss_func(X, Y):
    def loss_and_grad(w):
        w_torch = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        loss = sum((X[i]*w_torch[0] + w_torch[1] - Y[i])**2 for i in range(len(X)))
        loss.backward()
        return loss.detach().item(), w_torch.grad.detach().numpy()
    return loss_and_grad


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
