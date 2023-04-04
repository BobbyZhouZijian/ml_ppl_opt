import numpy as np


def ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


def holder(x, y):
    return - np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x * x + y * y) / np.pi)))


def eggholder(x, y):
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + y + 47))) - x * np.sin(np.sqrt(np.abs(x - y - 47)))


def model(params):
    x, y = params
    return ackley(x, y)
