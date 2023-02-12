"""
Модуль с различными функциями активации
"""
import numpy


def sigmoid_logistic(x: numpy.ndarray, alpha: float = 1.0) -> numpy.ndarray:
    """
    Применяет функцию активации ко всем элементам массива
    :param x: Массив значений
    :param alpha: Параметр насыщения
    :return: Массив значений в точках
    """
    return numpy.array([1 / (numpy.exp(-alpha * i) + 1) for i in x])


def diff_sigmoid_logistic(x: numpy.ndarray, alpha: float = 1.0) -> numpy.ndarray:
    """
    Считает производную в точке для всех элементов
    :param x: Массив значений
    :param alpha: Параметр насыщения
    :return: Массив производных в точках
    """
    f = sigmoid_logistic(x, alpha)
    return alpha * f * (1 - f)
