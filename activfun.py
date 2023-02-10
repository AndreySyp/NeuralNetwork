"""
Модуль с различными функциями активации
"""
import numpy


def sigmoid_logistic(x: numpy.ndarray, alpha: float = 1.0) -> numpy.ndarray:
    """
    Применяет функцию активации ко всем элементам массива
    :param x: Массив значений
    :param alpha: Параметр насыщения
    :return: Массив измененных значений
    """
    return numpy.array([1 / (numpy.exp(-alpha * i) + 1) for i in x])
