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


def gaussian(x: numpy.ndarray, alpha: float = 1.0, t: float = 1) -> numpy.ndarray:
    """
    Применяет функцию активации ко всем элементам массива
    :param x: Массив значений
    :param alpha: Параметр насыщения
    :param t: Вертикальная ось симметрии
    :return: Массив значений в точка
    """
    return numpy.array([(numpy.exp(-alpha * abs(i - t))) for i in x])


def diff_gaussian(x: numpy.ndarray, alpha: float = 1.0, t: float = 1) -> numpy.ndarray:
    """
    Применяет функцию активации ко всем элементам массива
    :param x: Массив значений
    :param alpha: Параметр насыщения
    :param t: Вертикальная ось симметрии
    :return: Массив значений в точка
    """
    f = gaussian(x, alpha, t)
    return -2 * alpha * (x - t) * f


def single_jump(x, t):
    """
    Применяет функцию активации ко всем элементам массива
    :param x: Массив значений
    :param t: Параметр сдвига
    :return:
    """
    return numpy.array([-1 if i <= t else 1 for i in x])


def linear_threshold(x, t):
    """
    Применяет функцию активации ко всем элементам массива
    :param x: Массив значений
    :param t: Параметр сдвига
    :return:
    """
    return numpy.array([0 if i <= 0 else i if i <= t else t for i in x])
