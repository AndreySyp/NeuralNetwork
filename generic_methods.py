"""
Методы, которые могут быть использованы к каждому типу нейронной сети
"""
import numpy
import data_worker


def neuron_state(x: numpy.ndarray, w: numpy.ndarray) -> numpy.ndarray:
    """
    Состояние нейрона
    :param x: Вектор сигналов
    :param w: Весовые коэффициенты
    :return: массив состояний
    """
    s = numpy.zeros(data_worker.AMOUNT_Y)
    s += w[0]
    s += numpy.dot(x, w[1:])

    return s


def w_recalculation(x: numpy.ndarray, w: numpy.ndarray, error: numpy.ndarray,
                    v: float = 9):
    """
    Коррекция старых значений весовых коэффициентов каждого нейрона
    :param x: Вектор сигналов
    :param w: Весовые коэффициенты
    :param error: Погрешности выходных значений
    :param v: Коэффициент скорости обучения
    """
    w[0] += v * error
    for i in range(1, len(w)):
        w[i] += x[i - 1] * error * v
