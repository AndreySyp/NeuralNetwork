import activfun
import data_worker
import numpy

from Singlelayer_perceptrons import w_recalculation
from activfun import sigmoid_logistic


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


def calculation_start(data: numpy.ndarray, epoch: int = 10, v: float = 0.9, y: int = 1, alpha: float = 1):
    """
    Начинает расчет многослойной нейронной сети
    :param data: Совмещенный массив данных по x, y
    :param epoch: Количество эпох
    :param v: Коэффициент скорости обучения
    :param y: Количество выходных значений (y)
    :param alpha: Параметр насыщения
    """
    data_worker.AMOUNT_Y = y  # Количество столбцов
    x, y = data_worker.array_splitting(data)  # Разделяем массив
    # w = numpy.random.uniform(low=-0.2, high=0.2, size=(len(x[0]) + 1, len(y[0])))  # Массив весов

    # первый слой
    w_1 = numpy.array([
        [ 0.03,  0.02],  # x0
        [-0.08, -0.05],  # x1
        [-0.06, -0.01],  # x2
        [ 0.02,  0.05],  # x3
        [ 0.02, -0.04],  # x4
        [-0.08, -0.01],  # x5
        [-0.07,  0.01],  # x6
    ])    # y1    y2
    w_2 = numpy.array([
        [ 0.04,  0.07],  # x0
        [ 0.01,  0.02],  # x1
        [ 0.04,  0.01],  # x2
    ])    # y1    y2

    # Хранение данных для вывода
    global_error = 0
    y_history = []
    w_history = []
    alpha = 2
    for e in range(epoch):
        print(f"Epoch = {e}")  # Номер эпохи

        # Обнуление
        global_error = 0
        y_history = []
        w_history = []

        for ind, num in enumerate(x):
            neuron_1 = neuron_state(num, w_1)
            y_calc_1 = sigmoid_logistic(neuron_1, alpha)

            neuron_2 = neuron_state(y_calc_1, w_2)
            y_calc_2 = sigmoid_logistic(neuron_2, alpha)

            error = y[ind] - y_calc_2

            diff_y_calc_1 = activfun.diff_sigmoid_logistic(neuron_1)
            diff_y_calc_2 = activfun.diff_sigmoid_logistic(neuron_2)

            d_2 = error * diff_y_calc_2
            d_1 = numpy.dot(w_2[1:, ], numpy.array([d_2]).transpose()).transpose()[0] * diff_y_calc_1

            w_recalculation(diff_y_calc_2, w_2, d_2, v)
            w_recalculation(x[ind], w_1, d_1, v)

            print(ind)
            print(y_calc_2)


if __name__ == "__main__":
    array = data_worker.read("data\\multilayer perceptrons\\met_norm.csv")
    array = data_worker.normalization(array)
    calculation_start(array, y=2, epoch=100, v=0.5)
