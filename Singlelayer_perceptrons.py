from activfun import sigmoid_logistic
from generic_methods import neuron_state, w_recalculation
import data_worker
import numpy


def calculation_start(data: numpy.ndarray, epoch: int = 10, v: float = 0.9, y: int = 1, alpha: float = 1):
    """
    Начинает расчет однослойной нейронной сети
    :param data: Совмещенный массив данных по x, y
    :param epoch: Количество эпох
    :param v: Коэффициент скорости обучения
    :param y: Количество выходных значений (y)
    :param alpha: Параметр насыщения
    """
    data_worker.AMOUNT_Y = y  # Количество столбцов
    x, y = data_worker.array_splitting(data)  # Разделяем массив
    w = numpy.random.uniform(low=-0.2, high=0.2, size=(len(x[0]) + 1, len(y[0])))  # Массив весов

    history = []  # Хранение данных для вывода

    for e in range(epoch):
        # Обнуление
        global_error = 0
        compare_y = []
        w_history = []

        for ind, num in enumerate(x):
            # Операции по алгоритму
            neuron = neuron_state(num, w)
            y_calc = sigmoid_logistic(neuron, alpha)
            error = y[ind] - y_calc
            w_recalculation(num, w, error, v)

            # Информация для вывода в консоль
            global_error += sum(k ** 2 for k in error)
            compare_y.append(numpy.hstack([y[ind], y_calc]))
            w_history.append([])
            for i in w:
                w_history[ind] = numpy.hstack([w_history[ind], i])

        # Перемешиваем значения
        x, y = data_worker.array_reshuffle(x, y)

        history.append({"Global error": numpy.sqrt(global_error / (len(y) * len(y[0]))),
                        "Weight": numpy.array(w_history),
                        "Compare": numpy.array(compare_y)})

    return [w, history]


if __name__ == "__main__":
    array = data_worker.read("data\\met_denorm_single.csv")
    array = data_worker.normalization(array)
    calculation_start(array, y=2, epoch=10)
