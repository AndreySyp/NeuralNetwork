from activfun import sigmoid_logistic, diff_sigmoid_logistic
from generic_methods import neuron_state, w_recalculation
import data_worker
import numpy


def calculation_start(data: numpy.ndarray, epoch: int = 10, v: float = 0.9, y: int = 1, alpha: float = 1,
                      layers: int = 2):
    """
    Начинает расчет многослойной нейронной сети
    :param data: Совмещенный массив данных по x, y
    :param epoch: Количество эпох
    :param v: Коэффициент скорости обучения
    :param y: Количество выходных значений (y)
    :param alpha: Параметр насыщения
    :param layers: Количество слоев
    """
    data_worker.AMOUNT_Y = y  # Количество столбцов
    x, y = data_worker.array_splitting(data)  # Разделяем массив
    # w = numpy.random.uniform(low=-0.2, high=0.2, size=(len(x[0]) + 1, len(y[0])))  # Массив весов

    w = [
        numpy.array([
            [ 0.03,  0.02],  # x0
            [-0.08, -0.05],  # x1
            [-0.06, -0.01],  # x2
            [ 0.02,  0.05],  # x3
            [ 0.02, -0.04],  # x4
            [-0.08, -0.01],  # x5
            [-0.07,  0.01],  # x6
        ]),
        numpy.array([
            [0.04, 0.07],  # x0
            [0.01, 0.02],  # x1
            [0.04, 0.01],  # x2
        ])
    ]

    history = []  # Хранение данных для вывода

    for e in range(epoch):
        # Обнуление
        global_error = 0
        y_history = []
        w_history = []

        for ind, num in enumerate(x):
            neuron = []
            y_calc = []
            diff_y_calc = []
            d = []

            for i in range(layers):
                t = num if i == 0 else y_calc[i - 1]
                neuron.append(neuron_state(t, w[i]))
                y_calc.append(sigmoid_logistic(neuron[i], alpha))

            error = y[ind] - y_calc[1]

            for i in range(layers):
                diff_y_calc.append(diff_sigmoid_logistic(neuron[i], alpha))

            for i in range(layers - 1, 0 - 1, -1):
                t = error if i == layers - 1 else numpy.dot(w[i + 1][1:, ], d[-i])  # ???
                d.insert(0, t * diff_y_calc[i])

            for i in range(layers):
                t = num if i == 0 else diff_y_calc[i - 1]
                w_recalculation(t, w[i], d[i], v)

            # Информация для вывода в консоль
            global_error += sum(k ** 2 for k in error)
            y_history.append(numpy.hstack([y[ind], y_calc[1]]))
            w_history.append([])
            for i in w[0]:
                w_history[ind] = numpy.hstack([w_history[ind], i])
            for i in w[1]:
                w_history[ind] = numpy.hstack([w_history[ind], i])

        x, y = data_worker.array_reshuffle(x, y)

        history.append({"Global error": numpy.sqrt(global_error / (len(y) * len(y[0]))),
                        "Weight": numpy.array(w_history),
                        "Compare": numpy.around(numpy.array(y_history))})

    return [w, history]


if __name__ == "__main__":
    # print(numpy.array([2, 4])*numpy.array([2,3]))
    array = data_worker.read("data\\met_denorm_multi.csv")
    array = data_worker.normalization(array)
    calculation_start(array, y=2, epoch=100, v=0.5, alpha=2)
