from activfun import sigmoid_logistic
from generic_methods import neuron_state, w_recalculation
import data_worker
import numpy


def education_start(data: numpy.ndarray, epoch: int = 10, v: float = 0.9, y: int = 1, alpha: float = 1,
                    dv: float = 0., w=None):
    """
    Начинает расчет однослойной нейронной сети
    :param data: Совмещенный массив данных по x, y
    :param epoch: Количество эпох
    :param v: Коэффициент скорости обучения
    :param dv: Изменение коэффициент скорости обучения
    :param y: Количество выходных значений (y)
    :param alpha: Параметр насыщения
    :param w: Весовые коэффициенты
    :return: Массив весовых коэффициентов и историю
    """
    data_worker.AMOUNT_Y = y  # Количество столбцов
    x, y = data_worker.array_splitting(data)  # Разделяем массив
    if w is None:
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
        v -= dv

        history.append({"Global error": numpy.sqrt(global_error / (len(y) * len(y[0]))),
                        "Weight": numpy.array(w_history),
                        "Compare": numpy.array(compare_y)})

    return [w, history]


def applying(w: numpy.ndarray, data: numpy.ndarray, alpha: float = 1):
    """
    Для практического использования
    :param w: Обученный массив весовых коэффициентов
    :param data: Вектор данных
    :param alpha: Параметр насыщения
    :return: нормированный выходной вектор
    """
    n = neuron_state(data, w)
    return sigmoid_logistic(n, alpha)


if __name__ == "__main__":
    array = data_worker.read("data\\met_denorm_single.csv")
    array, mm = data_worker.normalization(array)

    ww, h = education_start(array, y=2, epoch=100)
    data_worker.print_history(h)

    yc = applying(ww, numpy.array([0, 0.7]))
    yp = data_worker.denormalization(numpy.array([yc]), mm[:, 2:])
    print(f"В нормализованном виде = {yc}\nВ денормализованном виде{yp}")

