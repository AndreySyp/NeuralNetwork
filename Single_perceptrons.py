import data_worker
import numpy
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

    # Хранение данных для вывода
    global_error = 0
    y_history = []
    w_history = []

    for e in range(epoch):
        print(f"Epoch = {e}")  # Номер эпохи

        # Обнуление
        global_error = 0
        y_history = []
        w_history = []

        for ind, num in enumerate(x):
            # Операции по алгоритму
            neuron = neuron_state(num, w)
            y_calc = sigmoid_logistic(neuron, alpha)
            error = y[ind] - y_calc
            w_recalculation(num, w, error, v)

            # Информация для вывода в консоль
            global_error += sum(k ** 2 for k in error)
            y_history.append(numpy.hstack([y[ind], y_calc]))
            w_history.append([])
            for i in w:
                w_history[ind] = numpy.hstack([w_history[ind], i])

        # Перемешиваем значения
        union = numpy.hstack([x, y])
        numpy.random.shuffle(union)
        x, y = data_worker.array_splitting(union)

        # Информация для вывода в консоль
        global_error = numpy.sqrt(global_error / (len(y) * len(y[0])))
        print(f"Global error = {global_error}")
        print(f"Weight:\n{numpy.array(w_history)}")
        print("\n\n")

    print(f"Y on the last epoch:\n {numpy.array(y_history)}")


if __name__ == "__main__":
    array = data_worker.read("data\\single perceptrons\\met_norm.csv")
    array = data_worker.normalization(array)
    calculation_start(array, y=2, epoch=10)
