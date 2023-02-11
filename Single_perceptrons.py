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


def main(data: numpy.ndarray, amount_y: int = 1):

    data_worker.AMOUNT_Y = amount_y
    x, y = data_worker.array_splitting(data)
    w = numpy.random.uniform(low=-0.2, high=0.2, size=(len(x[0]) + 1, len(y[0])))

    epoch = 10
    alpha = 1
    v = 0.9

    for e in range(epoch):
        print(e)
        glob = 0
        y_list = []

        for i, num in enumerate(x):
            neuron = neuron_state(num, w)
            y_calc = sigmoid_logistic(neuron, alpha)
            error = y[i] - y_calc
            w_recalculation(num, w, error, v)

            glob += sum(k ** 2 for k in error)
            y_list.append(numpy.hstack([y[i], y_calc]))

            for m in w:
                for j in m:
                    print(f"{round(j, 3)}\t\t\t", end='')
            print()

        glob = numpy.sqrt(glob / (len(y) * len(y[0])))
        print(f"{glob}\n\n")
        union = numpy.hstack([x, y])
        numpy.random.shuffle(union)
        x, y = data_worker.array_splitting(union)

    print(numpy.array(y_list))


if __name__ == "__main__":
    array = data_worker.read("data\\single perceptrons\\my_data.csv")
    array = data_worker.normalization(array)
    main(array, 1)
