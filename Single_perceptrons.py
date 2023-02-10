import filework
import numpy
from activfun import sigmoid_logistic


def neuron_state(x: numpy.ndarray, w: numpy.ndarray, offset: bool = False) -> numpy.ndarray:
    """
    Состояние нейрона
    :param x: Вектор сигналов
    :param w: Весовые коэффициенты
    :param offset: Есть ли коэффициенты смещения
    :return: массив состояний
    """
    s = numpy.zeros(2)
    ind_offset = 0

    if offset:
        s += w[0]
        ind_offset = 1

    s += numpy.dot(x, w[ind_offset:])

    return s


def w_recalculation(x: numpy.ndarray, w: numpy.ndarray, error: numpy.ndarray,
                    v: float = 9, offset_local: bool = False):
    """
    Коррекция старых значений весовых коэффициентов каждого нейрона
    :param x: Вектор сигналов
    :param w: Весовые коэффициенты
    :param error: Погрешности выходных значений
    :param v: Коэффициент скорости обучения
    :param offset_local: Есть ли коэффициенты смещения
    """
    ind_offset = 0
    if offset_local:
        w[0] += v * error
        ind_offset = 1

    for i in range(ind_offset, len(w)):
        w[i] += x[i - 1] * error * v


def main():
    filework.AMOUNT_Y = 2
    array = filework.read("data\\single perceptrons\\denorm.csv")
    array = filework.normalization(array)
    x, y = filework.array_splitting(array)

    w = numpy.array([
        [0.000,    0.200],
        [-0.400,  -0.100],
        [0.300,    0.200]
    ])

    offset = True
    epoch = 10
    alpha = 1
    v = 0.9

    for e in range(epoch):
        print(e)
        glob = 0
        for i, num in enumerate(x):
            neuron = neuron_state(num, w, offset)
            y_calc = sigmoid_logistic(neuron, alpha)
            error = y[i] - y_calc
            w_recalculation(num, w, error, v, offset)

            glob += sum(k ** 2 for k in error)

            for m in w:
                for j in m:
                    print(f"{round(j, 3)}\t\t\t", end='')
            print()

        glob = numpy.sqrt(glob / (len(y) * len(y[0])))
        print(f"{glob}\n\n")
        union = numpy.hstack([x, y])
        numpy.random.shuffle(union)
        x, y = filework.array_splitting(union)


if __name__ == "__main__":
    main()
