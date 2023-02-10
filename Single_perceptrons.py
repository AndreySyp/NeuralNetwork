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
        for i in range(len(s)):
            s[i] += w[0][i]
        ind_offset = 1

    for i in range(len(s)):
        for j in range(len(x)):
            s[i] += w[j + ind_offset][i] * x[j]

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
        for i in range(len(w[0])):
            w[0][i] += v * error[i]
        ind_offset = 1

    for i in range(ind_offset, len(w)):
        for j in range(len(w[i])):
            w[i][j] += v * error[j] * x[i - ind_offset]


def main():
    offset = True
    alpha = 1
    v = 0.9

    x = numpy.array([
        [0.000,	0.700],
        [0.50,	1.000],
        [0.700,	0.100],
        [0.200,	0.600],
        [1.000,	0.500],
        [0.600,	0.000],
        [0.200,	0.400],
        [0.700,	1.000],
        [0.900,	0.800],
        [0.500, 0.300]
    ])
    y = numpy.array([
        [0.000,    0.537],
        [0.265,    0.951],
        [0.971,    0.122],
        [0.235,    0.488],
        [1.000,    0.585],
        [0.941,    0.000],
        [0.353,    0.293],
        [0.441,    1.000],
        [0.735,    0.854],
        [0.676,    0.268]
    ])
    w = numpy.array([
        [0.000,    0.200],
        [-0.400,  -0.100],
        [0.300,    0.200]
    ])

    epoch = 10
    glob = 0

    for k in range(epoch):
        print(k)
        glob = 0
        for i in range(len(x)):
            neuron = neuron_state(x[i], w, offset)
            y_calc = sigmoid_logistic(neuron, alpha)
            error = y[i] - y_calc
            w_recalculation(x[i], w, error, v, offset)

            glob = glob + error[0]**2 + error[1]**2
            glob = glob / (len(y) * len(y[0]))
            glob = glob ** 0.5
            for i in w:
                for j in i:
                    print(f"{round(j, 3)}\t\t\t", end='')
            print()

        print(f"{glob}\n\n")
        v -= 0.05
        l = numpy.hstack([x, y])
        numpy.random.shuffle(l)
        l = numpy.array_split(l, len(l), 1)
        x = numpy.hstack(l[:2])
        y = numpy.hstack(l[2:])


if __name__ == "__main__":
    main()
