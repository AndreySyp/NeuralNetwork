import activfun
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
    # x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # y = [1.00, 0.83, 0.71, 0.63, 0.56, 0.50]
    # d = [0, 0.5, 1]

    x = [-0.5,	-0.4,	-0.3,	-0.2,	-0.1,	0,	0.1,	0.2,	0.3,	0.4,	0.5]
    y = [0.111,	0.130,	0.149,	0.167,	0.184,	0.200,	0.216,	0.231,	0.245,	0.259,	0.273]
    d = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]

    r = 0.22
    alpha = 1 / (2 * r ** 2)

    h = []
    for ind_1, num_1 in enumerate(x):
        h.append([])
        for ind_2, num_2 in enumerate(d):
            h[ind_1].append(numpy.exp(-alpha * (num_1 - num_2) ** 2))
    h = numpy.array(h)
    print(numpy.around(h, 3))
    h_t = numpy.transpose(h)
    w = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(h_t, h)), h_t), numpy.transpose(y))
    print(numpy.around(w, 3))

    test = 0.35
    print(sum(([numpy.exp(-alpha * (test - num) ** 2) for ind, num in enumerate(d)] * w)))


if __name__ == "__main__":
    array = data_worker.read("data\\met_denorm_radial.csv")
    array = data_worker.normalization(array)
    calculation_start(array, y=1, epoch=10)
