import data_worker
import numpy


def calculation_start(data: numpy.ndarray, r: float = 1, c=None, y: int = 1):
    """
    Начинает расчет однослойной нейронной сети
    :param data: Совмещенный массив данных по x, y
    :param r: Радиус
    :param c: Центры
    :param y: Количество выходных значений (y)
    """
    data_worker.AMOUNT_Y = y  # Количество столбцов
    x, y = data_worker.array_splitting(data)  # Разделяем массив
    x = x.T[0]
    y = y.T[0]

    if c is None:
        c = [x[0], x[-1]]

    alpha = 1 / (2 * r ** 2)
    history = []
    h = []
    for ind_1, num_1 in enumerate(x):
        h.append([])
        for ind_2, num_2 in enumerate(c):
            h[ind_1].append(numpy.exp(-alpha * (num_1 - num_2) ** 2))

    h = numpy.array(h)
    h_t = numpy.transpose(h)
    w = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(h_t, h)), h_t), numpy.transpose(y))

    history.append({"H": h, "W": w})
    return [w, history]


if __name__ == "__main__":
    array = data_worker.read("data\\my_data_radial.csv")

    cc = [array[i][0] for i in range(0, len(array), 2)]
    ww, hh = calculation_start(array, y=1, r=0.22, c=cc)
    data_worker.print_history(hh)

    test = 0.35
    rr = 0.22
    a = 1 / (2 * rr ** 2)
    print(sum(([numpy.exp(-a * (test - num) ** 2) for ind, num in enumerate(cc)] * ww)))
