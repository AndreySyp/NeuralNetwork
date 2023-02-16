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
    data_worker.AMOUNT_Y = y  # Количество столбцов
    x, y = data_worker.array_splitting(data)  # Разделяем массив

    dote = [-2, -1, 0, 1, 2]
    h = numpy.zeros((len(x), len(dote)))
    for i in range(len(x)):
        for j in range(len(dote)):
            h[i][j] = activfun.gaussian(x[i], 0.22, dote[j])

    h_t = numpy.transpose(h)
    w = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(h_t, h)), h_t), y)
    print(w)


if __name__ == "__main__":
    array = data_worker.read("data\\met_denorm_radial.csv")
    # array = data_worker.normalization(array)
    calculation_start(array, y=1, epoch=10)
