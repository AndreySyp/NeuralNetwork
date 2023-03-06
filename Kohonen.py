import data_worker
import numpy


def calculation_start(data: numpy.ndarray, epoch: int = 10, v: float = 0.9, clusters: int = 2, dv: float = 0., w=None):
    """
    Начинает расчет сети Кохонена
    :param data: Массив данных
    :param epoch: Количество эпох
    :param v: Коэффициент скорости обучения
    :param clusters: Количество кластеров
    :param dv: Изменение скорости обучения
    :param w: Весовые коэффициенты
    :return: Массив весовых коэффициентов и историю
    """

    # Массив весов
    if w is None:
        w = numpy.random.uniform(low=0., high=1, size=(clusters, len(data[0])))

    history = []  # Хранение данных для вывода

    for e in range(epoch):
        # Обнуление
        w_history = []

        for ind_1, num_1 in enumerate(data):
            # Считаем расстояние
            r = numpy.zeros(len(w))
            for ind_2, num_2 in enumerate(w):
                r[ind_2] = sum((num_2 - num_1) ** 2)

            # Выбираем победителя и изменяем у него веса
            w[numpy.argmin(r)] += v * (num_1 - w[numpy.argmin(r)])
            w_history.append(w)

        # Перемешиваем значения и меняем скорость обучения
        numpy.random.shuffle(data)
        v -= dv

        history.append({"Weight": numpy.array(w_history)})

    return [w, history]


if __name__ == "__main__":
    array = data_worker.read("data\\met_norm_koh.csv")
    ww = numpy.array([[0.20, 0.20, 0.30, 0.40, 0.40, 0.20, 0.50],
                  [0.20, 0.80, 0.70, 0.80, 0.70, 0.70, 0.80],
                  [0.80, 0.20, 0.50, 0.50, 0.40, 0.40, 0.40],
                  [0.80, 0.80, 0.60, 0.70, 0.70, 0.60, 0.70]])
    ww, h = calculation_start(array, epoch=6, v=0.3, clusters=4, w=ww, dv=0.05)
    data_worker.print_history(h)

    amount = [0, 0, 0, 0]
    for i_1, n_1 in enumerate(array):
        rr = numpy.zeros(len(ww))
        for i_2, n_2 in enumerate(ww):
            rr[i_2] = sum((n_2 - n_1) ** 2)
        amount[numpy.argmin(rr)] += 1
    print(amount)
