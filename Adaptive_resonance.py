import data_worker
import numpy


def education_start_ART2(data: numpy.ndarray, v: float = 0.9, R: float = 0.8, w=None):
    """
    Начинает расчет сети Кохонена
    :param data: Массив данных
    :param v: Коэффициент скорости обучения
    :param R: Параметр сходства
    :param w: Весовые коэффициенты
    :return: Массив весовых коэффициентов и историю
    """
    history = []  # Хранение данных для вывода
    amount = []

    for ind, num in enumerate(data):
        data[ind] = num / ((sum(num ** 2)) ** 0.5)

    for ind_1, num_1 in enumerate(data):
        if w is None:
            w = numpy.array([data[0]])
            amount.append(1)
            history.append({"Weights": numpy.array(w), "R": "", "Amount": amount})
            continue

        r = numpy.zeros(len(w))
        for ind_2, num_2 in enumerate(w):
            r[ind_2] = sum(num_2 * num_1)

        if max(r) < R:
            w = numpy.vstack([w, num_1])
            amount.append(1)
        else:
            w[numpy.argmax(r)] = (1 - v) * w[numpy.argmax(r)] + v * num_1
            amount[numpy.argmax(r)] += 1

        history.append({"Weights": numpy.array(w), "R": r, "Amount": amount})

    return [w, history]


if __name__ == "__main__":
    array = data_worker.read("data\\met_norm_koh.csv")
    ww, h = education_start_ART2(array, v=0.5, R=0.8)
    data_worker.print_history(h)

