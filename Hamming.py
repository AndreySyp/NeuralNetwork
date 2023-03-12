import data_worker
import numpy


def act(x, t):
    return numpy.array([0 if i <= 0 else i if i <= t else t for i in x])


def education_start(x: numpy.ndarray):
    """
    Начинает расчет сети Хопфилда
    :param x: Входные образов
    :param y: Выходные образов
    :return: Массив весовых коэффициентов и историю
    """
    w = x / 2
    e = numpy.eye(len(x)) - 1 / len(x)
    for i in range(len(e)):
        e[i][i] = 1
    return [w, e]


def applying(w: list, x: numpy.ndarray, e_max: float = 0.1):
    history = []  # Хранение данных для вывода
    t = len(w[0][0]) / 2

    s = numpy.dot(w[0], x) + t
    yc = act(s, t)
    history.append({"s": numpy.array(s), "y": yc, "Condition_1": "-"})
    y = yc

    while True:
        s = numpy.dot(w[1], y)
        yc = act(s, t)
        history.append({"s": numpy.array(s), "y": yc, "Condition_1": sum((yc - y) ** 2)})
        if sum((yc - y) ** 2) <= e_max:
            break
        y = yc

    return history


if __name__ == "__main__":
    array = data_worker.read("data\\met_norm_ham.csv")
    test = data_worker.read("data\\test_met_norm_ham.csv")

    ww = education_start(array)

    for ind, num in enumerate(test):
        if ind == 1:
            continue
        data_worker.print_history(applying(ww, num))
