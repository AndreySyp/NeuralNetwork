import data_worker
import numpy


def act(x, t):
    return numpy.array([-1 if i <= t else 1 for i in x])


def education_start(data: numpy.ndarray):
    """
    Начинает расчет сети Хопфилда
    :param data: Массив данных
    :return: Массив весовых коэффициентов и историю
    """

    w = numpy.dot(data.T, data)
    for i in range(len(w)):
        w[i][i] = 0
    return w


def applying(w: numpy.ndarray, data: numpy.ndarray):
    history = []  # Хранение данных для вывода
    y = [data]
    while True:
        s = numpy.dot(w, y[-1])
        yc = act(s, 0)

        history.append({"y": numpy.array(y), "S": s, "Condition_1": sum((yc - y[-1]) ** 2)})

        if sum((yc - y[-1]) ** 2) == 0:
            break

        if len(y) == 3:
            history[-1].update({"Condition_2": [sum((yc - y[-2]) ** 2), sum((y[-1] - y[-3]) ** 2)]})
            if sum((yc - y[-2]) ** 2) == 0 and sum((y[-1] - y[-3]) ** 2) == 0:
                break
            numpy.delete(y, 1)

        y = numpy.vstack([y, yc])

    return history


if __name__ == "__main__":
    array = data_worker.read("data\\met_norm_hop.csv")
    test = data_worker.read("data\\test_met_norm_hop.csv")

    ww = education_start(array)

    for ind, num in enumerate(test):
        if ind == 0:
            continue
        data_worker.print_history(applying(ww, num))
