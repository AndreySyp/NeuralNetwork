from activation_functions import single_jump
import data_worker
import numpy


def education_start(x: numpy.ndarray, y: numpy.ndarray):
    """
    Начинает расчет сети Хопфилда
    :param x: Входные образов
    :param y: Выходные образов
    :return: Массив весовых коэффициентов и историю
    """
    w = numpy.dot(x.T, y)
    return w


def applying(w: numpy.ndarray, x: numpy.ndarray):
    """
    Для практического использования
    :param w: Обученный массив весовых коэффициентов
    :param x: Вектор данных
    :return: Выходной y и история
    """
    history = []  # Хранение данных для вывода
    ss = numpy.zeros(len(w[0]))
    while True:
        s_1 = numpy.dot(w.T, x)
        yc_1 = single_jump(s_1, 0)

        s_2 = numpy.dot(w, yc_1)
        x = single_jump(s_2, 0)

        history.append({"y_1": numpy.array(yc_1), "S_1": s_1,
                        "y_2": numpy.array(x), "S_2": s_2, "Condition_1": sum((yc_1 - ss) ** 2)})

        if sum((yc_1 - ss) ** 2) == 0:
            break
        ss = yc_1

    return [yc_1, history]


if __name__ == "__main__":
    array_in = data_worker.read("../data/met_norm_kos_input.csv")
    array_out = data_worker.read("../data/met_norm_kos_output.csv")

    test = data_worker.read("../data/test_met_norm_kos.csv")

    ww = education_start(array_in, array_out)
    yy, h = applying(ww, test)

    data_worker.print_history(h)
    print(f"Соответствует образу {numpy.where (array_out == yy )[0][0]}" if yy in array_out else "Не соответствует")
