import numpy


AMOUNT_Y = 0


def read(path: str, delimiter: str = ';') -> numpy.ndarray:
    """
    Чтение из файла
    :param path: Путь
    :param delimiter: Разделитель
    :return: Массив из файла
    """
    with open(path) as file_name:
        return numpy.loadtxt(file_name, delimiter=delimiter, dtype=float)


def function_normalization(x, min_num, max_num):
    return (x - min_num) / (max_num - min_num)


def function_denormalization(x, min_num, max_num):
    return min_num + x * (max_num - min_num)


def normalization(array: numpy.ndarray):
    """
    Линейно нормализует значения по каждому столбцу
    :param array: Массив, который требуется нормализовать
    :return: Нормализованный массив и массив с max и min значениями
    """
    result = []
    min_max = []

    for i in range(len(array[0])):
        x = array[:, i]
        max_num = numpy.max(x)
        min_num = numpy.min(x)
        min_max.append([min_num, max_num])
        result.append(function_normalization(x, min_num, max_num))

    return [numpy.transpose(numpy.array(result)), numpy.transpose(numpy.array(min_max))]


def denormalization(array: numpy.ndarray, min_max: numpy.ndarray) -> numpy.ndarray:
    """
    Линейно денормализует значения массива
    :param array: Массив, который требуется денормализовать
    :param min_max: Массив с max и min значениями
    :return: Денормализованный массив
    """
    result = []
    for ind, num in enumerate(array):
        result.append(function_denormalization(num, min_max[0], min_max[1]))

    return numpy.array(result)


def array_splitting(array: numpy.ndarray) -> list[numpy.ndarray]:
    """
    Разбивает один массив на два
    :param array: Исходных массив
    :return: Разбитый массив
    """
    size = len(array[0]) - AMOUNT_Y
    array = numpy.array_split(array, len(array), 1)
    x = numpy.hstack(array[:size])
    y = numpy.hstack(array[size:])
    return [x, y]


def array_reshuffle(array_1: numpy.ndarray, array_2: numpy.ndarray) -> list[numpy.ndarray]:
    """
    Принимает два массива и перемешивает строки
    :param array_1: Массив 1
    :param array_2: Массив 2
    :return: Перемешанный массив
    """
    union = numpy.hstack([array_1, array_2])
    numpy.random.shuffle(union)
    return array_splitting(union)


def print_history(h):
    for ind, num in enumerate(h):
        print(f"Эпоха: {ind}")
        for key, value in num.items():
            if key == "Weight":
                print(f"{key}:\n {value[-1]}")
                continue
            print(f"{key}:\n {value}")
        print("\n")
