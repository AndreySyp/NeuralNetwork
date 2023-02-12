import numpy


mm = []
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


def normalization(array: numpy.ndarray) -> numpy.ndarray:
    """
    Линейно нормализует значения по каждому столбцу
    :param array: Массив, который требуется нормализовать
    :return: Нормализованный массив
    """
    result = []

    for i in range(len(array[0])):
        x = array[:, i]

        max_num = numpy.max(x)
        min_num = numpy.min(x)
        mm.append([min_num, max_num])

        result.append(function_normalization(x, min_num, max_num))

    return numpy.transpose(numpy.array(result))


def denormalization(array: numpy.ndarray, is_y: bool = False) -> numpy.ndarray:
    """
    Линейно денормализует значения массива
    :param array: Массив, который требуется денормализовать
    :param is_y: Считать ли только для y
    :return: Денормализованный массив
    """
    result = []
    offset = 0
    if is_y:
        offset = len(array[0]) - AMOUNT_Y

    for i in range(offset, len(array[0])):
        x = array[:, i]

        result.append(function_denormalization(x, mm[i][0], mm[i][1]))

    return numpy.transpose(numpy.array(result))


def array_splitting(array: numpy.ndarray) -> list[numpy.ndarray]:
    size = len(array[0]) - AMOUNT_Y
    array = numpy.array_split(array, len(array), 1)
    x = numpy.hstack(array[:size])
    y = numpy.hstack(array[size:])
    return [x, y]
