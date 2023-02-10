import numpy


mm = []
AMOUNT_Y = 1


def function_normalization(x, min, max):
    return (x - min) / (max - min)


def function_denormalization(x, min, max):
    return min + x * (max - min)


def read(path: str, delimiter: str = ';') -> numpy.ndarray:
    """
    Чтение из файла
    :param path: Путь
    :param delimiter: Разделитель
    :return: Массив из файла
    """
    with open(path) as file_name:
        return numpy.loadtxt(file_name, delimiter=delimiter, dtype=float)


def normalization(array: numpy.ndarray) -> numpy.ndarray:
    """
    Линейно нормализует значения по каждому столбцу
    :param array: Массив, который требуется нормализовать
    :return: Нормализованный массив
    """
    result = []

    for i in range(len(array[0])):
        x = array[:, i]

        max = numpy.max(x)
        min = numpy.min(x)
        mm.append([min, max])

        result.append(function_normalization(x, min, max))

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


def array_splitting(array: numpy.ndarray):
    array = numpy.array_split(array, len(array), 1)
    x = numpy.hstack(array[:AMOUNT_Y])
    y = numpy.hstack(array[AMOUNT_Y:])
    return [x, y]
