import numpy as np
import detectron2.data.transforms as T
import torch
import cv2


def index2color(numpy_grid, num_classes, colors):
    '''
    Функция перевода индекса в RGB значение
    Входные параметры:
      - numpy_grid - 2D матрица значений индексов
      - num_classes - количество классов сегментации
      - colors - цвета классов в RGB  формате
    Выходные параметры:
      - output - RGB изображение карты сегментации
    '''

    index = numpy_grid.flatten()

    h = numpy_grid.shape[0]
    w = numpy_grid.shape[1]
    output = np.zeros(shape=(h * w, 3))

    for i in range(num_classes):
        indices = np.where(index == i)[0]
        output[indices] = colors[i]

    return output.reshape((h, w, 3))


def get_predict(model, model_for_people, img, img_for_people, height, width):
    '''
    Функция для получения предсказания обоих сетей и их обобщения
    Входные парамерты:
      - model - модель для сегментации 12 классов
      - model_for_people - модель для сегментации людей
      - img - изображение для сегментации 12 классов (192х256х3)
      - img_for_people - изображение для сегментации людей (произвольный размер, bgr)
      - height, width - высота и ширина резульата сегментации
    Выходные параметры:
      - pred - карта сегментации 13 классов
      - people_aura - обведенный на 1 пиксель контур класса человек
    '''

    prediction = model.predict(np.expand_dims(img, axis=0))[0]  # Предикт модели на 12 классов
    aug = T.ResizeShortestEdge([480, 480], 640)  # Настройка препроцессинга для модели сегментации людей
    image = aug.get_transform(img_for_people).apply_image(img_for_people)  # Решейп и нормализация
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # Приводим к виду для подачи в pytorch модель
    inputs = {"image": image, "height": height, "width": width}  # Формат инпута для модели detectron2
    people_prediction = model_for_people([inputs])[0]  # Предикт модели detectron2

    people_mask = np.zeros((height, width))  # Формирования заготовки под карту сегментации людей

    class_indices = people_prediction[
        "instances"].pred_classes.cpu().numpy()  # Берем индексы классов для элементов предикта
    mask_result = people_prediction[
        "instances"].pred_masks.cpu().numpy()  # Берем карты сегментации для каждого предикта
    # Оставляем только те карты сегментации, где есть люди
    indices = np.where(class_indices == 0)[0]
    for ind in indices:
        people_mask += mask_result[ind]

    people_mask = np.clip(people_mask, 0, 1)  # Приводим все значения к диапазону [0,1]
    # Получаем контур людей
    kernel = np.ones((3, 3), np.uint8)
    people_aura = cv2.dilate(people_mask, kernel, iterations=1) - people_mask
    # Поскольку в модели 12 классов люди были инвентарь, то удаляем из окружающей людей области предикт инвентаря
    # (чтобы не было вокруг людей синих пикселей)
    prediction[:, :, 8] = prediction[:, :, 8] - people_aura

    # Делаем тоже самое для людей
    people_mask = np.expand_dims(people_mask, axis=-1) + 0.1
    people_mask[:, :, 0] = people_mask[:, :, 0] - people_aura

    # Объединяем предикты и берем максимальные значения по каналам
    prediction = np.concatenate((prediction, people_mask), axis=2)
    pred = np.argmax(prediction, axis=2)

    return pred, people_aura


def correct_pixels(pred, people_aura, num_classes):
    '''
    Функция для корректировки значений пикселей
    Входные параметры:
      - pred - карта сегментации после предсказаний
      - people_aura - контуры людей
      - num_classes - количество классов сегментации
    '''

    # Позиции для обхода вокруг пикселя
    actions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    # Точки вогруг человека, где надо скорректировать значения пикселей
    aura_points = np.where(people_aura > 0)

    # Поскольку удаляли в ближайшем котуре человека пиксели класса инвентарь,
    # то возвращяем это значение, если в окружении пикселя контура присутствует больше
    # 2 пикселей класса инвентарь
    for i in range(2):
        for y, x in zip(aura_points[0], aura_points[1]):
            point_surrounding = {k: 0 for k in range(num_classes)}
            for act in actions:
                y1 = np.clip(y + act[0], 0, pred.shape[0] - 1)
                x1 = np.clip(x + act[1], 0, pred.shape[1] - 1)
                point_surrounding[pred[y1, x1]] += 1

            if point_surrounding[8] > 2 and point_surrounding[12] > 0:
                pred[y, x] = 8

    # Удаляем единичные пиксели определенных классов на изображении
    for y in range(1, pred.shape[0] - 1):
        for x in range(1, pred.shape[1] - 1):
            point_surrounding = {k: 0 for k in range(num_classes)}

            for act in actions:
                y1 = y + act[0]
                x1 = x + act[1]
                point_surrounding[pred[y1, x1]] += 1

            if point_surrounding[pred[y, x]] < 2 and pred[y, x] not in [3, 4, 5, 10]:
                pred[y, x] = [k for k, v in sorted(point_surrounding.items(), key=lambda item: item[1], reverse=True)][
                    0]

    return pred
