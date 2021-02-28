def predict(img_path, module_dir, result_dir):
    '''
    Функция предикта карты сегментации стройки для 13 классов
    Входные параметры:
    - img_path - путь к изображению
    - module_dir - путь к папке с модулем
    - result_dir - путь к папке, куда сохранять предикт
    Выходные параметры:
    - pred_path - путь, куда сохранено изображение
    '''

    # Импорт модулей
    import torch, torchvision
    import detectron2
    import numpy as np
    import os, json, cv2, random, sys
    import detectron2.data.transforms as T
    from keras.models import load_model
    sys.path.append(module_dir)
    from utils import index2color, get_predict, correct_pixels

    # Задаем размеры предикта карты сегментации и количества классов
    height = 192
    width = 256
    num_classes = 13

    # Задаем цвета классов
    colors = [[100, 100, 100],
              [0, 0, 100],
              [0, 100, 0],
              [100, 0, 0],
              [0, 100, 100],
              [100, 0, 100],
              [100, 100, 0],
              [200, 200, 200],
              [0, 0, 200],
              [0, 200, 0],
              [200, 0, 0],
              [0, 200, 200],
              [200, 0, 200],
              [200, 200, 0],
              [0, 100, 200],
              [100, 0, 200],
              [0, 0, 0]]

    # Загружаем модели
    model = load_model(module_dir + '12_classes.h5')
    model_for_people = torch.load(module_dir + 'pytorch_model.pt')

    # Загружаем изображение для предикта
    img_for_people = cv2.imread(img_path)

    # Сохраняем размеры оригинального изображения
    original_h = img_for_people.shape[0]
    original_w = img_for_people.shape[1]

    # Препроцессинг для модели 12 классов
    img = cv2.resize(img_for_people, (width, height), interpolation=cv2.INTER_AREA)
    img = np.array(img)[:, :, ::-1] / 255
    # Получаем предикт
    pred, people_aura = get_predict(model, model_for_people, img, img_for_people, height, width)
    # Исправляем некоторые пиксели
    pred = correct_pixels(pred, people_aura, num_classes)
    # Переводим индексы классов в цвета
    pred = index2color(pred, num_classes, colors)
    # Возвращаем оригинальные размеры
    pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    # Формируем путь для сохранения изображения
    pred_path = result_dir + img_path[img_path.rindex('/'):img_path.rindex('.')] + '.jpg'
    # Сохраняем изображенияе
    cv2.imwrite(pred_path, np.uint8(pred[:, :, ::-1]))

    return pred_path
