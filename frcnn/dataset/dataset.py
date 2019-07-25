import tensorflow as tf
import numpy as np
import os
from xml.etree import ElementTree as ET
def fake_dataset():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    girl = tf.image.decode_png(open(os.path.join(src_dir, 'girl.png'), 'rb').read(), channels=3)
    horse = tf.image.decode_jpeg(open(os.path.join(src_dir, 'horse.jpg'), 'rb').read(), channels=3)

    # y1, x1, y2, x2, cls
    girl_bb = tf.convert_to_tensor([[0.03049111, 0.18494931, 0.96302897, 0.9435849, 0],
                                       [0.35938117, 0.01586703, 0.6069674, 0.17582396, 56],
                                       [0.48252046, 0.09158827, 0.6403017, 0.26967454, 67]])

    horse_bb = tf.convert_to_tensor([
        [0.19683257918552036, 0.106, 0.9502262443438914, 0.942, 13],
        [0.09954751131221719, 0.316, 0.3778280542986425, 0.578, 15]
    ])

    def _data_generator():
        yield girl, girl_bb
        yield horse, horse_bb

    dataset = tf.data.Dataset.from_generator(_data_generator, (tf.int32, tf.float32), ((None, None, 3), (None, 5)))
    return dataset

def _load_id_list(path):
    with open(path, 'r') as f:
        ret = [line.rstrip('\n') for line in f]
    return ret

def _parse_classes(path):
    files = os.listdir(path)
    filtered = filter(lambda file: '_train.txt' in file or '_trainval.txt' in file or '_text.txt' in file, files)
    return set(map(lambda file: '_'.join(file.split('_')[:-1]), filtered))

def voc_dataset(data_path, file_name):
    dirs = os.listdir(data_path)
    years = list(filter(lambda dir: dir in ['VOC2007', 'VOC2012'], dirs))
    year_id_list = []
    class_list  = set()
    for year in years:
        id_list_file = os.path.join(data_path, year, 'ImageSets', 'Main', file_name)
        id_list = _load_id_list(id_list_file)
        class_list = class_list.union(_parse_classes(os.path.join(data_path, year, 'ImageSets', 'Main')))
        year_id_list += [(year, id) for id in id_list]
    class_list = sorted(class_list)
    def _data_generator():
        for year, id in year_id_list:
            img_path = os.path.join(data_path, year, 'JPEGImages', '{}.jpg'.format(id))
            anno_path = os.path.join(data_path, year, 'Annotations', '{}.xml'.format(id))
            img = tf.image.decode_png(open(img_path, 'rb').read(), channels=3)

            bboxes = []
            anno = ET.parse(anno_path)
            size = anno.getroot().find('size')
            W = float(size.find('width').text)
            H = float(size.find('height').text)
            for obj in anno.getroot().findall('object'):
                cls = obj.find('name').text
                bbox = obj.find('bndbox')
                y1 = float(bbox.find('ymin').text)
                x1 = float(bbox.find('xmin').text)
                y2 = float(bbox.find('ymax').text)
                x2 = float(bbox.find('xmax').text)
                cls_idx = class_list.index(cls)
                bboxes.append([y1/H, x1/W, y2/H, x2/W, cls_idx])
            yield img, np.array(bboxes)
    dataset = tf.data.Dataset.from_generator(_data_generator, (tf.int32, tf.float32), ((None, None, 3), (None, 5)))
    dataset = dataset.prefetch(16)
    return dataset
#
# if __name__ == '__main__':
#     data_path = '/dataset/voc_train/'
#     file_name = 'train.txt'
#     dirs = os.listdir(data_path)
#     years = list(filter(lambda dir: dir in ['VOC2007', 'VOC2012'], dirs))
#     year_id_list = []
#     class_list  = set()
#     for year in years:
#         id_list_file = os.path.join(data_path, year, 'ImageSets', 'Main', file_name)
#         class_list = _parse_classes(os.path.join(data_path, year, 'ImageSets', 'Main'))
#         id_list = _load_id_list(id_list_file)
#         year_id_list += [(year, id) for id in id_list]
#     class_list = sorted(class_list)
#     def _data_generator():
#         for year, id in year_id_list:
#             img_path = os.path.join(data_path, year, 'JPEGImages', '{}.jpg'.format(id))
#             anno_path = os.path.join(data_path, year, 'Annotations', '{}.xml'.format(id))
#             img = tf.image.decode_png(open(img_path, 'rb').read(), channels=3)
#
#             bboxes = []
#             anno = ET.parse(anno_path)
#             size = anno.getroot().find('size')
#             W = float(size.find('width').text)
#             H = float(size.find('height').text)
#             for obj in anno.getroot().findall('object'):
#                 cls = obj.find('name').text
#                 bbox = obj.find('bndbox')
#                 y1 = float(bbox.find('ymin').text)
#                 x1 = float(bbox.find('xmin').text)
#                 y2 = float(bbox.find('ymax').text)
#                 x2 = float(bbox.find('xmax').text)
#                 cls_idx = class_list.index(cls)
#                 bboxes.append([y1/H, x1/W, y2/H, x2/W, cls_idx])
#             yield img, np.array(bboxes)
#     get = _data_generator()
#     a = next(get)
#     a = next(get)
#     a = next(get)
#     a = next(get)