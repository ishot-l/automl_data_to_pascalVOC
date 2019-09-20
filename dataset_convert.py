# 480px x 480px に変形
# Counter({'TRAIN': 301, 'TEST': 41, 'VALIDATION': 35})
# なので、VALIDATIONデータはすべてtrainに

import os, sys, glob
from PIL import Image
import pandas as pd
import pprint
import collections


IMAGE_SIZE = 480
DATASET_PATH = 'carbon_dataset_20190919_export_data-carbon_dataset-2019-09-19T00_58_48.108Z_image_object_detection_1.csv'
BASE_DATASET_PATH = "gs://na-no_tegaki_dataset/" # datasetがgoogle storageに入っていたときのPATH
OUTPUT_FOLDER_PATH = "_output/test-PascalVOC-export/"
IMAGE_SAVE_PATH = OUTPUT_FOLDER_PATH + "JPEGImages/"
XML_SAVE_PATH = OUTPUT_FOLDER_PATH + "Annotations/"
IMAGESETS_PATH = OUTPUT_FOLDER_PATH + "ImageSets/Main/"
LABELS = ('1' , '2', '3', '4', '5', '6', '7', '8', '9', '0', 'yen_mark')


# ImageSets内のファイル作成用
imagesets_train = {label: {} for label in LABELS}
imagesets_val = {label: {} for label in LABELS}

dataset = pd.read_csv(DATASET_PATH, header=None)

# 1. データ整理

annotation_dict = {}

for row in dataset.itertuples():

    # まず画像の名前を変更する、[フォルダ名]-[ファイル名].jpgにして、
    file_path = row[2].replace(BASE_DATASET_PATH, "")
    dirname, basename = os.path.split(file_path)
    
    if not file_path in annotation_dict:
        # このファイルを初めて読み込んだ場合
        annotation_dict[file_path] = {}
        annotation_dict[file_path]['filename'] = "{d}-{b}".format(d=dirname, b=basename)
        annotation_dict[file_path]['path'] = file_path
        annotation_dict[file_path]['data_type'] = row[1]
        annotation_dict[file_path]['objects'] = []

    # objectの追加
    new_object = {}
    new_object['name'] = row[3]

    x_pos = row[4::2]
    y_pos = row[5::2]
    new_object['bndbox'] = {'xmin': min(x_pos), 'xmax': max(x_pos), 'ymin': min(y_pos), 'ymax': max(y_pos)}

    annotation_dict[file_path]['objects'].append(new_object)


# 2. 画像変形、コピー

for key in annotation_dict:

    # 画像サイズを変換
    
    image = Image.open(key)

    width = image.width
    height = image.height

    if width > height:
        height = (IMAGE_SIZE / width) * height
        width = IMAGE_SIZE
    else:
        width = (IMAGE_SIZE / height) * width
        height = IMAGE_SIZE

    rgb_im = image.convert('RGB')
    rgb_im.thumbnail((IMAGE_SIZE, IMAGE_SIZE))

    back_ground = Image.new("RGB", (IMAGE_SIZE,IMAGE_SIZE), color=(255,255,255))
    back_ground.paste(rgb_im)

    back_ground.save(IMAGE_SAVE_PATH + annotation_dict[key]['filename'], quality=95, format='JPEG')

    label_count = {label: 0 for label in LABELS}

    # bounding boxの座標を変換
    for i, obj in enumerate(annotation_dict[key]['objects']):
        obj['bndbox']['xmin'] *= width
        obj['bndbox']['xmax'] *= width
        obj['bndbox']['ymin'] *= height
        obj['bndbox']['ymax'] *= height
        annotation_dict[key]['objects'][i] = obj
        label_count[obj['name']] += 1

    for label in LABELS:
        choose_set = imagesets_val if annotation_dict[key]['data_type'] == 'TEST' else imagesets_train
        if label_count[label] == 0:
            choose_set[label][annotation_dict[key]['filename']] = -1
        else:
            choose_set[label][annotation_dict[key]['filename']] = 1



# 3. データ整形、出力

## その準備
import xml.dom.minidom

# DOMオブジェクト作成
dom = xml.dom.minidom.Document()

def text_tag(tag, text):
    new_tag = dom.createElement(tag)
    new_tag.appendChild(dom.createTextNode(text))
    return new_tag

def tag_contain_tag(tag, child_tags):
    new_tag = dom.createElement(tag)
    for child_tag in child_tags:
        new_tag.appendChild(child_tag)
    return new_tag


for annotation in annotation_dict.values():

    # rootノードの生成と追加
    root = dom.createElement('annotation')
    root_attr = dom.createAttribute('verified')
    root_attr.value = 'yes'
    root.setAttributeNode(root_attr)

    root.appendChild(text_tag('folder', 'Annotation'))
    root.appendChild(text_tag('filename', annotation['filename']))
    root.appendChild(text_tag('path', 'test-PascalVOC-export/Annotations/' + annotation['filename']))
    root.appendChild(tag_contain_tag('source', [text_tag(tag='database', text='Unknown')]))
    sizes = [text_tag('width', str(IMAGE_SIZE)), text_tag('height', str(IMAGE_SIZE)), text_tag('depth', '3')]
    root.appendChild(tag_contain_tag('size', sizes))
    root.appendChild(text_tag('segmented', '0'))

    for obj in annotation['objects']:
        object_tag = dom.createElement('object')
        object_tag.appendChild(text_tag('name', obj['name']))
        object_tag.appendChild(text_tag('pose', 'Unspecified'))
        object_tag.appendChild(text_tag('truncated', '0'))
        object_tag.appendChild(text_tag('difficult', '0'))
        bndbox = [text_tag(k, str(v)) for k, v in obj['bndbox'].items()]
        object_tag.appendChild(tag_contain_tag('bndbox', bndbox))

        root.appendChild(object_tag)

    with open(XML_SAVE_PATH + os.path.splitext(annotation['filename'])[0]  + '.xml', 'w') as f:
        f.write(root.toprettyxml())


# 4. ImageSetsの出力
for key, value in imagesets_train.items():
    with open(IMAGESETS_PATH + key + "_train.txt", "w") as f:
        for k, v in value.items():
            f.write("{} {}\n".format(k, str(v)))

for key, value in imagesets_val.items():
    with open(IMAGESETS_PATH + key + "_val.txt", "w") as f:
        for k, v in value.items():
            f.write("{} {}\n".format(k, str(v)))