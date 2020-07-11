import xml.etree.ElementTree as ET
from os import getcwd


def get_classes(path):
    with open(path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def convert_annotation(years, image_num, files):
    in_file = open(f"VOCdevkit/VOC{years}/Annotations/{image_num}.xml")
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        box_axis = (int(xmlbox.find('xmin').text),
                    int(xmlbox.find('ymin').text),
                    int(xmlbox.find('xmax').text),
                    int(xmlbox.find('ymax').text))
        files.write(" " + ",".join([str(value) for value in box_axis]) + ',' + str(cls_id))


if __name__ == '__main__':
    sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    classes_path = "./data/voc_classes.txt"
    classes = get_classes(classes_path)
    wd = getcwd()

    for year, image_set in sets:
        image_ids = open(f"VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt").read().strip().split()
        list_file = open(f"data/{year}_{image_set}.txt", 'w')
        for image_id in image_ids:
            list_file.write(f"{wd}/VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg")
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()

