import os
import cv2
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class Data(object):
    def __init__(self):
        self.data_path='G:\yolo_tensorflow-master\yolo_tensorflow-master1\yolo_tensorflow-master\data\pascal_voc'
        self.batch_size=1
        self.index=0
        self.image_size=448
        self.img_path = os.path.join(self.data_path, 'JPEGImages')
        self.label_path = os.path.join(self.data_path, 'Annotations')
        self.img_list = os.listdir(self.img_path)
        self.img_list.sort()

    def load_image(self, image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0

        return image







