"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import cv2 as cvbbox_params
import requests
#from .voc_eval import voc_eval
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC_CLASSES = ( '__background__', # always index 0
    'person')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class VOCSegmentation(data.Dataset):

    """VOC Segmentation Dataset Object
    input and target are both images

    NOTE: need to address https://github.com/pytorch/vision/issues/9

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg: 'train', 'val', 'test').
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target image
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='VOC2007'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join(
            self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(
            self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(
            self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id).convert('RGB')
        img = Image.open(self._imgpath % img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5)) 
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        # self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', '%s.jpg')
        self.ids = list()
        path_list=os.listdir(self.root)
        path_list.sort()
        for filename in path_list:
            self.ids.append((self.root, filename[0:6]))
        #for root,dirs,files in os.walk(self.root+'/JPEGImages'):
        #    for file_name in files:
        #        self.ids.append((self.root, file_name[0:6]))
        #for (year, name) in image_sets:
        #    self._year = year
        #    rootpath = os.path.join(self.root, 'VOC' + year)
        #    for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #        self.ids.append((rootpath, line.strip()))
        

    def __getitem__(self, index):
        img_id = self.ids[index]
        # target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        # if self.target_transform is not None:
        #     target = self.target_transform(target)


        # if self.preproc is not None:
        #     img, target = self.preproc(img, target)
            #print(img.size())

                    # target = self.target_transform(target, width, height)
        #print(target.shape)

        return img

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir, save_path, start_time, device_ip, num_th, bbox_params):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes, save_path, start_time, device_ip, num_th, bbox_params)
        # self._do_python_eval(output_dir) 跳过精度计算

    def _get_voc_results_file_template(self, save_path):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(
            save_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, save_path, start_time, device_ip, num_th, bbox_params):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            truth_count = 0
            if cls == '__background__':
                continue
            time_start = round(float(self.ids[0][1]))
            time_tmp = time_start
            time_list = [[time_start]]
            for im_ind, index in enumerate(self.ids):
                index = index[1]
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                if (self._paint_box(index, dets, bbox_params, save_path) == True):
                    truth_count = truth_count + 1
                    index_num = round(float(index))
                    if (index_num < time_tmp + 60):
                        time_tmp = index_num
                        time_list[len(time_list)-1].append(index_num);
                    else:
                        time_list.append([index_num])
                        time_tmp = index_num
            time_pair = []
            for array in time_list:
                if (array[-1] - array[0] > 30):
                    time_pair.append([array[0], array[-1]])
                        

            # if (truth_count > int(num_th)):
            #     print('有消毒操作', truth_count, 'ip', device_ip)
            # else:
            #     print('无消毒操作', truth_count, 'ip', device_ip)
            url = 'http://localhost:9090/callback'
            body = {"device_ip": device_ip, "result": truth_count > int(num_th), "count": truth_count, "time_pair": time_pair}
            print(body)
            headers = {'content-type': "application/json"}
            # response = requests.post(url, data=json.dumps(body), headers=headers)
            # print(response)

    
    def _paint_box(self,index,dets, bbox_params,save_path):
        img_dict=self.root
        painy_bbox_arr=[]
        bbox_params = bbox_params.split(',')
        img=cv2.imread(self.root+index+'.jpg')
        for bbox in dets:
            if bbox[4] > float(bbox_params[4]):
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                tag = 'person_'+str(bbox[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
                cv2.putText(img, tag, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),thickness=2)
                cv2.imwrite(save_path + index + '.jpg', img)
                if((bbox[0]>float(bbox_params[0])) and (bbox[1]>float(bbox_params[1])) and (bbox[2]<float(bbox_params[2])) and (bbox[3]<float(bbox_params[3]))):
                    return True

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
