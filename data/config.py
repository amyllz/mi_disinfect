# config.py
import os.path

# gets home dir cross platform
#home = os.path.expanduser("~")
#ddir = os.path.join(home,"/home/llz/data/4out/")

# note: if you used our download scripts, this should be right
#VOCroot = ddir # path to VOCdevkit root dir


#RFB CONFIG

COCO_mobile_300 = {
    'feature_maps' : [19, 10, 5, 3, 2, 1],

    'min_dim' : 300,

    'steps' : [16, 32, 64, 100, 150, 300],

    'min_sizes' : [45, 90, 135, 180, 225, 270],

    'max_sizes' : [90, 135, 180, 225, 270, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}
