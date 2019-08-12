import numpy as np
import cv2
from flask import Flask, render_template, url_for, request, json, jsonify
# import pymysql
import os
from threading import Thread
from time import sleep
 
def async_fun(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return wrapper

# 创建flask对象
app = Flask(__name__)

# 打开数据库连接 ip地址、用户名、密码、数据库名
# db = pymysql.connect("10.180.38.176", "root", "root", "disinfection")

# 使用 cursor() 方法创建一个游标对象 cursor
# cursor = db.cursor()

@async_fun
@app.route('/detection', methods=['POST'], strict_slashes=False)
def light_detection():
    data = request.form
    path = data.get('from')
    device_ip = data.get('deviceid')
    save_path = data.get('target')
    start_time = data.get('start')
    end_time = data.get('end')
    y_min = data.get('ymin')
    y_max = data.get('ymax')
    x_max = data.get('xmax')
    x_min = data.get('xmin')
    num_th = data.get('num')
    bbox_th = data.get('bbox')
    if (path and device_ip and save_path and start_time and bbox_th and y_max and y_min and x_max and x_min and num_th):
        if(os.path.exists(path) and os.path.exists(save_path)):
            light_detection_command(path, device_ip, save_path, start_time, end_time, y_max, y_min, x_max, x_min, num_th, bbox_th)
            return jsonify({"code": 200, "msg": '检测已开始'});
        else:
            return jsonify({"code": 400, "msg": '路径不存在'});
    else:
        return jsonify({"code": 400, "msg": '请求参数不正确'});

@async_fun
def img_detection(path, device_ip, save_path, start_time, end_time, y_max, y_min, x_max, x_min, num_th, bbox_th):
    bbox_params = x_min + ',' + y_min + ',' + x_max + ',' + y_max + ',' + bbox_th;
    str = ('python test_RFB.py -p '+path +' -save_path '+save_path+' -device_ip '+device_ip+' -start_time '+start_time+' -num_th '+num_th+' -bbox_params '+"".join(bbox_params))
    print(str)
    result1 = os.system(str)
    # 此处需添加调用java的程序
    print('结束检测')
    # for root, dirs, files in os.walk(path, topdown=True):
    #     if(root == path):
    #         print('开始清除路径下的原始图片',root)
    #         all_count=0
    #         for name in files:
    #             _, ending = os.path.splitext(name)
    #             if ending == ".jpg":
    #                 if all_count % 20 == 0:
    #                     print('remove pictures', all_count, '/', len(files))
    #                 os.remove(os.path.join(root, name))
    #                 all_count = all_count + 1



# 开关灯检测函数
@async_fun
def light_detection_command(path, device_ip, save_path, start_time, end_time, y_max, y_min, x_max, x_min, num_th, bbox_th):
    for root, dirs, files in os.walk(path, topdown=True):
        if(path == root):
            print('开始检测路径',root,'的',len(files),'张图片')
            all_count=0
            for name in files:
                _, ending = os.path.splitext(name)
                if all_count % 20 == 0:
                    print('light_detection', all_count, '/', len(files))
                if ending == ".jpg":
                    if(isTurnOn(os.path.join(root, name))==False):
                        os.remove(os.path.join(root, name))
                all_count = all_count + 1
    print('开关灯检测结束，开始人物检测')
    img_detection(path, device_ip, save_path, start_time, end_time, y_max, y_min, x_max, x_min, num_th, bbox_th);

def isTurnOn(img_path, threshold = 10):
    img = cv2.imread(img_path)
    (B, G, R) = cv2.split(img)
    b = np.asarray(B, dtype=np.int16)
    g = np.asarray(G, dtype=np.int16)
    r = np.asarray(R, dtype=np.int16)

    diff1 = (b - g).var()
    diff2 = (b - r).var()
    diff3 = (g - r).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return False
    else:
        #开灯
        return True

if __name__ == '__main__':
    app.run(host='10.13.23.46', port=5000, debug=True)

# 关闭数据库连接
# db.close()
