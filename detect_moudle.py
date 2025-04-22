# -*- coding: utf-8 -*-
import time
import random
from pathlib import Path
from typing import List, Union
import cv2
import numpy as np
import onnxruntime as ort
from skimage.metrics import structural_similarity as ssim

current_path = Path(__file__).parent.resolve()  # 获取当前文件路径
test_image_path = current_path / "testImage"  # 测试图片路径

GarbageClassMap = {
    1: "有害垃圾",  # hazardous waste
    2: "可回收垃圾",  # recyclable waste
    3: "厨余垃圾",  # kitchen waste
    4: "其他垃圾",  # other waste
}

GarbageClassMapEnglish = {1: "hazardous", 2: "recyclable", 3: "kitchen", 4: "other"}


DiC_LABELS = {
    0: 2,  # "可回收垃圾",
    1: 4,  # "其他垃圾",
    2: 3,  # "厨余垃圾",
    3: 4,  # "其他垃圾",
    4: 1,  # "有害垃圾",
    5: 4,  # "其他垃圾",
    6: 3,  # "厨余垃圾",
    7: 2,  # "可回收垃圾",
    8: 3,  # "厨余垃圾",
    9: 1,  # "有害垃圾",
    10: 1,  # "有害垃圾",
    11: 2,  # "可回收垃圾",
}

#接口模拟，此处自己实现

class sensor:
    @property
    def result_weight(self):
        time.sleep(0.1)  # 模拟操作
        wight = random.randint(2, 10)  # 模拟重量传感器的值
        # print(f"Weight: {wight} g")  # 打印重量值
        return wight


class infrared:
    def __init__(self):
        pass

    def ic_cover(self) -> bool:
        time.sleep(0.1)  # 模拟操作
        return random.choice([True, False])  # 模拟红外传感器的值


class led:
    def __init__(self):
        pass

    def turn_on(self):
        time.sleep(0.1)  # 模拟操作
        # print("LED turned on")  # 模拟LED灯打开操作

    def turn_off(self):
        time.sleep(0.1)  # 模拟操作
        # print("LED turned off")  # 模拟LED灯关闭操作

    def blink(self):
        time.sleep(0.1)  # 模拟操作
        # print("LED blinking")  # 模拟LED灯闪烁操作


class Led_system:

    def __init__(self, leds: list):
        self.leds: List[led] = leds  # LED灯列表

    def turn_on(self, index: int):
        self.leds[index].turn_on()
        print(f"LED {index+1} turned on")  # 打印LED灯打开操作

    def turn_off(self, index: int):
        self.leds[index].turn_off()
        print(f"LED {index+1} turned off")  # 打印LED灯关闭操作

    def blink(self, index: int):
        self.leds[index].blink()
        print(f"LED {index+1} blinking")  # 打印LED灯闪烁操作

    def all_turn_off(self):
        list(map(lambda l: l.turn_off(), self.leds))  # 关闭所有LED灯
        print("All LEDs turned off")  # 打印关闭所有LED灯操作

    def all_turn_on(self):
        list(map(lambda l: l.turn_on(), self.leds))
        print("All LEDs turned on")  # 打印打开所有LED灯操作


class Infrared_system:
    def __init__(self, infrareds: list):
        self.infrareds: List[infrared] = infrareds  # 红外传感器列表

    def check_cover(self, index: int) -> bool:
        print(f"Check cover for infrared {index+1}")  # 打印检查红外传感器覆盖操作
        return self.infrareds[index].ic_cover()

    def all_check_cover(self):
        return [i.ic_cover() for i in self.infrareds]


def set_target_quadrant(class_index: int) -> None:
    time.sleep(0.5)  # 模拟操作
    print(f"Set target quadrant to {class_index}")  # 模拟设置目标象限操作


def rest_motors() -> None:
    time.sleep(0.5)  # 模拟操作
    print("Motors reset")  # 模拟电机复位操作


ifa, ifb, ifc, ifd = infrared(), infrared(), infrared(), infrared()  # 模拟红外传感器
leda, ledb, ledc, ledd = led(), led(), led(), led()  # 模拟LED灯
led_system = Led_system([leda, ledb, ledc, ledd])  # 模拟LED灯系统
infrared_system = Infrared_system([ifa, ifb, ifc, ifd])  # 模拟红外传感器系统
hx711_sensor = sensor()  # 模拟重量传感器

def display_detection_results(
    boxes, confs, ids, img, cost_time, current_weight, return_drawed_img=False
):
    # img = img.copy() #不需要复制图像
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = DiC_LABELS[ids[i]]
        confidence_score = confs[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            img=img,
            text=f"{GarbageClassMapEnglish[label] if (isinstance(label,int) and label >0 and label < 5)  else label}: {confidence_score:.2f}",
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0, 0, 255),
            thickness=1,
        )
    fps = 1 / cost_time
    cv2.putText(
        img,
        f"FPS: {fps:.2f} {f'Weight:{current_weight} g' if current_weight else ''} ",
        (10, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )
    if not return_drawed_img:
        cv2.imshow("Detection", img)
    else:
        return img



class SurveillanceCamera:
    def __init__(self, mset_threshold: float, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.last_frame = None
        self.correspond_with_last_frame = 0
        self.mset_threshold = mset_threshold

        ##测试用，读取本地测试图片
        self.alltestimages_generator = list(test_image_path.glob("*.jpg"))

    def mse(self, imageA, imageB):
        if isinstance(imageA, np.ndarray) and isinstance(imageB, np.ndarray):
            if len(imageA.shape) == 3:
                imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            if len(imageB.shape) == 3:
                imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            # 计算SSIM值
            ssim_index, _ = ssim(imageA, imageB, full=True)
            return ssim_index
        return 0

    def get_frame(self) -> Union[np.ndarray, None]:

        # ##测试用，返回测试图片作为帧
        self.correspond_with_last_frame = random.randint(0, 10)
        # 测试用，读取本地图片 作为摄像头帧
        return cv2.imread(random.choice(self.alltestimages_generator))

    def reset_correspond_with_last_frame(self):
        self.correspond_with_last_frame = 0


class Detector:
    def __init__(self, model_path: Path):
        assert isinstance(model_path, (Path, str)), "model_path must be Path or str"
        # self.model_path = model_path
        self.so = ort.SessionOptions()
        self.net = ort.InferenceSession(
            model_path,
            self.so,
            providers=[
                (
                    "CUDAExecutionProvider"
                    if ort.get_device() == "GPU"
                    else "CPUExecutionProvider"
                )
            ],
        )
        assert self.so is not None, "Failed to create session options."
        assert self.net is not None, "Failed to load the model."
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride = [8.0, 16.0, 32.0]
        self.anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
        ]
        self.anchor_grid = np.asarray(self.anchors, dtype=np.float32).reshape(
            self.nl, -1, 2
        )
        self.nms_threshold = 0.4
        self.conf_threshold = 0.5

    def post_process_opencv(self, outputs, model_h, model_w, img_h, img_w):
        # 计算中心坐标及尺寸
        confs = outputs[:, 4]
        c_x = outputs[:, 0] / model_w * img_w
        c_y = outputs[:, 1] / model_h * img_h
        w = outputs[:, 2] / model_w * img_w
        h = outputs[:, 3] / model_h * img_h
        p_cls = outputs[:, 5:]
        if p_cls.ndim == 1:
            p_cls = np.expand_dims(p_cls, 1)
        cls_id = np.argmax(p_cls, axis=1)

        # 计算左上角和右下角坐标（用于显示）
        x1 = c_x - w / 2
        y1 = c_y - h / 2
        x2 = c_x + w / 2
        y2 = c_y + h / 2
        # 用于后续展示的坐标数组
        areas = np.stack((x1, y1, x2, y2), axis=-1)

        # 构造NMS输入：cv2.dnn.NMSBoxes 要求[x, y, w, h]
        boxes_xywh = np.stack((x1, y1, w, h), axis=-1).tolist()
        conf_list = confs.tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, conf_list, self.conf_threshold, self.nms_threshold
        )
        if len(indices) > 0:
            # cv2.dnn.NMSBoxes 返回的索引可能是 [[i],[j],...]，扁平化处理
            indices = np.array(indices).flatten()
            return areas[indices], confs[indices], cls_id[indices]
        else:
            return np.array([]), np.array([]), np.array([])

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def detect(self, image_source: Union[Path, str, np.ndarray]):
        start_time = time.time()
        assert isinstance(
            image_source, (Path, str, np.ndarray)
        ), "image_path must be Path or str or np.ndarray"
        img0 = (
            cv2.imread(image_source)
            if isinstance(image_source, (str, Path))
            else image_source
        )
        if img0 is None:
            raise ValueError("Image not found or unable to load.")
        img = (
            cv2.cvtColor(
                cv2.resize(
                    img0, (self.model_w, self.model_h), interpolation=cv2.INTER_AREA
                ),
                cv2.COLOR_BGR2RGB,
            )
        ).astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        outs_orig: list = self.net.run(None, {self.net.get_inputs()[0].name: blob})
        outs = outs_orig[0].squeeze(axis=0)
        row_ind = 0
        grid = [np.zeros(1)] * self.nl
        for i in range(self.nl):
            h, w = int(self.model_h / self.stride[i]), int(
                self.model_w / self.stride[i]
            )
            length = int(self.na * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)

            outs[row_ind : row_ind + length, 0:2] = (
                outs[row_ind : row_ind + length, 0:2] * 2.0
                - 0.5
                + np.tile(grid[i], (self.na, 1))
            ) * int(self.stride[i])
            outs[row_ind : row_ind + length, 2:4] = (
                outs[row_ind : row_ind + length, 2:4] * 2
            ) ** 2 * np.repeat(self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        img_h, img_w, _ = np.shape(img0)
        boxes, confs, ids = self.post_process_opencv(
            outs,
            self.model_h,
            self.model_w,
            img_h,
            img_w,
        )
        if len(ids) > 0:
            labels = np.vectorize(lambda x: DiC_LABELS[x])(ids)
            unique, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique, counts))
        else:
            label_counts = {label: 0 for label in DiC_LABELS.values()}

        if len(confs) > 0:
            max_idx = np.argmax(confs)
            max_label = DiC_LABELS[ids[max_idx]]
        else:
            max_label = None
        return dict(
            label_counts=label_counts,
            max_label=max_label,
            boxes=boxes,
            confs=confs,
            ids=ids,
            original_image=img0,
            cost_time=time.time() - start_time,
        )
