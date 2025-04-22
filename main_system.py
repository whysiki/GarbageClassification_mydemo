# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from enum import Enum
from detect_moudle import (
    Detector,
    SurveillanceCamera,
    hx711_sensor,
    led_system,
    infrared_system,
    display_detection_results,
    set_target_quadrant,
    rest_motors,
)
from datetime import datetime
import numpy as np 


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

current_path = Path(__file__).resolve().parent


class Ui_Show:
    def __init__(self):
        self.log_data: List[str] = []  # 日志数据
        self.garbage_data: Dict[str, int] = {
            "厨余垃圾": 0,
            "可回收垃圾": 0,
            "有害垃圾": 0,
            "其他垃圾": 0,
        }  # 垃圾数据
        self.garbage_record: List[Dict] = []  # 垃圾记录
        self.garbage_weights: Dict[str, int] = {
            "厨余垃圾": 0,
            "可回收垃圾": 0,
            "有害垃圾": 0,
            "其他垃圾": 0,
        }  # 垃圾重量数据
        self.drawed_frame: Optional[np.ndarray] = None  # 绘制的图像

    def add_log(self, log: str):
        log = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log}"
        self.log_data.append(log)

    def update_garbage_data(self, garbage_type: str, count: int):
        if garbage_type in self.garbage_data:
            self.garbage_data[garbage_type] += count
        else:
            self.garbage_data[garbage_type] = count

    def update_garbage_record(self, record: dict):
        self.garbage_record.append(record)

    def update_garbage_weights(self, garbage_type: str, weight: float):
        if garbage_type in self.garbage_weights:
            self.garbage_weights[garbage_type] += weight
        else:
            self.garbage_weights[garbage_type] = weight

    def update_drawed_frame(self, frame: np.ndarray):
        self.drawed_frame = frame


class SystemStatus(Enum):
    INIT = "init"
    DETECTING = "detecting"
    CLASSIFYING = "classifying"
    DONE = "done"
    ERROR = "error"
    OVERLOAD = "overload"


class SystemStateMachine:
    def __init__(self, model_path: Path):

        self.detector = Detector(model_path=model_path)
        self.mset_threshold = 0.87  # 帧间相似度阈值
        self.len_confs_threshold = 0  # 置信度数量阈值
        self.max_conf_threshold = 0.69  # 置信度最大值阈值
        self.len_boxes_threshold = 0  # 检测框数量阈值
        self.weight_threshold = 3  # 重量阈值 3g
        self.correspond_with_last_frame_threshold = 2  # 连续相似帧阈值
        self.classifying_timeout = 15  # 分类超时
        # self.drawed_frame = None  # 绘制的图像
        self.system_status: SystemStatus = SystemStatus.INIT
        self.camera = SurveillanceCamera(
            mset_threshold=self.mset_threshold, camera_index=0
        )  # 摄像头
        self.led_system = led_system  # LED灯系统
        self.infrared_system = infrared_system  ## 红外传感器系统
        # self.current_weight = None  # 当前重量
        self.ui_show = Ui_Show()  # UI显示系统

    def reset_system(self):  # 重置状态
        rest_motors()  # 重置电机
        self.led_system.all_turn_off()  # 关闭所有LED灯

    def change_state(self, new_state: SystemStatus):
        # print(f"change_state from {self.system_status} to {new_state}")
        self.ui_show.add_log(f"State changed from {self.system_status} to {new_state}")
        self.system_status = new_state

    def process_init(self) -> Optional[np.ndarray]:  # 下一个状态为 DETECTING
        current_frame = self.camera.get_frame()
        assert isinstance(current_frame, np.ndarray), "get Camera frame failed"
        self.change_state(SystemStatus.DETECTING)  # 状态转移到 DETECTING
        return current_frame

    def process_detecting(
        self, frame: np.ndarray
    ):  # 下一个状态为 CLASSIFYING / OVERLOAD
        detect_result = self.detector.detect(frame)
        label_counts, max_label, boxes, confs, ids, img0, cost_time = (
            detect_result[i]
            for i in [
                "label_counts",
                "max_label",
                "boxes",
                "confs",
                "ids",
                "original_image",
                "cost_time",
            ]
        )
        
        current_weight = round(hx711_sensor.result_weight, 2) if isinstance(
            hx711_sensor.result_weight, float
        ) else hx711_sensor.result_weight
        
        cover_status = self.infrared_system.all_check_cover()  # 检查垃圾桶是否满载
        
        if (
            max_label  # 确保最大置信度垃圾类型不为空
            and self.camera.correspond_with_last_frame
            > self.correspond_with_last_frame_threshold  # 连续相似帧阈值
            and len(confs) > self.len_confs_threshold  # 置信度数量阈值
            and max(confs) > self.max_conf_threshold  # 置信度最大值阈值
            and len(boxes) > self.len_boxes_threshold  # 检测框数量阈值
        ):
            # 有探测结果，但是根据当前重量和垃圾桶满载状态判断是否需要分类
            if not cover_status[max_label - 1]:
                self.change_state(SystemStatus.CLASSIFYING)  # 状态转移到 CLASSIFYING
                return max_label, current_weight
            else:
                self.change_state(SystemStatus.OVERLOAD)  # 状态转移到 OVERLOAD
                return max_label  # 返回最大置信度垃圾类型  CLASSIFYING 和 OVERLOAD 状态下的输入参数
        else:
            # 本轮检测结束

            self.change_state(SystemStatus.DONE)  # 状态转移到 DONE

        drawn = display_detection_results(
            boxes=boxes,
            confs=confs,
            ids=ids,
            img=img0,
            cost_time=cost_time,
            current_weight=current_weight,
            return_drawed_img=True,
        )  ## 绘制检测结果

        # self.drawed_frame = drawn
        self.ui_show.update_drawed_frame(drawn)  # 更新绘制的图像

    def process_classifying(
        self, class_index: int, weight: float
    ) -> None:  # 下一个状态为 DONE
        current_weight = weight  # 获取当前重量
        self.led_system.turn_on(class_index - 1)  # 打开对应LED灯
        set_target_quadrant(class_index)
        rest_motors()
        self.led_system.turn_off(class_index - 1)  ## 关闭对应LED灯
        self.camera.reset_correspond_with_last_frame()  # 重置连续相似帧计数器
        ##
        self.ui_show.update_garbage_data(GarbageClassMap[class_index], count=1)
        self.ui_show.update_garbage_weights(
            GarbageClassMap[class_index], weight=current_weight
        )
        self.ui_show.update_garbage_record(
            {
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "垃圾类型": GarbageClassMap[class_index],
                "重量(g)": current_weight,
            }
        )
        ##
        self.change_state(SystemStatus.DONE)  # 状态转移到 DONE

    def process_overload(self, class_index: int):  # 下一个状态为 DONE
        self.led_system.blink(class_index - 1)  # 闪烁提示
        # print(f"{class_index}垃圾桶满载，请清理")
        self.ui_show.add_log(f"{GarbageClassMap[class_index]}垃圾桶满载，请清理")
        self.change_state(SystemStatus.DONE)  # 状态转移到 DONE

    def process_done(self):  # 下一个状态为 INIT
        self.change_state(SystemStatus.INIT)  # 状态转移到 INIT

    def process_error(self, error: Exception):  # 下一个状态为 INIT
        # print(f"System error: {str(error)}")
        self.ui_show.add_log(f"System error: {str(error)}")
        self.reset_system()
        self.change_state(SystemStatus.INIT)  # 状态转移到 INIT

    def process_state(self, state: SystemStatus, **kargs) -> dict:
        try:
            if state == SystemStatus.INIT:
                frame = self.process_init()
                return {"frame": frame}
            elif state == SystemStatus.DETECTING:
                class_index, weight = self.process_detecting(**kargs)
                return {"class_index": class_index, "weight": weight}
            elif state == SystemStatus.CLASSIFYING:
                self.process_classifying(**kargs)
                return {}
            elif state == SystemStatus.OVERLOAD:
                self.process_overload(**kargs)
                return {}
            elif state == SystemStatus.DONE:
                self.process_done()
                return {}
            elif state == SystemStatus.ERROR:
                self.process_error(**kargs)
                return {}
            else:
                raise ValueError("Invalid state")
        except Exception as e:  # 捕获异常并处理
            self.change_state(SystemStatus.ERROR)  # 状态转移到 ERROR
            return {"error": e}


class Main_System:
    def __init__(self, model_path: Path):
        self.state_machine = SystemStateMachine(model_path=model_path)
        self.current_out_kargs = {}  # 当前输出参数
    
    def run_once(self):
        self.current_out_kargs = self.state_machine.process_state(
            self.state_machine.system_status, **self.current_out_kargs
        )



    
