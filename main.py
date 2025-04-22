# -*- coding: utf-8 -*-
from main_system import Main_System
from ui_layout import Ui_window
import numpy as np
import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal
import cv2
from typing import Union, List, Dict
from PySide6.QtWidgets import QTableWidgetItem, QHeaderView, QSizePolicy
from PySide6.QtCharts import (
    QChartView,
    QChart,
    QBarSeries,
    QBarSet,
    QBarCategoryAxis,
    QValueAxis,
)
from PySide6.QtGui import QColor, QPainter
from pathlib import Path

color_map = {
    "厨余垃圾": "green",
    "可回收垃圾": "blue",
    "有害垃圾": "red",
    "其他垃圾": "orange",
}


def garbage_data_to_bar_chart(garbage_data, color_map) -> QChartView:

    # 创建条形图数据
    bar_series = QBarSeries()

    # X 轴标签列表
    categories = list(garbage_data.keys())
    num_categories = len(categories)

    for index, (garbage_type, value) in enumerate(garbage_data.items()):
        bar_set = QBarSet(garbage_type)
        # 构造与类别数量相同的数据列表，只有当前类别赋予实际值，其他赋 0
        data = [0] * num_categories
        data[index] = value
        bar_set.append(data)
        bar_set.setColor(QColor(color_map[garbage_type]))

        bar_series.append(bar_set)

    # 创建图表并设置相关属性
    chart = QChart()
    chart.addSeries(bar_series)
    # chart.setTitle("垃圾分类条形图")
    chart.setAnimationOptions(QChart.SeriesAnimations)

    # 创建并设置 X 轴
    axis_x = QBarCategoryAxis()
    axis_x.append(categories)  # 设置 X 轴的标签
    chart.addAxis(axis_x, Qt.AlignBottom)  # 将 X 轴添加到底部

    # 创建并设置 Y 轴
    axis_y = QValueAxis()
    axis_y.setRange(0, max(garbage_data.values()) + 1)  # 设置 Y 轴范围
    chart.addAxis(axis_y, Qt.AlignLeft)  # 将 Y 轴添加到左边

    # 将条形图系列与图表的轴关联
    bar_series.attachAxis(axis_x)
    bar_series.attachAxis(axis_y)

    # 创建图表视图并
    chart_view = QChartView(chart)
    chart_view.setRenderHint(QPainter.Antialiasing)
    # chart_view.setMinimumSize(QSize(600, 400))
    chart_view.setMinimumHeight(200)  # 设置最小高度
    chart_view.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
    )  # 设置图表视图的大小策略为可扩展

    return chart_view


def np_ndarray_to_qpixmap(ndarray: np.ndarray) -> QPixmap:
    """将 numpy ndarray 转换为 QPixmap"""
    if isinstance(ndarray, np.ndarray):
        # BGR TO RGB
        ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
        if ndarray.ndim == 2:  # 灰度图
            h, w = ndarray.shape
            bytes_per_line = w
            q_img = QImage(
                ndarray.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
            )
        elif ndarray.ndim == 3 and ndarray.shape[2] == 3:  # RGB 图像
            h, w, c = ndarray.shape
            bytes_per_line = c * w
            q_img = QImage(
                ndarray.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            raise ValueError("Unsupported image format")
        return QPixmap.fromImage(q_img)
    else:
        # 返回空白图像
        pixmap = QPixmap(640 / 2, 480 / 2)
        pixmap.fill(Qt.GlobalColor.black)  # 填充黑色
        return pixmap


class Worker(QThread):
    update_log = Signal(object)  # 日志更新信号 #List[str]
    update_frame = Signal(QPixmap)  # 图像更新信号
    update_record = Signal(object)  # 记录更新信号 #List[Dict]
    update_statistics = Signal(object)  # 统计数据更新信号 #Dict[str, Dict[str, int]]

    def __init__(self, system):
        super().__init__()
        self.system: Main_System = system
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            self.system.run_once()
            # 发送更新信号
            self.update_log.emit(
                self.system.state_machine.ui_show.log_data
            )  # 日志更新信号
            self.update_statistics.emit(
                dict(
                    garbage_data=self.system.state_machine.ui_show.garbage_data,
                    garbage_weights=self.system.state_machine.ui_show.garbage_weights,
                )
            )  # 数据统计更新信号
            self.update_frame.emit(
                np_ndarray_to_qpixmap(self.system.state_machine.ui_show.drawed_frame)
            )  # 图像更新信号
            self.update_record.emit(
                self.system.state_machine.ui_show.garbage_record
            )  # 记录更新信号

    def stop(self):
        self.running = False


class Window(QWidget, Ui_window):
    def __init__(self,model_path: Path):
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("智能垃圾分类系统")
        self.system = Main_System(model_path=model_path)
        self.worker = None

        self.initui()
        self.initialize_connections()

    def initui(self):

        self.status_display_frame.setVisible(True)
        self.data_display_frame.setVisible(False)
        self.textEdit.setReadOnly(True)
        self.menu_frame.setMaximumHeight(63)
        self.log_frame.setMinimumWidth(300)

    def initialize_connections(self):
        self.status_button.clicked.connect(self.show_status)
        self.data_button.clicked.connect(self.show_data)
        self.start_button.clicked.connect(self.start_system)
        self.stop_button.clicked.connect(self.stop_system)

    def start_system(self):
        if self.worker is None:
            self.worker = Worker(self.system)
            self.worker.update_frame.connect(self.update_frame)
            self.worker.update_log.connect(self.update_log)
            self.worker.update_statistics.connect(self.update_garbage_statistics)
            self.worker.update_record.connect(self.update_garbage_records)
            self.worker.start()
            self.update_log("系统启动")
        else:
            self.update_log("系统已启动")

    def stop_system(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            self.update_log("系统关闭")
        else:
            self.update_log("系统已关闭")

    def show_data(self):

        self.status_display_frame.setVisible(False)
        self.data_display_frame.resize(self.status_display_frame.size())
        self.data_display_frame.setVisible(True)

    def show_status(self):

        self.data_display_frame.setVisible(False)
        self.status_display_frame.resize(self.data_display_frame.size())
        self.status_display_frame.setVisible(True)

    def update_chart(self, garbage_data: Dict[str, int]):
        categories = list(garbage_data.keys())
        max_value = max(garbage_data.values()) + 1

        # 如果已存在 chart_view，则更新现有图表的数据
        if hasattr(self, "chart_view") and self.chart_view is not None:
            chart = self.chart_view.chart()
            # 更新 BarSeries
            series_list = chart.series()
            if series_list and isinstance(series_list[0], QBarSeries):
                series: QBarSeries = series_list[0]
                # series.clear()  # 清空原有数据
                for index, (garbage_type, value) in enumerate(garbage_data.items()):
                    bar_set = series.barSets()[index]
                    # data = [0] * len(categories)
                    # data[index] = value
                    # bar_set.append(data)
                    bar_set.replace(index, value)  # 更新数据
                    bar_set.setColor(QColor(color_map.get(garbage_type, "gray")))
            else:
                # 若series不存在或类型错误，则重建图表
                self.chart_view = garbage_data_to_bar_chart(garbage_data, color_map)
                # 移除旧的 chart_view 并添加新的
                if self.chart_frame_layout.count() > 0:
                    old_widget = self.chart_frame_layout.itemAt(0).widget()
                    self.chart_frame_layout.removeWidget(old_widget)
                    old_widget.deleteLater()
                self.chart_frame_layout.addWidget(self.chart_view)
                chart = self.chart_view.chart()

            # 更新 X 轴和 Y 轴
            for axis in chart.axes():
                if isinstance(axis, QBarCategoryAxis):
                    axis.clear()
                    axis.append(categories)
                elif isinstance(axis, QValueAxis):
                    axis.setRange(0, max_value)

            chart.update()
        else:
            # 第一次创建图表
            self.chart_view = garbage_data_to_bar_chart(garbage_data, color_map)
            self.chart_frame_layout.addWidget(self.chart_view)

    def update_frame(self, pixmap: QPixmap):
        self.label.setPixmap(pixmap)

    def update_log(self, log: Union[List[str], str]):
        scroll_bar = self.textEdit.verticalScrollBar()
        auto_scroll = scroll_bar.value() == scroll_bar.maximum()
        if isinstance(log, list):
            current_text = self.textEdit.toPlainText()
            current_text_lines = current_text.split("\n")
            new_lines = [line for line in log if line not in current_text_lines]
            self.textEdit.append("\n".join(new_lines))
        elif isinstance(log, str):
            self.textEdit.append(log)
        else:
            raise ValueError("log must be a list or a string")
        if auto_scroll:
            scroll_bar.setValue(scroll_bar.maximum())

    def update_garbage_statistics(
        self, garbage_data_garbage_weights: Dict[str, Dict[str, int]]
    ):
        garbage_data = garbage_data_garbage_weights["garbage_data"]
        self.update_chart(garbage_data)
        garbage_weights = garbage_data_garbage_weights["garbage_weights"]
        self.tableWidget.setUpdatesEnabled(False)
        self.tableWidget.setRowCount(len(garbage_weights))
        headers = ["垃圾类型", "重量", "数量"]
        if self.tableWidget.columnCount() != len(headers):
            self.tableWidget.setColumnCount(len(headers))
            self.tableWidget.setHorizontalHeaderLabels(headers)
            self.tableWidget.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )
        self.tableWidget.setVerticalHeaderLabels(list(garbage_weights.keys()))
        self.tableWidget.verticalHeader().setVisible(False)
        
        self.tableWidget.horizontalHeader().setStyleSheet(
            "QHeaderView::section { background-color: orange; }"
            "QHeaderView::section:hover { background-color: green; }"
        )

        for i, (key, weight) in enumerate(garbage_weights.items()):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(key))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(weight)))
            self.tableWidget.setItem(
                i, 2, QTableWidgetItem(str(garbage_data.get(key, 0)))
            )
        self.tableWidget.setUpdatesEnabled(True)

        total_height = self.tableWidget.horizontalHeader().height()
        for row in range(self.tableWidget.rowCount()):
            total_height += self.tableWidget.rowHeight(row)
        total_height += 2 * self.tableWidget.frameWidth()
        self.tableWidget.setMinimumHeight(total_height)

    def update_garbage_records(self, garbage_record: List[Dict]):
        scroll_bar = self.tableWidget_2.verticalScrollBar()
        auto_scroll = scroll_bar.value() == scroll_bar.maximum()

        headers = ["时间", "垃圾类型", "重量(g)"]
        self.tableWidget_2.setUpdatesEnabled(False)
        self.tableWidget_2.setRowCount(len(garbage_record))
        if self.tableWidget_2.columnCount() != len(headers):
            self.tableWidget_2.setColumnCount(len(headers))
            self.tableWidget_2.setHorizontalHeaderLabels(headers)
            self.tableWidget_2.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )
        self.tableWidget_2.setVerticalHeaderLabels(
            [str(i) for i in range(len(garbage_record))]
        )
        for i, record in enumerate(garbage_record):
            self.tableWidget_2.setItem(i, 0, QTableWidgetItem(record.get("时间", "")))
            self.tableWidget_2.setItem(
                i, 1, QTableWidgetItem(record.get("垃圾类型", ""))
            )
            self.tableWidget_2.setItem(
                i, 2, QTableWidgetItem(str(record.get("重量(g)", "")))
            )
        self.tableWidget_2.horizontalHeader().setStyleSheet(
            "QHeaderView::section { background-color: blue; }"
            "QHeaderView::section:hover { background-color: green; }"
        )
        self.tableWidget_2.setMinimumHeight(200)
        self.tableWidget_2.setUpdatesEnabled(True)
        self.tableWidget_2.viewport().update()
        if auto_scroll:
            scroll_bar.setValue(scroll_bar.maximum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    current_path = Path(__file__).resolve().parent
    window = Window(model_path=current_path / "models/111.onnx")
    window.show()
    sys.exit(app.exec())
