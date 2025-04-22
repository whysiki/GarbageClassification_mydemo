# -*- coding: utf-8 -*-

from PySide6.QtCore import (
    QCoreApplication,
    QMetaObject,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
)


class Ui_window(object):
    def setupUi(self, window):
        if not window.objectName():
            window.setObjectName("window")
        window.resize(513, 409)
        self.verticalLayout = QVBoxLayout(window)
        self.verticalLayout.setObjectName("verticalLayout")
        self.menu_frame = QFrame(window)
        self.menu_frame.setObjectName("menu_frame")
        self.menu_frame.setEnabled(True)
        self.menu_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.menu_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.menu_frame)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.start_stop_frame = QFrame(self.menu_frame)
        self.start_stop_frame.setObjectName("start_stop_frame")
        self.start_stop_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.start_stop_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.start_stop_frame)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.start_button = QPushButton(self.start_stop_frame)
        self.start_button.setObjectName("start_button")
        self.start_button.setEnabled(True)
        self.start_button.setStyleSheet("background-color: green")

        self.horizontalLayout_6.addWidget(self.start_button)

        self.stop_button = QPushButton(self.start_stop_frame)
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setStyleSheet("background-color: red")

        self.horizontalLayout_6.addWidget(self.stop_button)

        self.horizontalLayout_2.addWidget(self.start_stop_frame)

        self.data_button = QPushButton(self.menu_frame)
        self.data_button.setObjectName("data_button")
        self.data_button.setStyleSheet("background-color: pink")

        self.horizontalLayout_2.addWidget(self.data_button)

        self.status_button = QPushButton(self.menu_frame)
        self.status_button.setObjectName("status_button")
        self.status_button.setStyleSheet("background-color: orange")

        self.horizontalLayout_2.addWidget(self.status_button)

        self.verticalLayout.addWidget(self.menu_frame)

        self.status_display_frame = QFrame(window)
        self.status_display_frame.setObjectName("status_display_frame")
        self.status_display_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.status_display_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.status_display_frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.image_frame = QFrame(self.status_display_frame)
        self.image_frame.setObjectName("image_frame")
        self.image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.image_frame)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QLabel(self.image_frame)
        self.label.setObjectName("label")

        self.horizontalLayout_4.addWidget(self.label)

        self.horizontalLayout.addWidget(self.image_frame)

        self.log_frame = QFrame(self.status_display_frame)
        self.log_frame.setObjectName("log_frame")
        self.log_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.log_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.log_frame)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.textEdit = QTextEdit(self.log_frame)
        self.textEdit.setObjectName("textEdit")

        self.horizontalLayout_5.addWidget(self.textEdit)

        self.horizontalLayout.addWidget(self.log_frame)

        self.verticalLayout.addWidget(self.status_display_frame)

        self.data_display_frame = QFrame(window)
        self.data_display_frame.setObjectName("data_display_frame")
        self.data_display_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.data_display_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.data_display_frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.chart_frame = QFrame(self.data_display_frame)
        self.chart_frame.setObjectName("chart_frame")
        self.chart_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.chart_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.chart_frame_layout = QHBoxLayout(self.chart_frame)
        self.chart_frame_layout.setObjectName("chart_frame_layout")

        self.verticalLayout_2.addWidget(self.chart_frame)

        self.tableWidget = QTableWidget(self.data_display_frame)
        self.tableWidget.setObjectName("tableWidget")

        self.verticalLayout_2.addWidget(self.tableWidget)

        self.tableWidget_2 = QTableWidget(self.data_display_frame)
        self.tableWidget_2.setObjectName("tableWidget_2")

        self.verticalLayout_2.addWidget(self.tableWidget_2)

        self.verticalLayout.addWidget(self.data_display_frame)

        self.retranslateUi(window)

        QMetaObject.connectSlotsByName(window)


    def retranslateUi(self, window):
        window.setWindowTitle(QCoreApplication.translate("window", "Form", None))
        self.start_button.setText(
            QCoreApplication.translate("window", "\u5f00\u59cb", None)
        )
        self.stop_button.setText(
            QCoreApplication.translate("window", "\u505c\u6b62", None)
        )
        self.data_button.setText(
            QCoreApplication.translate("window", "\u6570\u636e", None)
        )
        self.status_button.setText(
            QCoreApplication.translate("window", "\u72b6\u6001", None)
        )
        self.label.setText("")

