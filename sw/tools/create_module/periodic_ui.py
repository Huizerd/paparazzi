# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'periodic_ui.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Periodic(object):
    def setupUi(self, Periodic):
        Periodic.setObjectName("Periodic")
        Periodic.resize(699, 115)
        self.verticalLayout = QtWidgets.QVBoxLayout(Periodic)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_11 = QtWidgets.QLabel(Periodic)
        self.label_11.setStyleSheet("font: bold")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout.addWidget(self.label_11)
        self.remove_button = QtWidgets.QToolButton(Periodic)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("remove.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.remove_button.setIcon(icon)
        self.remove_button.setObjectName("remove_button")
        self.horizontalLayout.addWidget(self.remove_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_6 = QtWidgets.QLabel(Periodic)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.freq_spinbox = QtWidgets.QDoubleSpinBox(Periodic)
        self.freq_spinbox.setProperty("value", 1.0)
        self.freq_spinbox.setObjectName("freq_spinbox")
        self.horizontalLayout_5.addWidget(self.freq_spinbox)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_5 = QtWidgets.QLabel(Periodic)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.periodic_edit = QtWidgets.QLineEdit(Periodic)
        self.periodic_edit.setObjectName("periodic_edit")
        self.horizontalLayout_3.addWidget(self.periodic_edit)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_9 = QtWidgets.QLabel(Periodic)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_9.addWidget(self.label_9)
        self.autorun_comboBox = QtWidgets.QComboBox(Periodic)
        self.autorun_comboBox.setObjectName("autorun_comboBox")
        self.autorun_comboBox.addItem("")
        self.autorun_comboBox.setItemText(0, "")
        self.autorun_comboBox.addItem("")
        self.autorun_comboBox.addItem("")
        self.horizontalLayout_9.addWidget(self.autorun_comboBox)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_9)
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_8 = QtWidgets.QLabel(Periodic)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_8.addWidget(self.label_8)
        self.stop_edit = QtWidgets.QLineEdit(Periodic)
        self.stop_edit.setObjectName("stop_edit")
        self.horizontalLayout_8.addWidget(self.stop_edit)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_7 = QtWidgets.QLabel(Periodic)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.start_edit = QtWidgets.QLineEdit(Periodic)
        self.start_edit.setObjectName("start_edit")
        self.horizontalLayout_7.addWidget(self.start_edit)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_7)
        self.verticalLayout.addLayout(self.horizontalLayout_10)

        self.retranslateUi(Periodic)
        QtCore.QMetaObject.connectSlotsByName(Periodic)

    def retranslateUi(self, Periodic):
        _translate = QtCore.QCoreApplication.translate
        Periodic.setWindowTitle(_translate("Periodic", "Periodic"))
        self.label_11.setText(_translate("Periodic", "Periodic"))
        self.remove_button.setText(_translate("Periodic", "remove"))
        self.label_6.setText(_translate("Periodic", "Freq"))
        self.label_5.setText(_translate("Periodic", "fun"))
        self.label_9.setText(_translate("Periodic", "Autorun"))
        self.autorun_comboBox.setItemText(1, _translate("Periodic", "TRUE"))
        self.autorun_comboBox.setItemText(2, _translate("Periodic", "FALSE"))
        self.label_8.setText(_translate("Periodic", "Stop"))
        self.label_7.setText(_translate("Periodic", "Start"))
