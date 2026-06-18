# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_ParameterHZPhVs.ui'
##
## Created by: Qt User Interface Compiler version 6.4.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QHeaderView,
    QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QWidget)

class Ui_ParameterEdit(object):
    def setupUi(self, ParameterEdit):
        if not ParameterEdit.objectName():
            ParameterEdit.setObjectName(u"ParameterEdit")
        ParameterEdit.resize(400, 300)
        self.table_ParametersValues = QTableWidget(ParameterEdit)
        if (self.table_ParametersValues.columnCount() < 2):
            self.table_ParametersValues.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.table_ParametersValues.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.table_ParametersValues.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        self.table_ParametersValues.setObjectName(u"table_ParametersValues")
        self.table_ParametersValues.setGeometry(QRect(30, 10, 341, 251))
        self.table_ParametersValues.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_ParametersValues.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_ParametersValues.setSortingEnabled(True)
        self.table_ParametersValues.setColumnCount(2)
        self.table_ParametersValues.horizontalHeader().setVisible(True)
        self.table_ParametersValues.horizontalHeader().setCascadingSectionResizes(False)
        self.table_ParametersValues.horizontalHeader().setMinimumSectionSize(0)
        self.table_ParametersValues.horizontalHeader().setDefaultSectionSize(120)
        self.table_ParametersValues.horizontalHeader().setProperty("showSortIndicator", True)
        self.table_ParametersValues.horizontalHeader().setStretchLastSection(True)
        self.table_ParametersValues.verticalHeader().setStretchLastSection(False)
        self.Button_Save = QPushButton(ParameterEdit)
        self.Button_Save.setObjectName(u"Button_Save")
        self.Button_Save.setGeometry(QRect(30, 270, 171, 24))
        self.Button_Discard = QPushButton(ParameterEdit)
        self.Button_Discard.setObjectName(u"Button_Discard")
        self.Button_Discard.setGeometry(QRect(200, 270, 171, 24))

        self.retranslateUi(ParameterEdit)

        QMetaObject.connectSlotsByName(ParameterEdit)
    # setupUi

    def retranslateUi(self, ParameterEdit):
        ParameterEdit.setWindowTitle(QCoreApplication.translate("ParameterEdit", u"Parameters Edit", None))
        ___qtablewidgetitem = self.table_ParametersValues.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("ParameterEdit", u"Parameter", None));
        ___qtablewidgetitem1 = self.table_ParametersValues.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("ParameterEdit", u"Value", None));
        self.Button_Save.setText(QCoreApplication.translate("ParameterEdit", u"Save", None))
        self.Button_Discard.setText(QCoreApplication.translate("ParameterEdit", u"Discard", None))
    # retranslateUi

