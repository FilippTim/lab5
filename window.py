from PyQt6.QtWidgets import QComboBox, QSlider, QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Изображения")
        self.setGeometry(50, 50, 400, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.image_layout = QVBoxLayout()
        self.layout.addLayout(self.image_layout)

        self.image_label1 = QLabel()
        self.label1_title = QLabel('До обработки')
        self.image_label2 = QLabel()
        self.label2_title = QLabel('После обработки')

        #self.update_images1("stuff/images/white.jpg")
        #self.update_images2("stuff/images/white.jpg")

        self.image_layout.addWidget(self.label1_title, alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_layout.addWidget(self.image_label1, alignment=Qt.AlignmentFlag.AlignCenter)
        self.label1_title.hide()
        self.image_layout.addWidget(self.label2_title, alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_layout.addWidget(self.image_label2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label3 = QLabel()
        self.label3_title = QLabel('Участок с текстом')
        self.image_layout.addWidget(self.label3_title, alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_layout.addWidget(self.image_label3, alignment=Qt.AlignmentFlag.AlignCenter)

        self.button_layout = QVBoxLayout()
        self.layout.addLayout(self.button_layout)
        self.update_button()

        self.img_hide()
        self.show()

    def update_images1(self, image_path1):
        self.label1_title.show()
        pixmap1 = QPixmap(image_path1)
        scaled_pixmap1 = pixmap1.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label1.setPixmap(scaled_pixmap1)
        self.image_label1.show()
        self.update()

    def img_hide(self):
        self.label1_title.hide()
        self.image_label1.hide()
        self.label2_title.hide()
        self.image_label2.hide()
        self.label3_title.hide()
        self.image_label3.hide()

    def update_images2(self, image_path2):
        self.label2_title.show()
        pixmap2 = QPixmap(image_path2)
        scaled_pixmap2 = pixmap2.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label2.setPixmap(scaled_pixmap2)
        self.image_label2.show()
        self.update()

    def update_image3(self, image_patch2):
        self.label3_title.show()
        pixmap3 = QPixmap(image_patch2)
        scaled_pixmap3 = pixmap3.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label3.setPixmap(scaled_pixmap3)
        self.image_label3.show()

    def update_button(self):
        self.combo_box = QComboBox()
        self.combo_box.addItem("Выбрать видео")
        self.combo_box.addItem("Выбрать изображение")
        self.combo_box.currentIndexChanged.connect(self.on_combo_box_changed)
        self.button_layout.addWidget(self.combo_box)

        self.button_vtext = QPushButton("Выделить текст")
        self.button_vtext.clicked.connect(self.on_button_vtext_clicked)
        self.button_layout.addWidget(self.button_vtext)
        self.button_vtext.hide()

        self.combo_box_selectText = QComboBox()
        self.combo_box_selectText.currentIndexChanged.connect(self.on_combo_box_selectText_changed)
        self.create_combo_box_selectText(10)
        self.button_layout.addWidget(self.combo_box_selectText)
        self.combo_box_selectText.hide()

    def show_vtext(self, i):
        if i:
            self.button_vtext.show()
        else:
            self.button_vtext.hide()

    def show_uchastok(self, i):
        if i:
            self.combo_box_selectText.show()
        else:
            self.combo_box_selectText.hide()

    def img_selectedshow(self):
        self.show_vtext(1)
        self.show_uchastok(0)

    def select_text_show(self):
        self.show_vtext(0)
        self.show_uchastok(1)

    def select_text_rectangle(self):
        self.show_vtext(0)
        self.show_uchastok(1)
