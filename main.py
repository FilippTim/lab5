import cv2
import sys
from window import ImageWindow
import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
save_process_path = 'stuff/saved/save_proc.jpg'

class Mywindow(ImageWindow):
    def __init__(self):
        super().__init__()
        self.initial_path = ''
        self.img_or_video = 0
        self.cap = None
        self.textCount = None
        self.selected_text_index = None
        self.data = None
        self.create_video_buttons()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mask_image = cv2.imread('C:/opencv/lab5/stuff/images/mask.png', cv2.IMREAD_UNCHANGED)  # Ensure you have the mask image with an alpha channel

    
    def create_video_buttons(self):
        self.button_text = QPushButton("Текст")
        self.button_face = QPushButton("Лицо")
        self.button_mask = QPushButton("Маска")
        self.button_stop = QPushButton("Стоп")
        
        self.button_text.clicked.connect(self.on_button_text_clicked)
        self.button_face.clicked.connect(self.on_button_face_clicked)
        self.button_mask.clicked.connect(self.on_button_mask_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)
        
        self.button_layout.addWidget(self.button_text)
        self.button_layout.addWidget(self.button_face)
        self.button_layout.addWidget(self.button_mask)
        self.button_layout.addWidget(self.button_stop)
        
        self.show_video_buttons(False)

    def show_video_buttons(self, show):
        self.button_text.setVisible(show)
        self.button_face.setVisible(show)
        self.button_mask.setVisible(show)
        self.button_stop.setVisible(show)

    def on_button_text_clicked(self):
        self.stop_video_processing = False
        self.video_text()

    def on_button_face_clicked(self):
        self.stop_video_processing = False
        print("Ищем лица в видео")
        # self.detect_faces_in_video()
        self.detect_faces_in_video()

    def on_button_mask_clicked(self):
        self.stop_video_processing = False
        print("Ищем маски в видео")
        # self.detect_masks_in_video()
        self.apply_masks_in_video()
    def on_button_stop_clicked(self):
        self.stop_video_processing = True

    def on_combo_box_changed(self, value):
        self.img_or_video = value
        self.img_hide()
        if self.img_or_video == 0:
            self.combo_box_selectText.setVisible(0)
            self.download_video(1)
            self.show_video_buttons(True)
            self.show_vtext(0)
        elif self.img_or_video == 1:
            self.download_img(1)
            self.img_selectedshow()
            self.show_video_buttons(False)

    def on_button_vtext_clicked(self):
        print('Выделить текст')
        self.detect_text_area()
        self.select_text_show()

    def on_combo_box_selectText_changed(self, value):
        self.selected_text_index = value
        if self.selected_text_index >= 0:
            self.save_text_image()
            self.extract_text()
            self.select_text_rectangle()

    def create_combo_box_selectText(self, numItems):
        self.combo_box_selectText.clear()
        for i in range(1, numItems + 1):
            self.combo_box_selectText.addItem(f"Text {i}")

    def save_text_image(self):
        try:
            if self.selected_text_index is not None and self.data is not None:
                x, y, w, h = self.data['left'][self.selected_text_index], self.data['top'][self.selected_text_index], \
                             self.data['width'][self.selected_text_index], self.data['height'][self.selected_text_index]
                image = cv2.imread(self.initial_path)
                selected_text_image = image[y:y+h, x:x+w]
                cv2.imwrite(save_process_path, selected_text_image)
                self.update_image3(save_process_path)
        except Exception as e:
            print("Ошибка при сохранении участка изображения: ", e)

    def extract_text(self):
        try:
            if self.selected_text_index is not None and self.data is not None:
                text = self.data['text'][self.selected_text_index]
                print(f"Текст: {text}")
                QMessageBox.information(self, "Извлеченный текст", f"{text}")
        except Exception as e:
            print("Ошибка при выделении текста: ", e)

    # сервисные функции
    def download_img(self, i):
        try:
            self.initial_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Изображения (*.png *.jpg *.jpeg)")
            if not self.initial_path:
                raise FileNotFoundError("Путь к изображению не был выбран.")
            if i == 1:
                self.update_images1(self.initial_path)
            else:
                raise FileNotFoundError("Куда ты хочешь картинку?")
        except Exception as e:
            print("Ошибка при загрузке изображения", e)
            return None

    def download_video(self, i):
        try:
            self.initial_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Видео (*.mp4)")
            if not self.initial_path:
                raise FileNotFoundError("Путь к видео не был выбран.")
            if i == 1:
                self.cap = cv2.VideoCapture(self.initial_path)
                _, img = self.cap.read()
                cv2.imwrite(save_process_path, img)
                self.update_images1(save_process_path)
            else:
                raise FileNotFoundError("Куда ты хочешь картинку?")
        except Exception as e:
            print("Ошибка при загрузке видео", e)
            return None

    def video_process1(self):
        while True:
            ret, img = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imwrite(save_process_path, img)
            self.update_images1(save_process_path)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    def video_process2(self,oper='text'):
        try:
            tmp=0
            while True:
                if self.stop_video_processing:  # Добавлено
                    break
                tmp+=1
                if tmp%100000:
                    continue
                ret, img = self.cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                self.label1_title.hide()
                self.image_label1.hide()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if oper=='text':
                    # Применяем OCR к изображению
                    config = r'--oem 3 --psm 6'
                    self.data = pytesseract.image_to_data(binary, config=config)
                    
                    for i, el in enumerate(self.data.splitlines()):
                        if i ==0:
                            continue
                        el = el.split()
                        try:
                            x, y, w, h = int(el[6]), int(el[7]), int(el[8]), int(el[9])
                            cv2.rectangle(img, (x,y), (w+x, h+y), (0,0,255),1)
                            cv2.putText(img, el[11], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
                        except IndexError:
                            print('')
                    cv2.imwrite(save_process_path, img)
                    self.update_images2(save_process_path)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print("Ошибка при video_process2: ", e)
            return None
    def detect_text_area(self):
        try:
            image = cv2.imread(self.initial_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Применяем OCR к изображению
            config = r'--oem 3 --psm 6'
            self.data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
            
            number_of_boxes = 0
            for i in range(len(self.data['text'])):
                # Получаем координаты и размеры текущего слова
                x, y, w, h = self.data['left'][i], self.data['top'][i], self.data['width'][i], self.data['height'][i]

                # Отфильтруем слова с низкой уверенностью
                #if int(self.data['conf'][i]) > 0:
                    # Рисуем прямоугольник вокруг слова
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                number_of_boxes += 1
                    # Отображаем номер слова
                cv2.putText(image, str(number_of_boxes), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            self.textCount = number_of_boxes
            cv2.imwrite(save_process_path, image)
            self.update_images2(save_process_path)
            self.create_combo_box_selectText(self.textCount)
        except Exception as e:
            print("Ошибка при detect_text_area: ", e)
            return None

    def video_text(self):
        print('ищу текст с видео')
        self.stop_video_processing = False
        self.video_process2()
    def detect_faces_in_video(self):
        try:
            while True:
                if self.stop_video_processing:  
                    break
                ret, img = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imwrite(save_process_path, img)
                self.update_images2(save_process_path)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print("Ошибка при detect_faces_in_video: ", e)

    def apply_masks_in_video(self):
        try:
            while True:
                if self.stop_video_processing:
                    break
                ret, img = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    mask_resized = cv2.resize(self.mask_image, (w, h))
                    mask_gray = mask_resized[:, :, 3]
                    _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
                    mask_binary_inv = cv2.bitwise_not(mask_binary)
                    mask_rgb = mask_resized[:, :, :3]

                    roi = img[y:y+h, x:x+w]
                    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_binary_inv)
                    roi_fg = cv2.bitwise_and(mask_rgb, mask_rgb, mask=mask_binary)

                    dst = cv2.add(roi_bg, roi_fg)
                    img[y:y+h, x:x+w] = dst

                cv2.imwrite(save_process_path, img)
                self.update_images2(save_process_path)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print("Ошибка при apply_masks_in_video: ", e)

 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Mywindow()
    sys.exit(app.exec())

