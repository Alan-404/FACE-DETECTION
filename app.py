from cv2 import VideoCapture
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2 as cv
from recognize import Recognize
from kivy.uix.textinput import TextInput
import os

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_frontalface_default.xml')
import pandas as pd



class MainApp(App):
    def build(self):
        self.title = "Đồ Án Xử Lý Ảnh Số"
        self.main_layout = BoxLayout(orientation="vertical")
        self.screen = 1
        # Dashboard
        self.dashboard = BoxLayout(orientation="vertical")
        self.dashboard_lb = Label(text="Đồ Án Cuối Kỳ Môn Học Xử Lý Ảnh Số\n        19110302 - Nguyễn Đức Trí", font_size="20sp", size_hint=(1,0.5))

        self.btn_reg = Button(text="Nhận Dạng", on_press=self.change_recog_screen, background_color="blue")
        self.btn_train = Button(text="Thêm Người Dùng Mới", on_press=self.go_train_screen, background_color="green")

        self.btn_train_again = Button(text="Học Lại Dữ Liệu", size_hint=(1, 0.1), on_press=self.train_models_again)

        self.btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, 0.1))
        self.btn_layout.add_widget(self.btn_reg)
        self.btn_layout.add_widget(self.btn_train)
        self.dashboard.add_widget(self.dashboard_lb)
        


        self.dashboard.add_widget(self.btn_layout)

        self.dashboard.add_widget(self.btn_train_again)


        # Recognization Screen
        self.recog = BoxLayout(orientation="vertical")
        self.re_lb = Label(text="Nhận Diện Khuôn Mặt", size_hint=(1,0.2))
        self.recog.add_widget(self.re_lb)
        self.reg_cam = Image(size_hint=(1,1))
        self.recog.add_widget(self.reg_cam)
        
        btn_ret = Button(text="Trở Về", size_hint=(1,0.1), background_color='red',  on_press=self.go_back_dashboard)
        self.recog.add_widget(btn_ret)


        # Train Screen
        self.train_screen = BoxLayout(orientation='vertical')
        self.train_cam = Image(size_hint=(1,1))
        
        self.btn_submit = Button(text="Bắt Đầu", size_hint=(1, 0.1), on_press=self.begin_train_user, background_color='yellow')
        self.btn_ret_train = Button(text="Trở Về", size_hint=(1, 0.1), on_press=self.go_back_dashboard, background_color='orange')
        self.label_train = Label(text="Trang Học Người Dùng", font_size="15sp", size_hint=(1,0.1))

        self.id_input_box = BoxLayout(orientation='horizontal', size_hint=(1,0.1))
        self.id_lb = Label(text="Id", size_hint=(0.1,1))
        self.input_id = TextInput( multiline=False, size_hint=(0.9,1), readonly=True)
        self.id_input_box.add_widget(self.id_lb)
        self.id_input_box.add_widget(self.input_id)

        self.name_input_box = BoxLayout(orientation='horizontal', size_hint=(1,0.1))
        self.name_lb = Label(text="Name", size_hint=(0.1,1))
        self.input_name = TextInput(multiline=False, size_hint=(0.9,1), hint_text="Enter Your Name...")
        self.name_input_box.add_widget(self.name_lb)
        self.name_input_box.add_widget(self.input_name)

        self.train_screen.add_widget(self.label_train)
        self.train_screen.add_widget(self.train_cam)
        self.train_screen.add_widget(self.id_input_box)
        self.train_screen.add_widget(self.name_input_box)
        self.train_screen.add_widget(self.btn_submit)
        self.train_screen.add_widget(self.btn_ret_train)
        self.main_layout.add_widget(self.dashboard)


        self.capture = None
        Clock.schedule_interval(self.update, 1.0/33.0)

        self.train = False

        self.num_train = 0

        self.first = False
        self.recognizer = Recognize()
      


        return self.main_layout

    def begin_train_user(self, obj):
        if self.input_id.text == "" or self.input_name.text == "":
            self.label_train.text = "Vui Lòng Điền Đầy Đủ Thông Tin"
            self.label_train.color = "red"
            return
        user_id = self.input_id.text
        user_name = self.input_name.text
        self.add_to_csv(user_id, user_name)
        self.label_train.text = "Tiến Hành Chụp Hình"
        self.label_train.color = 'white'
        self.train = True
    
    def train_models_again(self, obj):
        self.recognizer.train_model()
        self.recognizer.__init__()

    def change_recog_screen(self, obj):
        self.main_layout.remove_widget(self.dashboard)
        self.main_layout.add_widget(self.recog)
        self.capture = cv.VideoCapture(0)
        self.screen = 2
        self.first = True
    
    def go_back_dashboard(self, obj):
        self.main_layout.clear_widgets()
        self.main_layout.add_widget(self.dashboard)
        self.capture = None
        self.screen = 1
    
    def go_train_screen(self, obj):
        self.main_layout.remove_widget(self.dashboard)
        self.main_layout.add_widget(self.train_screen)
        self.label_train.text = "Trang Học Người Dùng"
        self.label_train.color = "white"
        self.capture = cv.VideoCapture(0)
        self.screen = 3
        self.input_id.text = str(self.recognizer.get_next_id())


    def add_to_csv(self, id, name):
        new_user = {
            'id': [id],
            'name': [name]
        }

        user_df = pd.DataFrame(new_user)

        user_df.to_csv('./user.csv', mode='a', header=False, index= False)

    def update(self, dt):
        # display image from cam in opencv window
        if self.capture != None and self.screen == 2:
            ret, frame = self.capture.read()
            frame = self.recognizer.reg_user(frame)
            self.first = False
            buf1 = cv.flip(frame, 0)
            buf = buf1.tobytes()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 

            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.reg_cam.texture = texture1
            # self.train_cam.texture = texture1
        if self.capture != None and self.screen == 3:
            if self.train == False:
                ret, frame = self.capture.read()
                frame = self.recognizer.detect_face(frame)
                buf1 = cv.flip(frame, 0)
                buf = buf1.tobytes()
                texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 

                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                # display image from the texture
                self.train_cam.texture = texture1
            else:
                ret, frame = self.capture.read()

                faces = face_cascade.detectMultiScale(frame, 1.2, 10)
                
                for (x,y,w,h) in faces:
                    cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

                    if not os.path.exists('dataset'):
                        os.makedirs('dataset')
                    if not os.path.exists('dataset/' + self.input_id.text):
                        os.makedirs('dataset/' + self.input_id.text)
                    self.num_train += 1

                    cv.imwrite('dataset/'+self.input_id.text+'/User.'+ self.input_id.text+"." + str(self.num_train)+ ".jpg", frame[y: y+h, x:x+w])

                if self.num_train == 200:
                    self.train = False
                    self.num_train = 0
                    self.label_train.text = "Đang Huấn Luyện Model"
                    self.label_train.color = "yellow"
                    self.input_name = ""
                    self.recognizer.train_model()
                    self.recognizer.__init__()
                    self.label_train.text = "Trang Học Người Dùng"
                    self.label_train.color = "white"
                
                buf1 = cv.flip(frame, 0)
                buf = buf1.tobytes()
                texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 

                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                # display image from the texture
                self.train_cam.texture = texture1

if __name__ == "__main__":
    MainApp().run()