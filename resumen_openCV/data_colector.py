import cv2
import time
import os

class DataColector:

    def __init__(self, id_camera=0, dir_name= "dataset",  name="piece", photos_for_video=10, second_until_shot=3):
        self.id_camera = id_camera
        
        self.green, self.gray = (0, 255, 0), (0, 233, 255)
        self.red= (0, 0, 255)
        
        self.i_width, self.i_height = 720, 720
        
        self.take_a_video = False
        self.photos_for_video = photos_for_video
        self.dir_name = dir_name
        self.name = name
        self.n_photos = 0
        self.id_photos = len(os.listdir(self.dir_name)) - 1
        self.shot_time = second_until_shot
        self.shot = second_until_shot
        

    def _mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            if self.i_width - 50 < x < self.i_width - 10 and 10 < y < 50:
                
                self.running = False
            
            elif self.i_width - 100 < x < self.i_width - 60 and 10 < y < 50:
                
                if self.color_bottom2 == self.green:
                    self.color_bottom2 = self.gray
                    self.take_a_video = True
                    self.pTime = time.time()
        
    def run(self):
        
        cap = cv2.VideoCapture(self.id_camera)

        if cap.isOpened():
            cap.open(self.id_camera)

        cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('video', self._mouseEvent)
        
        self.color_bottom1 = self.red
        self.color_bottom2 = self.green
        
        self.running = True
    
        while self.running:
            success, img = cap.read()

            if success:
                self.i_height, self.i_width = img.shape[0:2]
                org_img = img.copy()
                
                self.decorate_window(img)
                
                cv2.imshow('video', img)    
                
                if self.take_a_video:
                    if self.n_photos < self.photos_for_video:
                        self.save_img(org_img)
                    
                    else:
                        self.n_photos = 0
                        self.take_a_video = False
                        self.color_bottom2 = self.green
                     
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
    def decorate_window(self, img):
        if self.shot != self.shot_time:
            cv2.putText(img, f"Shot in: {3 - self.shot}", (10, 150),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        
        cv2.putText(img, f"N photos: {self.id_photos}", (10, 70),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            
        cv2.rectangle(img, (self.i_width - 50, 10, 40, 40), self.color_bottom1, -1)
        cv2.rectangle(img, (self.i_width - 100, 10, 40, 40), self.color_bottom2, -1)
    
    def save_img(self, img):
        self.cTime = time.time()
        self.shot = int(self.cTime - self.pTime)
        
        if self.shot == self.shot_time:
            cv2.imwrite(f"{self.dir_name}/{self.name}_{self.id_photos}.png", img)
            self.id_photos += 1
            self.n_photos += 1   
            self.pTime = time.time()

if __name__ == "__main__":
    # id_camera, mode, min_confidence
    fr = DataColector(dir_name="dataset_CV")
    fr.run()
