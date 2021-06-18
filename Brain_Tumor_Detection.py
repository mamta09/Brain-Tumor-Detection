import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from tensorflow.keras.models import load_model
import imutils
import numpy as np



class gui(tkinter.Tk):

    def __init__(self):
        super().__init__()
        self.geometry('1200x700')
        
        self.labelFrame =tkinter.Label(self, text = "Brain Tumor Detection",height=1, width=40)
        self.labelFrame.place(x=150,y=30)
        self.labelFrame.configure(background="White", font=("Comic Sans MS Underline", 30, "bold"))
        

        btnk=tkinter.Button(self,text="Segmented image",font=('times',15,'bold'),width=15,command=self.kmean)
        btnk.place(x=420,y=600)
        btnClose = tkinter.Button(self, text="Close",font=('times',15,'bold'), width=8,command=self.destroy)
        btnClose.place(x=1020, y=600)
        btnView = tkinter.Button(self, text="View Tumor Area",font=('times',15,'bold'), width=16,command=self.colorImage)
        btnView.place(x=620, y=600)
        
        btnb=tkinter.Button(self,text="Browse",width=10,font=('times',15,'bold'),command=self.browseWindow)
        btnb.place(x=250,y=600)
        
        btnp=tkinter.Button(self,text="Predict",font=('times',15,'bold'),width=10,command=self.predict)
        btnp.place(x=850,y=600)
        
    def browseWindow(self):
        global mriImage
        FILEOPENOPTIONS = dict(defaultextension='*.*',
                               filetypes=[('jpg', '*.jpg'), ('png', '*.png'), ('jpeg', '*.jpeg'), ('All Files', '*.*')])
        self.fileName = filedialog.askopenfilename(**FILEOPENOPTIONS)
        image = Image.open(self.fileName)
        imageName = str(self.fileName)
        mriImage = cv2.imread(imageName, 1)
        self.display()
        
    def display(self):
        global mriImage
        mriImage = cv2.resize(mriImage , (250,250))
        photo=ImageTk.PhotoImage(image=Image.fromarray(mriImage))
        panelC =tkinter.Label(image=photo)
        panelC.image=photo
        panelC.place(x=50,y=200)
        
    
        
        
         
    def predictTumor(self,image):
     model = load_model('brain_tumor_detector.h5')
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
     thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
     thresh = cv2.erode(thresh, None, iterations=2)
     thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     cnts = imutils.grab_contours(cnts)
     c = max(cnts, key=cv2.contourArea)
     
     

    # Find the extreme points
     extLeft = tuple(c[c[:, :, 0].argmin()][0])
     extRight = tuple(c[c[:, :, 0].argmax()][0])
     extTop = tuple(c[c[:, :, 1].argmin()][0])
     extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
     new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
     
     
     image = cv2.resize(image, dsize=(240, 240))
     image = image / 255.

     image = image.reshape((1, 240, 240, 3))
     res = model.predict(image)

     return res

    def kmean(self):
      global mriImage
      global segmented_image
      ret,img = cv2.threshold(mriImage,127,255,cv2.THRESH_TOZERO)
      pixel_values = img.reshape((-1, 3))
      pixel_values = np.float32(pixel_values)
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
      k = 2

      _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

      centers = np.uint8(centers)
      labels = labels.flatten()
      segmented_image = centers[labels.flatten()]
      segmented_image = segmented_image.reshape(img.shape)
      
     
      segmented_image = cv2.resize(segmented_image , (250,250))
      photo=ImageTk.PhotoImage(image=Image.fromarray(segmented_image))
      panelA =tkinter.Label(image=photo)
      panelA.image=photo
      panelA.place(x=450,y=200)  
      
      
    def colorImage(self):
      global segmented_image
      segmented_image1 = np.copy(segmented_image)
      segmented_image1 = cv2.erode(segmented_image1, None, iterations=2)
      segmemted_image1= cv2.dilate(segmented_image1, None, iterations=2)
      segmented_image1=cv2.cvtColor(segmented_image1,cv2.COLOR_RGB2GRAY)
      segmented_image1 = cv2.resize(segmented_image1 , (250,250))
      photo=ImageTk.PhotoImage(image=Image.fromarray(segmented_image1))
      panelB =tkinter.Label(image=photo)
      panelB.image=photo
      panelB.place(x=850,y=200)
      

        
    def predict(self):
           global mriImage
           res=self.predictTumor(mriImage)

           if res > 0.5:
                resLabel = tkinter.Label(self, text="Tumor Detected", height=3, width=30)
                resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="red")
                resLabel.place(x=400,y=500)
            
           else:
                resLabel = tkinter.Label(self, text="Tumor Not Detected", height=3, width=30)
                resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="Green")
                resLabel.place(x=400,y=500)   
  
    

        
if __name__ == "__main__":
    app = gui()
    app.mainloop()

