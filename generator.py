import threading
from random import shuffle, seed 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 

from PIL import Image
from PIL.ImageEnhance import Brightness, Contrast

from keras.preprocessing import image

class Generator:  
    
    def __init__(self, labels_file='labels.csv', batch_size=32, val_split = 0.2, train=True, 
                 scale = 3, min_area = 1200, 
                 flip=False, rotate=False, translate=False, brightness=False):
        
        self.batch_size = batch_size  #batch size 
        self.min_area = min_area      #the minimum area of a car in the image in pixels 
        self.lock = threading.Lock()  #for multithreading on next()
        self.image_IDs = None         #the image IDs in the dataframe 
        self.labels = None            #.csv file
        self.val_split = val_split    #validatin split fraction 
        self.train = train            #float - whether this is a training or validation generator 
        self.scale = tuple([int(x/scale) for x in (1200, 1920)])
        
        # call to set up labels
        self.setup_data(labels_file)
        self.num_batches = int(np.ceil(len(self.image_IDs)/ self.batch_size)) #number of batches 
        
        #variables holding temporary data 
        self.img = None #the current input image  
        self.curr_img_info = None #all label info of the current image
        self.mask = None #mask of current image 
        
        #image augmentation 
        self.flip = flip
        self.rotate = rotate 
        self.brightness = brightness
        self.translate = translate
        self.non = lambda s: s if s<0 else None
        self.mom = lambda s: max(0,s)
        
        #for frame shift of images in list img_dir
        self.start = 0
        self.end = self.batch_size 
        
        #init output batch
        self.X_batch = np.zeros((batch_size, self.scale[0], self.scale[1], 3), dtype=np.uint8)
        self.y_batch = np.zeros((batch_size, self.scale[0], self.scale[1], 1), dtype=np.uint8)


    def setup_data(self, labels_file):
        '''get and verify images are in directory, labels are correct '''
        labels = pd.read_csv(labels_file)
        self.labels = labels[labels.Label != 'Pedestrian']
        self.image_IDs = self.labels.Frame.unique() #get all jpg file names 

        #create training split 
        if self.train: 
            self.image_IDs = self.image_IDs[:int(len(self.image_IDs) *(1-self.val_split))]
        else:  
            self.image_IDs = self.image_IDs[int(len(self.image_IDs) *(1-self.val_split)):]
        
        seed(0)
        shuffle(self.image_IDs)
        assert len(self.image_IDs) > 0, 'no images found, check directory'
        
    def read_image(self, img_name): 
        '''read in the image and color correction'''
        self.curr_img_info = self.labels[self.labels.Frame == img_name] #all IDs in that jpg
        im_path = os.path.join('../crowdai', img_name)
        self.img = image.load_img(im_path)
    
    def get_area(self, x): return (x.ymax - x.xmax) * (x.ymin - x.xmin)
    
    def create_mask(self): 
        '''create a vehicle mask from the image'''
        self.mask = np.zeros(shape=(1200, 1920))
    
        for i in range(self.curr_img_info.shape[0]):
            vehicle_ID = self.curr_img_info.iloc[i]
            
            #thresold small cars out 
            area = self.get_area(vehicle_ID)
            if area > self.min_area: 
                self.mask[vehicle_ID.xmax:vehicle_ID.ymax,
                          vehicle_ID.xmin:vehicle_ID.ymin ] = 1
        
        self.mask = np.expand_dims(self.mask, axis=2)
        self.mask = image.array_to_img(self.mask)
            
    def flip_img(self):
        '''50/50 odds to randomly flip the image'''
        if np.random.randint(0, 2): 
            self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
            self.mask = self.mask.transpose(Image.FLIP_LEFT_RIGHT)
            
    def rotate_img(self): 
        '''50/50 odds to rotate the image and mask by +/- 6 degrees'''
        if np.random.randint(0, 2):
            angle = np.random.random() * 12 - 6
            self.img = self.img.rotate(angle)
            self.mask = self.mask.rotate(angle)

    def translate_img(self):
        '''50/50 odds to shift the img asn mask
            by +/- 50 pixels in x and/or y direction'''
        if np.random.randint(0, 2):
            # (ax+by+c, dx+ey+f)
            # c, f = left/right, up.down
            a, b, d, e = 1, 0, 0, 1
            c, f = np.random.uniform(-50, 50), np.random.uniform(-50, 50)
            self.img = self.img.transform(self.img.size, Image.AFFINE, (a, b, c, d, e, f))
            self.mask = self.mask.transform(self.img.size, Image.AFFINE, (a, b, c, d, e, f))
            
    def jitter_brightness(self):
        '''jitter the brightness of img'''
        if np.random.randint(0, 2) == 0:
            self.img = Contrast(self.img).enhance(np.random.uniform(.5, 2.2))
        if np.random.randint(0, 2) == 0:
            self.img = Brightness(self.img).enhance(np.random.uniform(.5, 1.5))
        
    def process(self): 
        '''resize and convert PIL images to arrays'''
        self.img = self.img.resize(self.scale[::-1])
        self.mask = self.mask.resize(self.scale[::-1])
        
        #scale from 255 -> 1 to match output of old generator 
        self.img = image.img_to_array(self.img)
        self.mask = image.img_to_array(self.mask).reshape(self.scale) / 255
        
    def __next__(self): 
        '''Yields data tensor of size [batch_size, 1200, 1920, 1], 
        label tensor of size [batch_size, 1]. GPU compatible. '''

        #lock and release threads at iteration execution 
        with self.lock:      
            for i in range(self.num_batches):
                img_batch_files = self.image_IDs[self.start:self.end]
                
                for j, img_name in enumerate(img_batch_files): 
                    
                    self.read_image(img_name)
                    self.create_mask()
                    
                    #augment image and mask 
                    if self.flip: self.flip_img() 
                    if self.translate: self.translate_img() 
                    if self.rotate: self.rotate_img()
                    if self.brightness: self.jitter_brightness()
        
                    #resize and cvt to array 
                    self.process()
                    
                    #for debugging 
                    #print (j, img_name)
                    # plt.imshow(self.img)
                    # plt.show()
                    # plt.imshow(self.mask)
                    # plt.show()
              
                    self.X_batch[j, :, :, :] = self.img.reshape(self.scale[0], self.scale[1], 3)
                    self.y_batch[j, :, :, :] = self.mask.reshape(self.scale[0], self.scale[1], 1)

                #clip last batch 
                if i == self.num_batches - 1:
                    self.X_batch = self.X_batch[:j, :, :, :]       

                #increment images for next iteration 
                self.start += self.batch_size
                self.end += self.batch_size
                
                return  self.X_batch, self.y_batch
                
    def __iter__(self):
        return self