import threading
from random import shuffle, seed 
from skimage import transform
import pandas as pd 
import numpy as np 
import cv2 
import os 
import matplotlib.pyplot as plt  


class Generator:  
    
    def __init__(self, labels_file='labels.csv', batch_size=32, 
                 scale = 3, val_split = 0.2, train=True, 
                 flip=False, rotate=False, translate=False, brightness=False):
        
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.image_IDs = None
        self.labels = None
        self.val_split = val_split
        self.train = train
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
        self.img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    
    def create_mask(self): 
        '''create a vehicle mask from the image'''
        self.mask = np.zeros(shape=self.img.shape[:2])
    
        for i in range(self.curr_img_info.shape[0]):
            vehicle_ID = self.curr_img_info.iloc[i]
            self.mask[vehicle_ID.xmax:vehicle_ID.ymax,
                    vehicle_ID.xmin:vehicle_ID.ymin ] = 1
            
    def flip_img(self):
        '''50/50 odds to randomly flip the image and reverse the steering angle'''
        if np.random.randint(0, 2): 
            self.img = np.flip(self.img, axis=1) 
            self.mask = np.flip(self.mask, axis=1) 
            
    def rotate_img(self): 
        '''50/50 odds to rotate the image by +/- 7 degrees'''
        if np.random.randint(0, 2):
            angle = (np.random.random()-0.5)*14 #[-7, 7]
            self.img = transform.rotate(self.img, angle)
            self.mask = transform.rotate(self.mask, angle)

    def translate_img(self):
        '''50/50 odds to randomly shift the img
            by +/- 100 pixels in x and/or y direction'''
#         https://stackoverflow.com/questions/27087139/shifting-an-image-in-numpy
#         faster than matrix transform 
        if np.random.randint(0, 2):
            ox = int((np.random.random()-0.5)*200)
            oy = int((np.random.random()-0.5)*200)
            shift_img = np.zeros_like(self.img)
            shift_mask = np.zeros_like(self.mask)
            
            #transformation points 
            x1n, x2n, y1n, y2n = self.mom(oy), self.non(oy), self.mom(ox), self.non(ox)
            x1o, x2o, y1o, y2o = self.mom(-oy), self.non(-oy), self.mom(-ox), self.non(-ox)
            
            #perform the shifts 
            shift_img[x1n:x2n, y1n:y2n] = self.img[x1o:x2o, y1o:y2o]
            shift_mask[x1n:x2n, y1n:y2n] = self.mask[x1o:x2o, y1o:y2o]
            self.img = shift_img
            self.mask = shift_mask
            
    def jitter_brightness(self):
        '''jitter the brightness in the HSV space'''
        temp = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        temp[:,:,2] = temp[:,:,2]*random_bright
        self.img = cv2.cvtColor(temp,cv2.COLOR_HSV2RGB)
        
    def resize(self): 
        self.img = cv2.resize(self.img, (self.scale[1], self.scale[0]))
        self.mask = cv2.resize(self.mask, (self.scale[1], self.scale[0]))

    def __next__(self): 
        '''Yields data tensor of size [batch_size, 1200, 1920, 1], 
        label tensor of size [batch_size, 1]. GPU compatible. '''

        #lock and release threads at iteration execution 
        with self.lock:      
            for i in range(self.num_batches):
                img_batch_files = self.image_IDs[self.start:self.end]
                for j, img_name in enumerate(img_batch_files): 
                    # print (j, img_name)
                    self.read_image(img_name)
                    self.create_mask()
                    
                    #augment image and mask 
                    if self.flip: self.flip_img() 
                    if self.translate: self.translate_img() 
#                     if self.rotate: self.rotate_img() #SLOW 
                    if self.brightness: self.jitter_brightness()
                
                    self.resize()
                    
#                     plt.imshow(self.img)
#                     plt.show()
#                     plt.imshow(self.mask)
#                     plt.show()
                                        
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