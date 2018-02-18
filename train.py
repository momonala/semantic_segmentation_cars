import os 
import unet 
from generator import Generator
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

############ GLOBAL VARS #######################################################
# need same batchsize for train and validation - for batchnorm statistics 
BATCHSIZE = 2
SCALE = 3
EPOCHS = 50
PATIENCE = 3
OUTDIR = '/output'

#################################################################################
def run(model_name, model_type = None, loss=unet.IOU_inverse,
        flip=False, translate=False, rotate=False, brightness=False, #augmentation 
        batchnorm=False, droprate=None, regularizer=None, #model 
        ):
    
    print (model_name)

    cp_path = os.path.join(OUTDIR, '{}-checkpoint.h5'.format(model_name)) #checkpoint path
    model_path = os.path.join(OUTDIR,'{}.h5'.format(model_name)) #final model path 

    #create generators 
    train_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=True, flip=flip, translate=translate, rotate=rotate, brightness=brightness)
    val_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=False)

    # build model
    model = unet.build_model(scale_factor = SCALE, verbose=1, loss=loss, batchnorm=batchnorm, droprate=droprate, regularizer=regularizer)
    model.name = model_name
        
    # train! 
    earlystop = EarlyStopping(patience=PATIENCE, verbose=1)
    checkpoint = ModelCheckpoint(cp_path, verbose=0, save_best_only=True)
    
    model_hist = model.fit_generator(generator = train_gen,
                            steps_per_epoch = len(train_gen.image_IDs)//train_gen.batch_size,
                            validation_data = val_gen,
                            validation_steps = len(val_gen.image_IDs)//val_gen.batch_size, 
                            epochs = EPOCHS,
                            callbacks = [earlystop, checkpoint],
                            verbose = 2)

    #save weights
    model.save(model_path)
    print("{} saved to disk".format(model.name))
        
    print (model_hist.history, '\n')


############# UNET CONTROL MODEL ######################################################
train = False
if train: 
   run(model_name = 'control_model', model_type=unet)

################### UNET REGULARIZED MODEL 1 #############################################

# add batchnorm 
# add dropout = 0.25
# add some image augmentation 

train = False
if train: 
   run(model_name ='model_V1', model_type=unet,
        flip=True, translate=True,
        batchnorm=True, droprate=0.25)
    
################### UNET REGULARIZED MODEL 2 #############################################

# increase droprate 0.25 --> 0.40 
# add rotation, brightness augmentation with PIL
# add regularizer l2 to all conv2d and convd_transpose layers 

train = False
if train: 
   run(model_name ='model_V2', model_type=unet, 
         flip=True, translate=True, rotate=True, brightness=True,
         batchnorm=True, droprate=0.40, regularizer=regularizers.l2(0.01))

############# UNET CONTROL MODEL BINARY CROSS ENTROPY ######################################################
train = True
if train: 
   run(model_name = 'control_model', model_type=unet, loss='binary_crossentropy')

################### UNET REGULARIZED MODEL 1 BINARY CROSS ENTROPY  #############################################

# droprate 0.40 
# rotation, brightness augmentation with PIL
# regularizer l2 to all conv2d and convd_transpose layers 

train = True
if train: 
   run(model_name ='model_V2', model_type=unet, loss='binary_crossentropy',
        flip=True, translate=True, rotate=True, brightness=True,
        batchnorm=True, droprate=0.40, regularizer=regularizers.l2(0.01))
