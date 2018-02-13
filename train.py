import os 
from generator import Generator
import unet 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

############ GLOBAL VARS #######################################################
# need same batchsize for train and validation - for batchnorm statistics 
BATCHSIZE = 8
SCALE = 3
EPOCHS = 50
PATIENCE = 5

############# CONTROL MODEL ######################################################

model_name = 'control_model'
train = True
if train: 

    cp_path = os.path.join('/output' '{}-checkpoint.h5'.format(model_name)) #checkpoint path
    model_path = os.path.join('/output','{}.h5'.format(model_name)) #final model path 

    #create generators 
    train_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=True)
    val_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=False)

    # test on subset for speed 
    # train_gen.image_IDs = train_gen.image_IDs[:8]
    # val_gen.image_IDs = val_gen.image_IDs[:8]

    # build model
    model = unet.build_model(scale_factor = SCALE)
    model.name = model_name

    # train!     
    earlystop = EarlyStopping(patience=PATIENCE, verbose=1)
    checkpoint = ModelCheckpoint(cp_path, verbose=0, save_best_only=True)
    
    model_hist = model.fit_generator(generator = train_gen,
                               steps_per_epoch = len(train_gen.image_IDs)//train_gen.batch_size,
                               validation_data = val_gen,
                               validation_steps = len(val_gen.image_IDs)//val_gen.batch_size, 
                               epochs = 50,
                               callbacks=[earlystop, checkpoint],
                               verbose = 2)

    model.save_weights(model_path)
    print("{} saved to disk".format(model.name))

    print (model_hist.history)
    # unet.plot_history(model_hist, metric = 'loss')
    # unet.plot_history(model_hist, metric = 'IOU_loss')

################### REGULARIZED MODEL #############################################

model_name = 'model_V1'
train = True
if train:

    cp_path = os.path.join('/output', '{}-checkpoint.h5'.format(model_name)) #checkpoint path
    model_path = os.path.join('/output','{}.h5'.format(model_name)) #final model path 

    #create generators 
    train_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=True, flip=True, translate=True)
    val_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=False)

    # build model
    model = unet.build_model(scale_factor = SCALE, batchnorm=True, droprate=0.25)
    model.name = model_name
        
    # train! 
    earlystop = EarlyStopping(patience=PATIENCE, verbose=1)
    checkpoint = ModelCheckpoint(cp_path, verbose=0, save_best_only=True)
    
    model_hist = model.fit_generator(generator = train_gen,
                               steps_per_epoch = len(train_gen.image_IDs)//train_gen.batch_size,
                               validation_data = val_gen,
                               validation_steps = len(val_gen.image_IDs)//val_gen.batch_size, 
                               epochs = EPOCHS,
                               callbacks=[earlystop, checkpoint],
                               verbose = 2)

    model.save_weights(model_path)
    print("{} saved to disk".format(model.name))

    # unet.plot_history(model_hist, metric = 'loss')
    # unet.plot_history(model_hist, metric = 'IOU_loss')

################### DONE ####################################################################