import os 
from generator import Generator
import unet 
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

############ GLOBAL VARS #######################################################
# need same batchsize for train and validation - for batchnorm statistics 
BATCHSIZE = 2
SCALE = 3
EPOCHS = 50
PATIENCE = 3
OUTDIR = 'models' #'/output'

############# CONTROL MODEL ######################################################

model_name = 'control_model'
train = False
if train: 

    cp_path = os.path.join(OUTDIR, '{}-checkpoint.h5'.format(model_name)) #checkpoint path
    model_path = os.path.join(OUTDIR,'{}.h5'.format(model_name)) #final model path 

    #create generators 
    train_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=True)
    val_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=False)

    # test on subset for speed 
    # train_gen.image_IDs = train_gen.image_IDs[:8]
    # val_gen.image_IDs = val_gen.image_IDs[:8]

    # build model
    model = unet.build_model(scale_factor = SCALE, verbose=1)
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

    #save weights 
    model.save_weights(model_path)
    print("{} saved to disk".format(model.name))

    # serialize model to JSON
    json_name = os.path.join(OUTDIR,'{}.json'.format(model_name))
    with open(json_name, "w") as json_file:
        json_file.write(model.to_json())


    print (model_hist.history, '\n')
    # unet.plot_history(model_hist, metric = 'loss')
    # unet.plot_history(model_hist, metric = 'IOU_loss')

################### REGULARIZED MODEL #############################################

# add batchnorm 
# add dropout = 0.25
# add image augmentation 

model_name = 'model_V1'
train = False
if train:

    cp_path = os.path.join(OUTDIR, '{}-checkpoint.h5'.format(model_name)) #checkpoint path
    model_path = os.path.join(OUTDIR,'{}.h5'.format(model_name)) #final model path 

    #create generators 
    train_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=True, flip=True, translate=True)
    val_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=False)

    # build model
    model = unet.build_model(scale_factor = SCALE, batchnorm=True, droprate=0.25, verbose=1)
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

    #save weights
    model.save_weights(model_path)
    print("{} saved to disk".format(model.name))
    
    # serialize model to JSON
    json_name = os.path.join(OUTDIR,'{}.json'.format(model_name))
    with open(json_name, "w") as json_file:
        json_file.write(model.to_json())
    
    print (model_hist.history, '\n')
    # unet.plot_history(model_hist, metric = 'loss')
    # unet.plot_history(model_hist, metric = 'IOU_loss')

################### DONE ####################################################################

################### REGULARIZED MODEL 2 #############################################

# increase droprate 0.25 --> 0.40 
# add regularizer l2 to all conv2d and convd_transpose layers 

model_name = 'model_V2'
train = True
if train:

    cp_path = os.path.join(OUTDIR, '{}-checkpoint.h5'.format(model_name)) #checkpoint path
    model_path = os.path.join(OUTDIR,'{}.h5'.format(model_name)) #final model path 

    #create generators 
    train_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=True, flip=True, translate=True)
    val_gen = Generator(scale = SCALE, batch_size=BATCHSIZE, train=False)

    # build model
    model = unet.build_model(scale_factor = SCALE, batchnorm=True, droprate=0.40, regularizer=regularizers.l2(0.01), verbose=1)
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

    #save weights
    model.save_weights(model_path)
    print("{} saved to disk".format(model.name))
    
    # serialize model to JSON
    json_name = os.path.join(OUTDIR,'{}.json'.format(model_name))
    with open(json_name, "w") as json_file:
        json_file.write(model.to_json())
    
    print (model_hist.history, '\n')
    # unet.plot_history(model_hist, metric = 'loss')
    # unet.plot_history(model_hist, metric = 'IOU_loss')

################### DONE ####################################################################