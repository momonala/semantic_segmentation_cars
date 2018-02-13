# Semantic Segmentation for Dashcams 

In this project I create a semantic segmentation model for detection of cars on the road from a front-facing dashcam camera on a car. The purpose of this is to play around with self-driving-car technologies. The project is modeled after a similar project undertaken  with Udacity's Self Driving Car Nanodegree, where I built a CNN classifier and ensemble of various other models with HOG feature extraction to create bounding boxes around cars. I decided to take this to the next level with semantic segmentation. 

<img src="/img/inferece.png" width="800" />

## Data 

The datset is from the Udacity/CrowdAI collaboration and[ can be found here](https://github.com/udacity/self-driving-car/tree/master/annotations). It is 1.6 GB and includes driving in Mountain View California and neighboring cities during daylight conditions. It contains over 65,000 labels across 9,423 frames collected from a Point Grey research cameras running at full resolution of 1920x1200 at 2hz. The dataset was annotated by CrowdAI using a combination of machine learning and humans. For training, the dataset is located in a directory called `../crowdai/<images>`

<img src="/img/crowdai.png" width="500" />

## UNet 

The first architecture tried is a UNet, which is an autoecoder network with the added bonus of skip connecetions between the corresponding encoder-decoder units. The model compresses an input image down to a certain representation with convolutions, then with upsampling convolutions, it resizes those representations back into an image mask. Each intermediate image representation is concatenated with the on the upsample/downsample bits with the same image depth. The image from the original paper below shows this nicely. 

<img src="/img/architecture.png" width="800" />

I implemeted the original architecture by Olaf Ronneberger, Philipp Fischer, Thomas Brox, who's use case was biomedical imaging, with a few exceptions. Because of resource constraints, I had a shallow architeture (512 max depth compared to their 1028), and I resized images by a factor of 3 to (360x400x3), which sped up training. I also used a batch size of 8. 

[arXiv link](https://arxiv.org/abs/1505.04597)

### Source Code 

The break down of my code is as follows: 

- model architeture code is in [`/unet.py`'](unet.py)
- training code is in [`train.py`](train.py)
- an image generator with augmentation is in [`generator.py`](generator.py)
- an explanation of how I made the dataset is in [`data_exploration.ipynb`](data_exploration.ipynb)

### Training 

I trained the network on Floydhub with the command `floyd run --gpu --env tensorflow-1.4 --data mnalavadi/datasets/crowdai:/crowdai "python train.py"`. You can replicate the results as long as I have the dataset up, but I may eventually take it down due to resource constraints. If so, you will need to redownload and mount the dataset yourself, but no additional processing is necessary. 

<img src="/img/control_hist.png" width="500" />

<img src="/img/regularized_hist.png" width="500" />

### Evaluation 

## Next Steps: 

- Try with a FCN network!
- add road segmentation as well 