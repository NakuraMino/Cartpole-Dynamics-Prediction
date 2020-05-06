# Cartpole_preds_with_nn.py 
execute with: python cartpole_preds_with_nn.py
Used to generate the videos displaying the outputs of CartpoleNetLite. The predict_gp() function was modified to take in a matrix of size 46x1x5x128x128. Each cartpole image is a 128x128 gray scale image. The neural network predicts on 5 images at a time, and since there are 50 time stamps per epoch, each epoch results in 46 datapoints to predict on. 

# cartpolenetlite.py 

The python file with the architecture for cartpolenetlite. The Jupyter Notebook file can be found under models/

# models/CartpoleNet3.pth

The pytorch file with the trained cartpolenetlite. 

# models/CartpoleNet3needsmoretrainig.pth

as the name implies, it wasn't trained enough and didn't predict very well.

# models/errors.txt 

just kept track of the errors during training so I can use them for graphs later on.
