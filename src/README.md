# Cartpole_preds_with_nn.py 

execute with: python cartpole_preds_with_nn.py

Used to generate the videos displaying the outputs of the neural networks. The predict_cartpole() function was modified to take in a matrix of size 46x1x5x128x128. Each cartpole image is a 128x128 gray scale image. The neural network predicts on 5 images at a time, and since there are 50 time stamps per epoch, each epoch results in 46 datapoints to predict on. 

# models/

Folder that keeps track of all the models we trained. 
