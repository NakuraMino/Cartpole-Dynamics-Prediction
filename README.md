# CSE571_Project1

NOTE: before you work on anything, please set up a new branch for your own work. We can merge the branch with master if we want to, but I find it really hard to fix master once it becomes dirty. Thanks!

For the TAs:

src/ contains the CNN used to predict the delta states of the cartpole. Run
```
python cartpole_preds_with_nn.py
```
To check out the architecture of CartpoleNetLite, checkout src/models/CartpoleNetLite.ipynb. 

GPNet/ contains the other neural network. Like with the other neural network, run 
```
python cartpole_preds_with_nn.py
```
to see the GPNet in action. 
