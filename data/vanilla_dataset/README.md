# Data Format

Data is stored in a NUM_DATAPOINTS x 10 csv

first 6 columns represent train_x data (1 x 6)
last 4 columns represent the train_y data (1 x 4)

To use, use the python command: 

savedData = np.loadtxt('vanilla_dataset/data.csv', delimiter=',')

- adjust the path as necessary 