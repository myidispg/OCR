from scipy import io as spio
import numpy as np
import pandas as pd
import gc

emnist = spio.loadmat("../Datasets/matlab/emnist-byclass.mat")
# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.int64)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1].astype(np.int64)

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.int64)

# load test labels
y_test_2d = emnist["dataset"][0][0][1][0][0][1]

del emnist
gc.collect()

# Function to invert pixels and append them.
def invert_append(pix_array):
    invert_array = []
    for i in range(pix_array.shape[0]):
        row = []
        for j in range(len(pix_array[i])):
            if j != 784:
                row.append(255 - pix_array[i][j])
            else:
                row.append(pix_array[i][j])
        invert_array.append(row)
    print(len(invert_array))
    invert_array = np.asarray(invert_array) 
    pix_array = np.concatenate((pix_array, invert_array))
    print(len(pix_array))
    return pix_array

# Convert the train and test data to pandas dataframe and then write to a csv file. 
df_test = pd.DataFrame(x_test)
del x_test
df_y_test = pd.DataFrame(y_test_2d)
del y_test_2d

df_test = pd.concat([df_test, df_y_test], axis=1)
del df_y_test

# Convert to numpy array
X_test = df_test.iloc[:,:].values
del df_test
# Get a numpy array with inverted values
X_test = invert_append(X_test)
    
df_test= pd.DataFrame(X_test)

del X_test
gc.collect()

df_test.to_csv('Test_Images_with_labels_invert.csv', index=False)
del df_test
gc.collect()


df_train = pd.DataFrame(x_train)
del x_train
df_y_train = pd.DataFrame(y_train)
del y_train

gc.collect()

df_train = pd.concat([df_train, df_y_train], axis=1)
del df_y_train

df_train.to_csv('Train_Images_with_labels.csv', index=False)
del df_train
gc.collect()