import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras.layers import *
from keras.utils import to_categorical
import os
import json
import zipfile

# Create a Kaggle directory
os.makedirs('/root/.kaggle', exist_ok=True)

# Move the kaggle.json file to the Kaggle directory
subprocess.run(['mv', 'kaggle.json', '/root/.kaggle/'])

# Verify the API key is set up correctly
with open('/root/.kaggle/kaggle.json', 'r') as file:
    kaggle_json = json.load(file)

# Download the dataset
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'crawford/emnist'])

# Extract the dataset
with zipfile.ZipFile('emnist.zip', 'r') as zip_ref:
    zip_ref.extractall('emnist')

# Load the training data
train_df = pd.read_csv('emnist/emnist-balanced-train.csv', header=None)
train_df.head()

# Check the shape of the training data
train_df.shape

# Separate features and labels
X_train = train_df.loc[:, 1:]
y_train = train_df.loc[:, 0]

X_train.shape, y_train.shape

X_train.head()
y_train.head()

# Load the label mapping
label_map = pd.read_csv("emnist/emnist-balanced-mapping.txt", delimiter=' ', index_col=0, header=None)
label_map = label_map.squeeze("columns")
label_map.head()

# Create a dictionary for label mapping
label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

# Size of the image
W = 28
H = 28

# Function to reshape and rotate the image
def reshape_and_rotate(image):
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# Apply the transformation to the entire dataset
X_train = np.apply_along_axis(reshape_and_rotate, 1, X_train.values)

# Normalize the dataset
X_train = X_train.astype('float32') / 255

# Get the number of unique classes
number_of_classes = y_train.nunique()

# Convert labels to categorical format
y_train = to_categorical(y_train, number_of_classes)

# Reshape the dataset to fit the model input shape
X_train = X_train.reshape(-1, W, H, 1)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.1,random_state=88)

# Build the model
model = Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(W, H, 1)))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(number_of_classes, activation='softmax'))

# Display the model summary
model.summary()

# Compile the model
optimizer_name = 'adam'
model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
mcp_save = ModelCheckpoint('my_model.keras', save_best_only=True, monitor='val_loss', verbose=1, mode='auto')

# Train the model
history = model.fit(X_train,y_train,epochs=30,batch_size=32,verbose=1,validation_split=0.1,callbacks=[early_stopping, mcp_save])

# Function to plot accuracy and loss
def plotgraph(epochs, acc, val_acc):
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

# Plot accuracy and loss curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plotgraph(epochs, acc, val_acc)
plotgraph(epochs, loss, val_loss)

# Load the best model
model = load_model('my_model.keras')
model.summary()

# Predict on the validation set
y_pred = model.predict(X_val)

# Display a few predictions
for i in range(10, 16):
    plt.subplot(380 + (i % 10 + 1))
    plt.imshow(X_val[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.title(label_dictionary[y_pred[i].argmax()])

for i in range(42, 48):
    plt.subplot(380 + (i % 10 + 1))
    plt.imshow(X_val[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.title(label_dictionary[y_pred[i].argmax()])

# Evaluate the model on the validation set
model.evaluate(X_val, y_val)

# Load the test data
test_df = pd.read_csv('emnist/emnist-balanced-test.csv', header=None)

test_df.describe()

# Separate features and labels in the test data
X_test = test_df.loc[:, 1:]
y_test = test_df.loc[:, 0]

# Apply the transformation to the test data
X_test = np.apply_along_axis(reshape_and_rotate, 1, X_test.values)
y_test = to_categorical(y_test, number_of_classes)

# Normalize the test data
X_test = X_test.astype('float32') / 255

# Reshape the test data to fit the model input shape
X_test = X_test.reshape(-1, W, H, 1)

# Evaluate the model on the test set
model.evaluate(X_test, y_test)

# Save the model as model.h5
model.save('model.h5')

print("Model saved successfully!")