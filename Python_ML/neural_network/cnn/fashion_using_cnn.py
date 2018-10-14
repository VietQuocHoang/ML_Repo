import numpy as np
import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Input, Model
from keras.layers import (Dense, Dropout, Flatten,
                          Conv2D, MaxPooling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report

(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

print("Training data shape: ", train_X.shape, train_y.shape)
print("Test data shape: ", test_X.shape, test_y.shape)

classes = np.unique(train_y)
nClasses = len(classes)
print("Total number of input: ", nClasses)
print("Output classes: ", classes)

plt.figure(figsize=[5, 5])

plt.subplot(121)
plt.imshow(train_X[0, :, :], cmap='gray')
plt.title("Ground truth: {}".format(train_y[0]))

plt.subplot(122)
plt.imshow(test_X[0, :, :], cmap='gray')
plt.title("Ground truth: {}".format(test_y[0]))

plt.show()

print("-----Reshaping-----")
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X.shape, test_X.shape

print("Train_X shape: ", train_X.shape)
print("Test_X shape: ", test_X.shape)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X/255.
test_X = test_X/255.

train_y_one_hot = to_categorical(train_y)
test_y_one_hot = to_categorical(test_y)

print("Original label: ", train_y[0])
print("After conversion: ", train_y_one_hot[0])

train_X, valid_X, train_label, valid_label = train_test_split(
    train_X, train_y_one_hot, test_size=0.2, random_state=13)

batch_size = 64
epoch = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(
    3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2), padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))
fashion_model.summary()

fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,
                                  epochs=epoch, verbose=1,
                                  validation_data=(valid_X, valid_label))

fashion_model.save("fashion_model_dropout.h5py")

test_eval = fashion_model.evaluate(test_X, test_y_one_hot, verbose=0)

print('Test loss: ', test_eval[0])
print('Test accuracy: ', test_eval[1])

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation loss')
plt.title("Training and validation loss")
plt.legend()
plt.show()

predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
correct = np.where(predicted_classes == test_y)[0]
print("Found %d correct labels " % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[correct].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title("Predicted {}, class {}".format(
        predicted_classes[correct], test_y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes != test_y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[incorrect].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(
        predicted_classes[incorrect], test_y[incorrect]))
    plt.tight_layout()

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_y, predicted_classes,
                            target_names=target_names))
