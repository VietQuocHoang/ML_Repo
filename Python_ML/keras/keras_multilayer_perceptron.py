import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras import metrics

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print('x_train shape: \t', x_train.shape)
print('y_train shape: \t', y_train.shape)
print('x_test shape: \t', x_test.shape)
print('y_test shape: \t', y_test.shape)

x_train = x_train/255
x_test = x_test/255
num_classes = 10

y_train = keras.utils.to_categorical(y=y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y=y_test, num_classes=num_classes)

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print('Test loss: %.4f' % score[0])
print('Test accuracy loss: %.4f' % score[1])
