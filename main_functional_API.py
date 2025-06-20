import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data() 

x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train)

class_names = ["T-shirt/Top","Trouser","Pullover",
              "Dress", "Coat", "Sandal","Shirt","Sneaker","Bag","Ankle Bot"
        ]

uninque,counts=np.unique(y_test, return_counts=True)
print("test has",dict(zip(uninque,counts)))

uninque,counts=np.unique(y_train, return_counts=True)
print("train has",dict(zip(uninque,counts)))

indexes = np.random.randint(0,x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]
print(labels)
plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5,5,i+1)
    image=images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(class_names[labels[i]], fontsize=6)

plt.tight_layout()
plt.show()
plt.savefig("fashion-mnist-samples.png")

for i in labels:
    print(class_names[i])

input_shape=(x_train.shape[1:]+(1,))
num_classes=len(np.unique(y_train))
print(num_classes)
print(input_shape)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

batch_size=128
hidden_units=256
dropout=0.45

input_ =keras.layers.Input(shape=input_shape)
batchnorm=keras.layers.BatchNormalization()(input_)
layer1= keras.layers.Dense(units=128,activation='relu')(batchnorm)
dropout1 = keras.layers.Dropout(0.2)(layer1)
layer2 = keras.layers.Dense(units=128,activation='relu')(dropout1)
dropout2 = keras.layers.Dropout(0.2)(layer2)
flatten = keras.layers.Flatten()(dropout2)
layer4 = keras.layers.Dense(units=128,activation='relu')(flatten)
layer3 = keras.layers.Dense(units=128,activation='relu')(layer4)
output = keras.layers.Dense(units=num_classes,activation='softmax')(layer3)
model = keras.Model( inputs = [input_],outputs=[output])

model.summary()



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(np.expand_dims(x_train,-1),y_train,epochs=5,batch_size=64)
loss,acc = model.evaluate(x_test,y_test,batch_size=batch_size)