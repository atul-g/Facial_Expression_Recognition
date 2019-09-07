

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, AvgPool2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam



#Since I did the training in Google Colab, I had to upload the dataset in zip format. Below code extracts the folder.
'''
zipobj=ZipFile(os.path.abspath('.')+'/drive/My Drive/facial_expression.zip', 'r')
zipobj.extractall()
'''

img_gen_train=ImageDataGenerator(rescale=1./255, rotation_range=40, horizontal_flip=True)
train_gen=img_gen_train.flow_from_directory(batch_size=64, directory=os.path.abspath('.')+'/fac_exp_no_test', shuffle=True, target_size=(48, 48), class_mode='sparse', color_mode='grayscale')

final_model=tf.keras.Sequential([
                           tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)),
                           tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.MaxPool2D((2,2), (2,2)),
                           tf.keras.layers.Dropout(0.5),

                           tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.MaxPool2D((2,2), (2,2)),
                           tf.keras.layers.Dropout(0.5),
                           
                           tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.MaxPool2D((2,2), (2,2)),
                           tf.keras.layers.Dropout(0.5),

                           tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.MaxPool2D((2,2), (2,2)),
                           tf.keras.layers.Dropout(0.5),


                           tf.keras.layers.Flatten(),

                           tf.keras.layers.Dense(512, activation='relu'),
                           tf.keras.layers.Dropout(0.4),

                           tf.keras.layers.Dense(256, activation='relu'), 
                           tf.keras.layers.Dropout(0.4),
                           
                           tf.keras.layers.Dense(128, activation='relu'),
                           tf.keras.layers.Dropout(0.5),

                           tf.keras.layers.Dense(7, activation='softmax')

])
final_model.summary()

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')


final_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
epochs=100
fin_history=final_model.fit_generator(train_gen, epochs=epochs, callbacks=[lr_reducer, early_stopper])
final_model.save('fac_exp_mod.h5')


#Plotting the graphs for validation and training accuracies and losses
plt.plot(fin_history.history['acc'])
plt.plot(fin_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(fin_history.history['loss'])
plt.plot(fin_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
