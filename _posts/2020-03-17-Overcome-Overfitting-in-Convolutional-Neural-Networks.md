
When using convolutional neural networks (CNNs) it’s easy to run into the problem of overfitting. However there are a few things we can do to tackle this problem and here I intend to demonstrate the effectiveness of a couple of them.

Our task is to classify images of cats and dogs, using a convolutional neural network. Sounds easy enough. We start by downloading a dataset(or part of it) from Microsoft Research.



```
from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # Use the %tensorflow_version magic if in colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
```

    TensorFlow 2.x selected.



```
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
print(zip_dir)

```

    /root/.keras/datasets/cats_and_dogs_filterted.zip


Now we have extracted all the data needed, now let's assign each directory to a variable.


```
train_cats_dir = os.path.join(train_dir, 'cats')  
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats') 
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  
```

## Explore the Dataset

Now that we have downloaded the data let's see how many figures we have


 of cats and dogs in the training and validation set.



```
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
```

    total training cat images: 1000
    total training dog images: 1000
    total validation cat images: 500
    total validation dog images: 500
    --
    Total training images: 2000
    Total validation images: 1000


Let's take a look at the dimensions of a couple of figures:


```
import cv2
train_cats_dir = os.path.join(train_dir, 'cats')  

im1 = cv2.imread(train_cats_dir+"/cat.0.jpg")
im2 = cv2.imread(train_cats_dir+"/cat.1.jpg")

print(type(im1), type(im2))
print(im1.shape, im2.shape)

```

    <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    (374, 500, 3) (280, 300, 3)


We see that they don't have the same size. However we can simply resize the figures later. Let's define the target size as 150 pixels.


```
IMG_SHAPE = 150
```

## Data Preparation

Now we would like to read all the images into 3-dimensional tensors (floating point). Then we scale the tensor values from values between 0 and 255 to values between 0 and 1.

We can do just that in a really easy way using the class **tf.keras.preprocessing.image.ImageDataGenerator**.


```
train_image_generator      = ImageDataGenerator(rescale=1./255) 
validation_image_generator = ImageDataGenerator(rescale=1./255) 
```

the method flow_from_directory method will load images from the disk for each batch it will also rescale the pixel values using the rescale factor we defined above, and resize the figures to IMG_SHAPE a using single line of code.


```
BATCH_SIZE = 100 # number of samples to use to calculate the gradient at each update step.
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE), 
                                                           class_mode='binary')
```

    Found 2000 images belonging to 2 classes.



```
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='binary')
```

    Found 1000 images belonging to 2 classes.


Let's display a few of the figures. First we get the next batch from the generator train_data_gen.


```
sample_training_images, _ = next(train_data_gen) 
```


```
def plotImages(images_arr, n):
    fig, axes = plt.subplots(1, n, figsize=(30,30))
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
```


```
plotImages(sample_training_images[:6], 6)
```


![png](images/cats_dogs_1.png)


## Defining the network

The network has four layers of convolution blocks, and we apply a max pool layer after each of the convolution blocks. Finally there is a fully connected layer with 512 units and the final ouput has two units (cats and dogs) which we can view as logits. We can then transform these logits to probabilities using the softmax function.





```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])
```

Next we compile the model, we use the Adam optimizer and since our dataset is balanced then we are interested in seeing the accuracy metric reported after each epoch.



```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 148, 148, 32)      896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 15, 15, 128)       147584    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 6272)              0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               3211776   
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 1026      
    =================================================================
    Total params: 3,453,634
    Trainable params: 3,453,634
    Non-trainable params: 0
    _________________________________________________________________


## Training the model

Next we train the model using the parameters defined before, we use the fit_generator method since we are using a generator to generate the batches of images. Here we track the perfomance of the model on the training data and the validation data.


```
EPOCHS = 100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)
```

    WARNING:tensorflow:From <ipython-input-17-d70e0a339f06>:7: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use Model.fit, which supports generators.
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 20 steps, validate for 10 steps
    Epoch 1/100
    20/20 [==============================] - 14s 706ms/step - loss: 0.7051 - accuracy: 0.5135 - val_loss: 0.6873 - val_accuracy: 0.5000
    Epoch 2/100
    20/20 [==============================] - 8s 417ms/step - loss: 0.6789 - accuracy: 0.5825 - val_loss: 0.6886 - val_accuracy: 0.5330
    Epoch 3/100
    20/20 [==============================] - 8s 416ms/step - loss: 0.6580 - accuracy: 0.6080 - val_loss: 0.6420 - val_accuracy: 0.6380
    Epoch 4/100
    20/20 [==============================] - 8s 411ms/step - loss: 0.6299 - accuracy: 0.6640 - val_loss: 0.6480 - val_accuracy: 0.6370
    Epoch 5/100
    20/20 [==============================] - 8s 413ms/step - loss: 0.5996 - accuracy: 0.6870 - val_loss: 0.5952 - val_accuracy: 0.6930
    Epoch 6/100
    20/20 [==============================] - 8s 409ms/step - loss: 0.5415 - accuracy: 0.7275 - val_loss: 0.5717 - val_accuracy: 0.7150
    Epoch 7/100
    20/20 [==============================] - 8s 416ms/step - loss: 0.5000 - accuracy: 0.7505 - val_loss: 0.5459 - val_accuracy: 0.7370
    Epoch 8/100
    20/20 [==============================] - 8s 411ms/step - loss: 0.4561 - accuracy: 0.7870 - val_loss: 0.5644 - val_accuracy: 0.7330
    Epoch 9/100
    20/20 [==============================] - 8s 415ms/step - loss: 0.4216 - accuracy: 0.7980 - val_loss: 0.5383 - val_accuracy: 0.7460
    Epoch 10/100
    20/20 [==============================] - 8s 413ms/step - loss: 0.3771 - accuracy: 0.8340 - val_loss: 0.5632 - val_accuracy: 0.7280
    Epoch 11/100
    20/20 [==============================] - 8s 413ms/step - loss: 0.3287 - accuracy: 0.8555 - val_loss: 0.5596 - val_accuracy: 0.7500
    Epoch 12/100
    20/20 [==============================] - 8s 416ms/step - loss: 0.2914 - accuracy: 0.8775 - val_loss: 0.6863 - val_accuracy: 0.7260
    Epoch 13/100
    20/20 [==============================] - 8s 415ms/step - loss: 0.2391 - accuracy: 0.9025 - val_loss: 0.6376 - val_accuracy: 0.7440
    Epoch 14/100
    20/20 [==============================] - 8s 415ms/step - loss: 0.1862 - accuracy: 0.9280 - val_loss: 0.6653 - val_accuracy: 0.7190
    Epoch 15/100
    20/20 [==============================] - 8s 414ms/step - loss: 0.1588 - accuracy: 0.9420 - val_loss: 0.7442 - val_accuracy: 0.7370
    Epoch 16/100
    20/20 [==============================] - 8s 411ms/step - loss: 0.1521 - accuracy: 0.9375 - val_loss: 0.7440 - val_accuracy: 0.7420
    Epoch 17/100
    20/20 [==============================] - 8s 415ms/step - loss: 0.1135 - accuracy: 0.9630 - val_loss: 0.8746 - val_accuracy: 0.7320
    Epoch 18/100
    20/20 [==============================] - 8s 412ms/step - loss: 0.0767 - accuracy: 0.9715 - val_loss: 0.8553 - val_accuracy: 0.7610
    Epoch 19/100
    20/20 [==============================] - 8s 414ms/step - loss: 0.0436 - accuracy: 0.9880 - val_loss: 0.9930 - val_accuracy: 0.7590
    Epoch 20/100
    20/20 [==============================] - 8s 414ms/step - loss: 0.0306 - accuracy: 0.9935 - val_loss: 1.1228 - val_accuracy: 0.7440
    Epoch 21/100
    20/20 [==============================] - 8s 417ms/step - loss: 0.0140 - accuracy: 0.9985 - val_loss: 1.1363 - val_accuracy: 0.7630
    Epoch 22/100
    20/20 [==============================] - 8s 415ms/step - loss: 0.0056 - accuracy: 1.0000 - val_loss: 1.2152 - val_accuracy: 0.7580
    Epoch 23/100
    20/20 [==============================] - 8s 413ms/step - loss: 0.0027 - accuracy: 1.0000 - val_loss: 1.3012 - val_accuracy: 0.7720
    Epoch 24/100
    20/20 [==============================] - 8s 412ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 1.3439 - val_accuracy: 0.7670
    Epoch 25/100
    20/20 [==============================] - 8s 414ms/step - loss: 9.6970e-04 - accuracy: 1.0000 - val_loss: 1.3821 - val_accuracy: 0.7720
    Epoch 26/100
    20/20 [==============================] - 8s 413ms/step - loss: 7.6783e-04 - accuracy: 1.0000 - val_loss: 1.4154 - val_accuracy: 0.7710
    Epoch 27/100
    20/20 [==============================] - 8s 413ms/step - loss: 6.2505e-04 - accuracy: 1.0000 - val_loss: 1.4370 - val_accuracy: 0.7700
    Epoch 28/100
    20/20 [==============================] - 8s 410ms/step - loss: 5.3057e-04 - accuracy: 1.0000 - val_loss: 1.4674 - val_accuracy: 0.7670
    Epoch 29/100
    20/20 [==============================] - 8s 409ms/step - loss: 4.5979e-04 - accuracy: 1.0000 - val_loss: 1.4970 - val_accuracy: 0.7710
    Epoch 30/100
    20/20 [==============================] - 8s 416ms/step - loss: 3.9665e-04 - accuracy: 1.0000 - val_loss: 1.5153 - val_accuracy: 0.7690
    Epoch 31/100
    20/20 [==============================] - 8s 417ms/step - loss: 3.5446e-04 - accuracy: 1.0000 - val_loss: 1.5369 - val_accuracy: 0.7700
    Epoch 32/100
    20/20 [==============================] - 8s 410ms/step - loss: 3.1602e-04 - accuracy: 1.0000 - val_loss: 1.5574 - val_accuracy: 0.7710
    Epoch 33/100
    20/20 [==============================] - 8s 410ms/step - loss: 2.9047e-04 - accuracy: 1.0000 - val_loss: 1.5784 - val_accuracy: 0.7720
    Epoch 34/100
    20/20 [==============================] - 8s 412ms/step - loss: 2.6221e-04 - accuracy: 1.0000 - val_loss: 1.5918 - val_accuracy: 0.7690
    Epoch 35/100
    20/20 [==============================] - 8s 414ms/step - loss: 2.3993e-04 - accuracy: 1.0000 - val_loss: 1.6045 - val_accuracy: 0.7720
    Epoch 36/100
    20/20 [==============================] - 8s 417ms/step - loss: 2.1603e-04 - accuracy: 1.0000 - val_loss: 1.6236 - val_accuracy: 0.7690
    Epoch 37/100
    20/20 [==============================] - 8s 413ms/step - loss: 2.0168e-04 - accuracy: 1.0000 - val_loss: 1.6322 - val_accuracy: 0.7720
    Epoch 38/100
    20/20 [==============================] - 8s 407ms/step - loss: 1.8368e-04 - accuracy: 1.0000 - val_loss: 1.6499 - val_accuracy: 0.7700
    Epoch 39/100
    20/20 [==============================] - 8s 408ms/step - loss: 1.6933e-04 - accuracy: 1.0000 - val_loss: 1.6640 - val_accuracy: 0.7710
    Epoch 40/100
    20/20 [==============================] - 8s 416ms/step - loss: 1.5847e-04 - accuracy: 1.0000 - val_loss: 1.6747 - val_accuracy: 0.7690
    Epoch 41/100
    20/20 [==============================] - 8s 408ms/step - loss: 1.4691e-04 - accuracy: 1.0000 - val_loss: 1.6868 - val_accuracy: 0.7700
    Epoch 42/100
    20/20 [==============================] - 8s 408ms/step - loss: 1.3784e-04 - accuracy: 1.0000 - val_loss: 1.7034 - val_accuracy: 0.7700
    Epoch 43/100
    20/20 [==============================] - 8s 410ms/step - loss: 1.2832e-04 - accuracy: 1.0000 - val_loss: 1.7099 - val_accuracy: 0.7680
    Epoch 44/100
    20/20 [==============================] - 8s 413ms/step - loss: 1.2130e-04 - accuracy: 1.0000 - val_loss: 1.7192 - val_accuracy: 0.7690
    Epoch 45/100
    20/20 [==============================] - 8s 410ms/step - loss: 1.1370e-04 - accuracy: 1.0000 - val_loss: 1.7340 - val_accuracy: 0.7680
    Epoch 46/100
    20/20 [==============================] - 8s 417ms/step - loss: 1.0617e-04 - accuracy: 1.0000 - val_loss: 1.7422 - val_accuracy: 0.7680
    Epoch 47/100
    20/20 [==============================] - 8s 410ms/step - loss: 1.0099e-04 - accuracy: 1.0000 - val_loss: 1.7535 - val_accuracy: 0.7680
    Epoch 48/100
    20/20 [==============================] - 8s 406ms/step - loss: 9.5048e-05 - accuracy: 1.0000 - val_loss: 1.7628 - val_accuracy: 0.7680
    Epoch 49/100
    20/20 [==============================] - 8s 408ms/step - loss: 9.0130e-05 - accuracy: 1.0000 - val_loss: 1.7704 - val_accuracy: 0.7660
    Epoch 50/100
    20/20 [==============================] - 8s 405ms/step - loss: 8.5485e-05 - accuracy: 1.0000 - val_loss: 1.7811 - val_accuracy: 0.7680
    Epoch 51/100
    20/20 [==============================] - 8s 409ms/step - loss: 8.1116e-05 - accuracy: 1.0000 - val_loss: 1.7925 - val_accuracy: 0.7670
    Epoch 52/100
    20/20 [==============================] - 8s 411ms/step - loss: 7.7838e-05 - accuracy: 1.0000 - val_loss: 1.7979 - val_accuracy: 0.7670
    Epoch 53/100
    20/20 [==============================] - 8s 408ms/step - loss: 7.3498e-05 - accuracy: 1.0000 - val_loss: 1.8076 - val_accuracy: 0.7700
    Epoch 54/100
    20/20 [==============================] - 8s 404ms/step - loss: 7.0162e-05 - accuracy: 1.0000 - val_loss: 1.8151 - val_accuracy: 0.7680
    Epoch 55/100
    20/20 [==============================] - 8s 410ms/step - loss: 6.6616e-05 - accuracy: 1.0000 - val_loss: 1.8236 - val_accuracy: 0.7680
    Epoch 56/100
    20/20 [==============================] - 8s 413ms/step - loss: 6.4590e-05 - accuracy: 1.0000 - val_loss: 1.8301 - val_accuracy: 0.7700
    Epoch 57/100
    20/20 [==============================] - 8s 409ms/step - loss: 6.1270e-05 - accuracy: 1.0000 - val_loss: 1.8412 - val_accuracy: 0.7660
    Epoch 58/100
    20/20 [==============================] - 8s 411ms/step - loss: 5.8172e-05 - accuracy: 1.0000 - val_loss: 1.8500 - val_accuracy: 0.7660
    Epoch 59/100
    20/20 [==============================] - 8s 408ms/step - loss: 5.6200e-05 - accuracy: 1.0000 - val_loss: 1.8535 - val_accuracy: 0.7700
    Epoch 60/100
    20/20 [==============================] - 8s 409ms/step - loss: 5.3802e-05 - accuracy: 1.0000 - val_loss: 1.8644 - val_accuracy: 0.7640
    Epoch 61/100
    20/20 [==============================] - 8s 406ms/step - loss: 5.1548e-05 - accuracy: 1.0000 - val_loss: 1.8696 - val_accuracy: 0.7660
    Epoch 62/100
    20/20 [==============================] - 8s 407ms/step - loss: 4.9285e-05 - accuracy: 1.0000 - val_loss: 1.8745 - val_accuracy: 0.7690
    Epoch 63/100
    20/20 [==============================] - 8s 405ms/step - loss: 4.7639e-05 - accuracy: 1.0000 - val_loss: 1.8829 - val_accuracy: 0.7670
    Epoch 64/100
    20/20 [==============================] - 8s 409ms/step - loss: 4.5998e-05 - accuracy: 1.0000 - val_loss: 1.8907 - val_accuracy: 0.7640
    Epoch 65/100
    20/20 [==============================] - 8s 411ms/step - loss: 4.3797e-05 - accuracy: 1.0000 - val_loss: 1.8964 - val_accuracy: 0.7660
    Epoch 66/100
    20/20 [==============================] - 8s 410ms/step - loss: 4.2450e-05 - accuracy: 1.0000 - val_loss: 1.9036 - val_accuracy: 0.7670
    Epoch 67/100
    20/20 [==============================] - 8s 411ms/step - loss: 4.1141e-05 - accuracy: 1.0000 - val_loss: 1.9093 - val_accuracy: 0.7640
    Epoch 68/100
    20/20 [==============================] - 8s 413ms/step - loss: 3.9484e-05 - accuracy: 1.0000 - val_loss: 1.9139 - val_accuracy: 0.7670
    Epoch 69/100
    20/20 [==============================] - 8s 414ms/step - loss: 3.7653e-05 - accuracy: 1.0000 - val_loss: 1.9205 - val_accuracy: 0.7670
    Epoch 70/100
    20/20 [==============================] - 8s 404ms/step - loss: 3.6372e-05 - accuracy: 1.0000 - val_loss: 1.9268 - val_accuracy: 0.7670
    Epoch 71/100
    20/20 [==============================] - 8s 409ms/step - loss: 3.5042e-05 - accuracy: 1.0000 - val_loss: 1.9326 - val_accuracy: 0.7640
    Epoch 72/100
    20/20 [==============================] - 8s 404ms/step - loss: 3.3825e-05 - accuracy: 1.0000 - val_loss: 1.9387 - val_accuracy: 0.7660
    Epoch 73/100
    20/20 [==============================] - 8s 406ms/step - loss: 3.2899e-05 - accuracy: 1.0000 - val_loss: 1.9461 - val_accuracy: 0.7640
    Epoch 74/100
    20/20 [==============================] - 8s 407ms/step - loss: 3.1653e-05 - accuracy: 1.0000 - val_loss: 1.9517 - val_accuracy: 0.7650
    Epoch 75/100
    20/20 [==============================] - 8s 410ms/step - loss: 3.0598e-05 - accuracy: 1.0000 - val_loss: 1.9551 - val_accuracy: 0.7660
    Epoch 76/100
    20/20 [==============================] - 8s 410ms/step - loss: 2.9703e-05 - accuracy: 1.0000 - val_loss: 1.9620 - val_accuracy: 0.7660
    Epoch 77/100
    20/20 [==============================] - 8s 411ms/step - loss: 2.8771e-05 - accuracy: 1.0000 - val_loss: 1.9655 - val_accuracy: 0.7660
    Epoch 78/100
    20/20 [==============================] - 8s 412ms/step - loss: 2.7807e-05 - accuracy: 1.0000 - val_loss: 1.9720 - val_accuracy: 0.7660
    Epoch 79/100
    20/20 [==============================] - 8s 407ms/step - loss: 2.6900e-05 - accuracy: 1.0000 - val_loss: 1.9773 - val_accuracy: 0.7670
    Epoch 80/100
    20/20 [==============================] - 8s 408ms/step - loss: 2.5944e-05 - accuracy: 1.0000 - val_loss: 1.9828 - val_accuracy: 0.7660
    Epoch 81/100
    20/20 [==============================] - 8s 405ms/step - loss: 2.4907e-05 - accuracy: 1.0000 - val_loss: 1.9914 - val_accuracy: 0.7660
    Epoch 82/100
    20/20 [==============================] - 8s 408ms/step - loss: 2.4131e-05 - accuracy: 1.0000 - val_loss: 2.0003 - val_accuracy: 0.7640
    Epoch 83/100
    20/20 [==============================] - 8s 406ms/step - loss: 2.3009e-05 - accuracy: 1.0000 - val_loss: 2.0086 - val_accuracy: 0.7660
    Epoch 84/100
    20/20 [==============================] - 8s 408ms/step - loss: 2.1957e-05 - accuracy: 1.0000 - val_loss: 2.0133 - val_accuracy: 0.7650
    Epoch 85/100
    20/20 [==============================] - 8s 410ms/step - loss: 2.0569e-05 - accuracy: 1.0000 - val_loss: 2.0286 - val_accuracy: 0.7670
    Epoch 86/100
    20/20 [==============================] - 8s 410ms/step - loss: 1.9244e-05 - accuracy: 1.0000 - val_loss: 2.0388 - val_accuracy: 0.7670
    Epoch 87/100
    20/20 [==============================] - 8s 408ms/step - loss: 1.8113e-05 - accuracy: 1.0000 - val_loss: 2.0527 - val_accuracy: 0.7670
    Epoch 88/100
    20/20 [==============================] - 8s 409ms/step - loss: 1.6763e-05 - accuracy: 1.0000 - val_loss: 2.0660 - val_accuracy: 0.7650
    Epoch 89/100
    20/20 [==============================] - 8s 408ms/step - loss: 1.5258e-05 - accuracy: 1.0000 - val_loss: 2.0823 - val_accuracy: 0.7620
    Epoch 90/100
    20/20 [==============================] - 8s 408ms/step - loss: 1.4234e-05 - accuracy: 1.0000 - val_loss: 2.0939 - val_accuracy: 0.7680
    Epoch 91/100
    20/20 [==============================] - 8s 408ms/step - loss: 1.3230e-05 - accuracy: 1.0000 - val_loss: 2.1036 - val_accuracy: 0.7650
    Epoch 92/100
    20/20 [==============================] - 8s 407ms/step - loss: 1.1979e-05 - accuracy: 1.0000 - val_loss: 2.1224 - val_accuracy: 0.7670
    Epoch 93/100
    20/20 [==============================] - 8s 406ms/step - loss: 1.0822e-05 - accuracy: 1.0000 - val_loss: 2.1374 - val_accuracy: 0.7680
    Epoch 94/100
    20/20 [==============================] - 8s 409ms/step - loss: 9.9499e-06 - accuracy: 1.0000 - val_loss: 2.1465 - val_accuracy: 0.7650
    Epoch 95/100
    20/20 [==============================] - 8s 411ms/step - loss: 9.1220e-06 - accuracy: 1.0000 - val_loss: 2.1655 - val_accuracy: 0.7670
    Epoch 96/100
    20/20 [==============================] - 8s 408ms/step - loss: 8.6043e-06 - accuracy: 1.0000 - val_loss: 2.1729 - val_accuracy: 0.7650
    Epoch 97/100
    20/20 [==============================] - 8s 406ms/step - loss: 7.6894e-06 - accuracy: 1.0000 - val_loss: 2.1856 - val_accuracy: 0.7650
    Epoch 98/100
    20/20 [==============================] - 8s 410ms/step - loss: 7.2230e-06 - accuracy: 1.0000 - val_loss: 2.2059 - val_accuracy: 0.7650
    Epoch 99/100
    20/20 [==============================] - 8s 407ms/step - loss: 6.6712e-06 - accuracy: 1.0000 - val_loss: 2.2149 - val_accuracy: 0.7650
    Epoch 100/100
    20/20 [==============================] - 8s 407ms/step - loss: 6.2036e-06 - accuracy: 1.0000 - val_loss: 2.2236 - val_accuracy: 0.7650


Now let's plot the perfomance of the model on the two data sets.


```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
```


![png](CNN_classification_files/CNN_classification_27_0.png)


We see that after ca. 10 epochs the perfomance, of the model on the training and testing set, starts to diverge. This is a clear indication of overfitting. What we could do stop the training after ca. 10 epochs(early stopping). But let's explore some alternatives to that.

## How to prevent overfitting

When our dataset is limited we often run into the problem of overfitting. But what if we don't have any more data, how can we handle that? One way is to augment the dataset we already have. We can apply random transformations to the images for instance: zoom, rotate and flip them horizontally.

We can apply these transformations using the ImageDataGenerator class.



### Transformations

let's see how the some of the transformations look like:

### Horizontal flip

Here we take one figure and the transformation is applied randomly (or not).


```
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE,IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(6)]
plotImages(augmented_images, 6)
```

    Found 2000 images belonging to 2 classes.



![png](CNN_classification_files/CNN_classification_33_1.png)


### Rotating 

Here we demonstrate how rotate the image up to 45 degrees


```
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))
```

    Found 2000 images belonging to 2 classes.



```
augmented_images = [train_data_gen[0][0][0] for i in range(6)]
plotImages(augmented_images, 6)
```


![png](CNN_classification_files/CNN_classification_36_0.png)


### Shear mapping

We can also apply a shear mapping to the figures



```
image_gen = ImageDataGenerator(rescale=1./255, shear_range=10)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(6)]
plotImages(augmented_images, 6)
```

    Found 2000 images belonging to 2 classes.



![png](CNN_classification_files/CNN_classification_38_1.png)


## Final augmented dataset

Now we can apply all these transformations with one generator,so then this generator gives us batches from the "augmented" dataset.


```
image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')
```

    Found 2000 images belonging to 2 classes.


Let's see a few examples


```
augmented_images = [train_data_gen[0][0][0] for i in range(6)]
plotImages(augmented_images, 6)
```


![png](CNN_classification_files/CNN_classification_43_0.png)


The generator for the validation dataset does not apply any transformations besides simply rescaling them and splitting into batches.


```
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')
```

    Found 1000 images belonging to 2 classes.


## Improving the model

Now we have augmented the training dataset to help reduce overfitting. We can also improve the model itself to fight overfitting by applying dropout with probability of 50 %. Which means that 50% (randomly chosen) of the values coming into the dropout layers will be set to zero.






```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])
```


```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

Now let's just train the model


```
epochs=100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 20 steps, validate for 10 steps
    Epoch 1/100
    20/20 [==============================] - 17s 867ms/step - loss: 0.7447 - accuracy: 0.5035 - val_loss: 0.6953 - val_accuracy: 0.5000
    Epoch 2/100
    20/20 [==============================] - 17s 841ms/step - loss: 0.6935 - accuracy: 0.5000 - val_loss: 0.6928 - val_accuracy: 0.5950
    Epoch 3/100
    20/20 [==============================] - 17s 842ms/step - loss: 0.6932 - accuracy: 0.5105 - val_loss: 0.6888 - val_accuracy: 0.5210
    Epoch 4/100
    20/20 [==============================] - 17s 835ms/step - loss: 0.6850 - accuracy: 0.5335 - val_loss: 0.6458 - val_accuracy: 0.6180
    Epoch 5/100
    20/20 [==============================] - 17s 837ms/step - loss: 0.6584 - accuracy: 0.5930 - val_loss: 0.6348 - val_accuracy: 0.6340
    Epoch 6/100
    20/20 [==============================] - 17s 840ms/step - loss: 0.6448 - accuracy: 0.6175 - val_loss: 0.6655 - val_accuracy: 0.5700
    Epoch 7/100
    20/20 [==============================] - 17s 834ms/step - loss: 0.6628 - accuracy: 0.5665 - val_loss: 0.6396 - val_accuracy: 0.6090
    Epoch 8/100
    20/20 [==============================] - 17s 841ms/step - loss: 0.6461 - accuracy: 0.5990 - val_loss: 0.6070 - val_accuracy: 0.6500
    Epoch 9/100
    20/20 [==============================] - 17s 838ms/step - loss: 0.6484 - accuracy: 0.6065 - val_loss: 0.6376 - val_accuracy: 0.6110
    Epoch 10/100
    20/20 [==============================] - 17s 837ms/step - loss: 0.6355 - accuracy: 0.6130 - val_loss: 0.6068 - val_accuracy: 0.6750
    Epoch 11/100
    20/20 [==============================] - 17s 837ms/step - loss: 0.6322 - accuracy: 0.6350 - val_loss: 0.6346 - val_accuracy: 0.6450
    Epoch 12/100
    20/20 [==============================] - 17s 837ms/step - loss: 0.6225 - accuracy: 0.6425 - val_loss: 0.5869 - val_accuracy: 0.6930
    Epoch 13/100
    20/20 [==============================] - 17s 854ms/step - loss: 0.6023 - accuracy: 0.6765 - val_loss: 0.5970 - val_accuracy: 0.7000
    Epoch 14/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.6291 - accuracy: 0.6345 - val_loss: 0.5948 - val_accuracy: 0.6800
    Epoch 15/100
    20/20 [==============================] - 17s 852ms/step - loss: 0.5955 - accuracy: 0.6605 - val_loss: 0.5875 - val_accuracy: 0.6830
    Epoch 16/100
    20/20 [==============================] - 17s 850ms/step - loss: 0.5946 - accuracy: 0.6850 - val_loss: 0.5849 - val_accuracy: 0.6700
    Epoch 17/100
    20/20 [==============================] - 17s 859ms/step - loss: 0.6028 - accuracy: 0.6705 - val_loss: 0.5736 - val_accuracy: 0.7110
    Epoch 18/100
    20/20 [==============================] - 17s 857ms/step - loss: 0.5684 - accuracy: 0.7005 - val_loss: 0.5827 - val_accuracy: 0.7000
    Epoch 19/100
    20/20 [==============================] - 17s 861ms/step - loss: 0.5856 - accuracy: 0.6875 - val_loss: 0.6226 - val_accuracy: 0.6610
    Epoch 20/100
    20/20 [==============================] - 17s 863ms/step - loss: 0.5904 - accuracy: 0.6830 - val_loss: 0.6043 - val_accuracy: 0.6570
    Epoch 21/100
    20/20 [==============================] - 17s 853ms/step - loss: 0.5732 - accuracy: 0.7090 - val_loss: 0.5673 - val_accuracy: 0.7080
    Epoch 22/100
    20/20 [==============================] - 17s 857ms/step - loss: 0.5584 - accuracy: 0.7230 - val_loss: 0.5539 - val_accuracy: 0.7090
    Epoch 23/100
    20/20 [==============================] - 17s 853ms/step - loss: 0.5607 - accuracy: 0.6990 - val_loss: 0.5516 - val_accuracy: 0.7060
    Epoch 24/100
    20/20 [==============================] - 17s 859ms/step - loss: 0.5640 - accuracy: 0.7110 - val_loss: 0.5476 - val_accuracy: 0.7180
    Epoch 25/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.5477 - accuracy: 0.7280 - val_loss: 0.5571 - val_accuracy: 0.7160
    Epoch 26/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.5665 - accuracy: 0.7135 - val_loss: 0.5446 - val_accuracy: 0.7090
    Epoch 27/100
    20/20 [==============================] - 17s 858ms/step - loss: 0.5432 - accuracy: 0.7285 - val_loss: 0.5449 - val_accuracy: 0.7280
    Epoch 28/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.5327 - accuracy: 0.7365 - val_loss: 0.5236 - val_accuracy: 0.7250
    Epoch 29/100
    20/20 [==============================] - 17s 840ms/step - loss: 0.5208 - accuracy: 0.7370 - val_loss: 0.5021 - val_accuracy: 0.7630
    Epoch 30/100
    20/20 [==============================] - 17s 841ms/step - loss: 0.5313 - accuracy: 0.7290 - val_loss: 0.5393 - val_accuracy: 0.7250
    Epoch 31/100
    20/20 [==============================] - 17s 841ms/step - loss: 0.5135 - accuracy: 0.7435 - val_loss: 0.5394 - val_accuracy: 0.7490
    Epoch 32/100
    20/20 [==============================] - 17s 844ms/step - loss: 0.5237 - accuracy: 0.7410 - val_loss: 0.5208 - val_accuracy: 0.7390
    Epoch 33/100
    20/20 [==============================] - 17s 837ms/step - loss: 0.5282 - accuracy: 0.7360 - val_loss: 0.5137 - val_accuracy: 0.7420
    Epoch 34/100
    20/20 [==============================] - 17s 844ms/step - loss: 0.5094 - accuracy: 0.7525 - val_loss: 0.5126 - val_accuracy: 0.7470
    Epoch 35/100
    20/20 [==============================] - 17s 842ms/step - loss: 0.5091 - accuracy: 0.7475 - val_loss: 0.5114 - val_accuracy: 0.7420
    Epoch 36/100
    20/20 [==============================] - 17s 842ms/step - loss: 0.5190 - accuracy: 0.7325 - val_loss: 0.5456 - val_accuracy: 0.7210
    Epoch 37/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.5250 - accuracy: 0.7410 - val_loss: 0.4791 - val_accuracy: 0.7760
    Epoch 38/100
    20/20 [==============================] - 17s 854ms/step - loss: 0.5007 - accuracy: 0.7690 - val_loss: 0.5104 - val_accuracy: 0.7360
    Epoch 39/100
    20/20 [==============================] - 17s 840ms/step - loss: 0.4962 - accuracy: 0.7580 - val_loss: 0.4631 - val_accuracy: 0.7920
    Epoch 40/100
    20/20 [==============================] - 17s 843ms/step - loss: 0.4805 - accuracy: 0.7690 - val_loss: 0.4769 - val_accuracy: 0.7650
    Epoch 41/100
    20/20 [==============================] - 17s 855ms/step - loss: 0.4683 - accuracy: 0.7755 - val_loss: 0.5139 - val_accuracy: 0.7480
    Epoch 42/100
    20/20 [==============================] - 17s 851ms/step - loss: 0.5033 - accuracy: 0.7495 - val_loss: 0.4837 - val_accuracy: 0.7600
    Epoch 43/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.4948 - accuracy: 0.7565 - val_loss: 0.5172 - val_accuracy: 0.7540
    Epoch 44/100
    20/20 [==============================] - 17s 845ms/step - loss: 0.4707 - accuracy: 0.7765 - val_loss: 0.4559 - val_accuracy: 0.7770
    Epoch 45/100
    20/20 [==============================] - 17s 861ms/step - loss: 0.4539 - accuracy: 0.7780 - val_loss: 0.5193 - val_accuracy: 0.7360
    Epoch 46/100
    20/20 [==============================] - 17s 861ms/step - loss: 0.4870 - accuracy: 0.7735 - val_loss: 0.4606 - val_accuracy: 0.7860
    Epoch 47/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.4536 - accuracy: 0.7790 - val_loss: 0.4483 - val_accuracy: 0.7960
    Epoch 48/100
    20/20 [==============================] - 17s 852ms/step - loss: 0.4515 - accuracy: 0.7930 - val_loss: 0.4416 - val_accuracy: 0.7960
    Epoch 49/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.4305 - accuracy: 0.8070 - val_loss: 0.4450 - val_accuracy: 0.7920
    Epoch 50/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.4344 - accuracy: 0.7935 - val_loss: 0.4553 - val_accuracy: 0.7900
    Epoch 51/100
    20/20 [==============================] - 17s 850ms/step - loss: 0.4495 - accuracy: 0.7915 - val_loss: 0.5085 - val_accuracy: 0.7530
    Epoch 52/100
    20/20 [==============================] - 17s 850ms/step - loss: 0.4480 - accuracy: 0.7865 - val_loss: 0.4229 - val_accuracy: 0.8010
    Epoch 53/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.4238 - accuracy: 0.8025 - val_loss: 0.4246 - val_accuracy: 0.7950
    Epoch 54/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.4210 - accuracy: 0.8155 - val_loss: 0.4294 - val_accuracy: 0.7920
    Epoch 55/100
    20/20 [==============================] - 17s 852ms/step - loss: 0.3942 - accuracy: 0.8285 - val_loss: 0.4600 - val_accuracy: 0.7990
    Epoch 56/100
    20/20 [==============================] - 17s 863ms/step - loss: 0.4146 - accuracy: 0.8075 - val_loss: 0.4263 - val_accuracy: 0.7940
    Epoch 57/100
    20/20 [==============================] - 17s 841ms/step - loss: 0.3987 - accuracy: 0.8220 - val_loss: 0.4133 - val_accuracy: 0.8050
    Epoch 58/100
    20/20 [==============================] - 17s 844ms/step - loss: 0.4500 - accuracy: 0.7765 - val_loss: 0.4333 - val_accuracy: 0.8050
    Epoch 59/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.4047 - accuracy: 0.8190 - val_loss: 0.4104 - val_accuracy: 0.8000
    Epoch 60/100
    20/20 [==============================] - 17s 848ms/step - loss: 0.3919 - accuracy: 0.8245 - val_loss: 0.4486 - val_accuracy: 0.7990
    Epoch 61/100
    20/20 [==============================] - 17s 839ms/step - loss: 0.4138 - accuracy: 0.8040 - val_loss: 0.4265 - val_accuracy: 0.8050
    Epoch 62/100
    20/20 [==============================] - 17s 842ms/step - loss: 0.3836 - accuracy: 0.8285 - val_loss: 0.4942 - val_accuracy: 0.7650
    Epoch 63/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.3894 - accuracy: 0.8330 - val_loss: 0.4436 - val_accuracy: 0.7850
    Epoch 64/100
    20/20 [==============================] - 17s 843ms/step - loss: 0.3610 - accuracy: 0.8300 - val_loss: 0.4054 - val_accuracy: 0.8120
    Epoch 65/100
    20/20 [==============================] - 17s 851ms/step - loss: 0.3754 - accuracy: 0.8330 - val_loss: 0.4086 - val_accuracy: 0.8120
    Epoch 66/100
    20/20 [==============================] - 17s 848ms/step - loss: 0.3639 - accuracy: 0.8290 - val_loss: 0.4439 - val_accuracy: 0.7930
    Epoch 67/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.3789 - accuracy: 0.8275 - val_loss: 0.4155 - val_accuracy: 0.8030
    Epoch 68/100
    20/20 [==============================] - 17s 843ms/step - loss: 0.3809 - accuracy: 0.8310 - val_loss: 0.3749 - val_accuracy: 0.8230
    Epoch 69/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.3673 - accuracy: 0.8350 - val_loss: 0.4231 - val_accuracy: 0.8160
    Epoch 70/100
    20/20 [==============================] - 17s 852ms/step - loss: 0.3650 - accuracy: 0.8350 - val_loss: 0.4328 - val_accuracy: 0.8050
    Epoch 71/100
    20/20 [==============================] - 17s 848ms/step - loss: 0.3666 - accuracy: 0.8370 - val_loss: 0.4025 - val_accuracy: 0.8090
    Epoch 72/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.3560 - accuracy: 0.8490 - val_loss: 0.3940 - val_accuracy: 0.8120
    Epoch 73/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.3645 - accuracy: 0.8370 - val_loss: 0.3945 - val_accuracy: 0.8210
    Epoch 74/100
    20/20 [==============================] - 17s 857ms/step - loss: 0.3755 - accuracy: 0.8270 - val_loss: 0.3962 - val_accuracy: 0.8130
    Epoch 75/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.3453 - accuracy: 0.8455 - val_loss: 0.4051 - val_accuracy: 0.8220
    Epoch 76/100
    20/20 [==============================] - 17s 841ms/step - loss: 0.3296 - accuracy: 0.8520 - val_loss: 0.4213 - val_accuracy: 0.8110
    Epoch 77/100
    20/20 [==============================] - 17s 844ms/step - loss: 0.3245 - accuracy: 0.8560 - val_loss: 0.4194 - val_accuracy: 0.8160
    Epoch 78/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.3269 - accuracy: 0.8560 - val_loss: 0.4086 - val_accuracy: 0.8230
    Epoch 79/100
    20/20 [==============================] - 17s 844ms/step - loss: 0.3459 - accuracy: 0.8485 - val_loss: 0.4893 - val_accuracy: 0.7800
    Epoch 80/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.3871 - accuracy: 0.8185 - val_loss: 0.4117 - val_accuracy: 0.8120
    Epoch 81/100
    20/20 [==============================] - 17s 845ms/step - loss: 0.3397 - accuracy: 0.8475 - val_loss: 0.4040 - val_accuracy: 0.8170
    Epoch 82/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.3349 - accuracy: 0.8525 - val_loss: 0.4030 - val_accuracy: 0.8110
    Epoch 83/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.3388 - accuracy: 0.8415 - val_loss: 0.3807 - val_accuracy: 0.8220
    Epoch 84/100
    20/20 [==============================] - 17s 856ms/step - loss: 0.3146 - accuracy: 0.8650 - val_loss: 0.4720 - val_accuracy: 0.8090
    Epoch 85/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.3265 - accuracy: 0.8660 - val_loss: 0.3810 - val_accuracy: 0.8230
    Epoch 86/100
    20/20 [==============================] - 17s 847ms/step - loss: 0.2937 - accuracy: 0.8715 - val_loss: 0.4077 - val_accuracy: 0.8220
    Epoch 87/100
    20/20 [==============================] - 17s 846ms/step - loss: 0.2871 - accuracy: 0.8785 - val_loss: 0.4043 - val_accuracy: 0.8170
    Epoch 88/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.3410 - accuracy: 0.8590 - val_loss: 0.3787 - val_accuracy: 0.8240
    Epoch 89/100
    20/20 [==============================] - 17s 854ms/step - loss: 0.3122 - accuracy: 0.8620 - val_loss: 0.4360 - val_accuracy: 0.8120
    Epoch 90/100
    20/20 [==============================] - 17s 849ms/step - loss: 0.3034 - accuracy: 0.8685 - val_loss: 0.3895 - val_accuracy: 0.8210
    Epoch 91/100
    20/20 [==============================] - 17s 852ms/step - loss: 0.2910 - accuracy: 0.8805 - val_loss: 0.4592 - val_accuracy: 0.7950
    Epoch 92/100
    20/20 [==============================] - 17s 850ms/step - loss: 0.2830 - accuracy: 0.8800 - val_loss: 0.4029 - val_accuracy: 0.8110
    Epoch 93/100
    20/20 [==============================] - 17s 855ms/step - loss: 0.3129 - accuracy: 0.8625 - val_loss: 0.3854 - val_accuracy: 0.8330
    Epoch 94/100
    20/20 [==============================] - 17s 851ms/step - loss: 0.3182 - accuracy: 0.8585 - val_loss: 0.3972 - val_accuracy: 0.8190
    Epoch 95/100
    20/20 [==============================] - 17s 843ms/step - loss: 0.2914 - accuracy: 0.8720 - val_loss: 0.3799 - val_accuracy: 0.8290
    Epoch 96/100
    20/20 [==============================] - 16s 824ms/step - loss: 0.2592 - accuracy: 0.8930 - val_loss: 0.3881 - val_accuracy: 0.8310
    Epoch 97/100
    20/20 [==============================] - 17s 826ms/step - loss: 0.2753 - accuracy: 0.8815 - val_loss: 0.3670 - val_accuracy: 0.8320
    Epoch 98/100
    20/20 [==============================] - 17s 828ms/step - loss: 0.2667 - accuracy: 0.8900 - val_loss: 0.3695 - val_accuracy: 0.8480
    Epoch 99/100
    20/20 [==============================] - 17s 843ms/step - loss: 0.2939 - accuracy: 0.8760 - val_loss: 0.4049 - val_accuracy: 0.8240
    Epoch 100/100
    20/20 [==============================] - 17s 835ms/step - loss: 0.2830 - accuracy: 0.8755 - val_loss: 0.4179 - val_accuracy: 0.8200


## Results of improved model

Now that we have used the augmented training dataset to train the improved (dropout) model Let's look at its performance


```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```


![png](CNN_classification_files/CNN_classification_52_0.png)


We see that the accuracy goes up "almost" monotonically for both the training and validation set and the performance on the validation set is considerably better than before.
