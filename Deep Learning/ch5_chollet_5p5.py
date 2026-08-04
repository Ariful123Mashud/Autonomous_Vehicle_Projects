import os
from pyexpat import model
import shutil

original_dataset_dir = "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/DL_AV/Chollet_DL/kaggle_cat_dog_dataset/training_set/training_set"
base_dir = "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/DL_AV/Chollet_DL/cats_and_dogs_small"
os.makedirs(base_dir, exist_ok=True)

##--------Make directories for train, validation and test datasets
train_dir = os.path.join(base_dir, "train")
os.makedirs(train_dir, exist_ok=True)

validation_dir = os.path.join(base_dir, "validation")
os.makedirs(validation_dir, exist_ok=True)

test_dir = os.path.join(base_dir, "test")
os.makedirs(test_dir, exist_ok=True)


## --------Make directories for train, validation and test datasets for cats and dogs
train_cats_dir = os.path.join(train_dir, "cats")
os.makedirs(train_cats_dir, exist_ok=True)

train_dogs_dir = os.path.join(train_dir, "dogs")
os.makedirs(train_dogs_dir, exist_ok=True)

validation_cats_dir = os.path.join(validation_dir, "cats")
os.makedirs(validation_cats_dir, exist_ok=True)

validation_dogs_dir = os.path.join(validation_dir, "dogs")
os.makedirs(validation_dogs_dir, exist_ok=True)

test_cats_dir = os.path.join(test_dir, "cats")
os.makedirs(test_cats_dir, exist_ok=True)

test_dogs_dir = os.path.join(test_dir, "dogs")
os.makedirs(test_dogs_dir, exist_ok=True)


fnames = [f"cat.{i}.jpg" for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, "cats", fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f"cat.{i}.jpg" for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, "cats", fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f"cat.{i}.jpg" for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, "cats", fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f"dog.{i}.jpg" for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, "dogs", fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f"dog.{i}.jpg" for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, "dogs", fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f"dog.{i}.jpg" for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, "dogs", fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print("Total training cat images:", len(os.listdir(train_cats_dir)))
print("Total training dog images:", len(os.listdir(train_dogs_dir)))
print("Total validation cat images:", len(os.listdir(validation_cats_dir)))
print("Total validation dog images:", len(os.listdir(validation_dogs_dir)))
print("Total test cat images:", len(os.listdir(test_cats_dir)))
print("Total test dog images:", len(os.listdir(test_dogs_dir)))

# ##--------------------------------------
from tensorflow import keras
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-8),
    metrics=["acc"],
)


# ##------------------------------------------

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

tarin_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
)


for data_batch, labels_batch in tarin_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break


history = model.fit_generator(
    tarin_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
)

model.save('cats_dogs_small_1.h5')

import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]  

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.show()  


plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.figure()
plt.show()


