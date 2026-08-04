from keras.preprocessing.image import ImageDataGenerator
from keras import models, optimizers
import os
import matplotlib.pyplot as plt

import os
import keras
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

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)


###------Dispay some randomly augmented training images
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[10]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()

###-----fighting  overfitting with data augmentation and dropout
model = models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=["acc"],
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)
model.save("cats_and_dogs.h5")

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
print(history.history.keys())

epochs = range(1, len(acc) + 1)
print(len(acc))
print(len(val_acc))
print(len(loss))
print(len(val_loss))
print(len(epochs))



plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()    
