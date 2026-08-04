import os
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














