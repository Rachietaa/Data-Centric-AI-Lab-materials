import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

# suppress warning about tensorflow not finding a GPU in Coursera
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

def split_copy(origin, train_dest, dev_dest, train_ratio=0.7):
    # Ensure directories exist
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(dev_dest, exist_ok=True)

    # Sorted for deterministic behavior
    images = sorted([
        f for f in os.listdir(origin)
        if os.path.isfile(os.path.join(origin, f))
    ])

    split_idx = int(len(images) * train_ratio)

    train_images = images[:split_idx]
    dev_images = images[split_idx:]

    # Copy train portion
    for img in train_images:
        shutil.copy2(
            os.path.join(origin, img),
            os.path.join(train_dest, img)
        )

    # Copy dev portion
    for img in dev_images:
        shutil.copy2(
            os.path.join(origin, img),
            os.path.join(dev_dest, img)
        )


def copy_with_limit(origin, destination, percentage_split):
  num_images = int(len(os.listdir(origin))*percentage_split)
  for image_name, image_number in zip(sorted(os.listdir(origin)), range(num_images)):
    shutil.copy(os.path.join(origin, image_name), destination)


def create_model():
  # A simple CNN architecture based on the one found here: https://www.tensorflow.org/tutorials/images/classification
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])


  # Compile the model
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['sparse_categorical_accuracy']
  )

  return model


def get_training_metrics(history):

  # This is needed depending on if you used the pretrained model or you trained it yourself
  if not isinstance(history, pd.core.frame.DataFrame):
    history = history.history

  acc = history['sparse_categorical_accuracy']
  val_acc = history['val_sparse_categorical_accuracy']

  loss = history['loss']
  val_loss = history['val_loss']

  return acc, val_acc, loss, val_loss


def plot_train_eval(history):
  acc, val_acc, loss, val_loss = get_training_metrics(history)

  acc_plot = pd.DataFrame({"training accuracy":acc, "validation accuracy":val_acc})
  acc_plot = sns.lineplot(data=acc_plot)
  acc_plot.set_title('training vs validation accuracy')
  acc_plot.set_xlabel('epoch')
  acc_plot.set_ylabel('sparse_categorical_accuracy')
  plt.show()

  print("")

  loss_plot = pd.DataFrame({"training loss":loss, "validation loss":val_loss})
  loss_plot = sns.lineplot(data=loss_plot)
  loss_plot.set_title('training vs validation loss')
  loss_plot.set_xlabel('epoch')
  loss_plot.set_ylabel('loss')
  plt.show()


def demo_augmentation(sample_image, model, num_aug):
  '''Takes a single image array, then uses a model to generate num_aug transformations'''

  # Instantiate preview list
  image_preview = []

  # Convert input image to a PIL image instance
  sample_image_pil = tf.keras.utils.array_to_img(sample_image)

  # Append the result to the list
  image_preview.append(sample_image_pil)

  # Apply the image augmentation and append the results to the list
  for i in range(num_aug):
    sample_image_aug = model(tf.expand_dims(sample_image, axis=0))
    sample_image_aug_pil = tf.keras.utils.array_to_img(tf.squeeze(sample_image_aug))
    image_preview.append(sample_image_aug_pil)

  # Instantiate a subplot
  fig, axes = plt.subplots(1, num_aug + 1, figsize=(12, 12))

  # Preview the images.
  for index, ax in enumerate(axes):
    ax.imshow(image_preview[index])
    ax.set_axis_off()

    if index == 0:
      ax.set_title('original')
    else:
      ax.set_title(f'augment {index}')
