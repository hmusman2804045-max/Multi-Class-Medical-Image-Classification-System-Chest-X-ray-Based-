import os
import tensorflow as tf
from tensorflow.keras import layers

def create_data_pipelines(data_dir, image_size=(224, 224), batch_size=32, validation_split=0.2):
    print(f"Loading data from: {data_dir}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    print(f"Classes detected: {class_names}\n")

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(factor=0.05, fill_mode='nearest'),
        layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode='nearest'),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode='nearest'),
        layers.RandomBrightness(factor=0.1)
    ], name="medical_augmentation")

    normalization_layer = layers.Rescaling(1./255)

    def preprocess_train(images, labels):
        images = data_augmentation(images, training=True)
        images = normalization_layer(images)
        return images, labels

    def preprocess_val(images, labels):
        images = normalization_layer(images)
        return images, labels

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    dataset_path = "dataset/train"
    
    if os.path.exists(dataset_path):
        train_data, val_data, classes = create_data_pipelines(data_dir=dataset_path)
        print("------------- Pipeline Check -------------")
        for images, labels in train_data.take(1):
            print(f"Batched Image Shape: {images.shape}")
            print(f"Batched Label Shape: {labels.shape}")
            print(f"Max Pixel Value: {tf.reduce_max(images):.2f}")
            print(f"Min Pixel Value: {tf.reduce_min(images):.2f}")
            print("------------------------------------------")
            break
