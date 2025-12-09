import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- SETTINGS ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATA_DIR_TRAIN = "data/train"
DATA_DIR_VAL   = "data/val"

# ---------- LOAD DATA ----------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_VAL,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ---------- DATA AUGMENTATION ----------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ---------- MODEL ----------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------- TRAIN ----------
EPOCHS = 15   # keep small for now
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# ---------- SAVE MODEL ----------
model.save("ai_waste_classification_model.keras")  # new Keras native format
print("Model saved as ai_waste_classification_model.keras")
