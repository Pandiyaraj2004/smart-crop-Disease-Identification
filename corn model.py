import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# --------------------------
# Paths & Parameters
# --------------------------
dataset_path = "corn"  # your dataset path
img_size = (224, 224)          # MobileNetV2 input size
batch_size = 32
epochs = 10                     # increase later if needed
num_classes = 4                 # Common Rust, Gray Leaf Spot, Blight, Healthy

# --------------------------
# Data Generators
# --------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',   # categorical labels for softmax
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --------------------------
# Load Pretrained MobileNetV2
# --------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1],3))
base_model.trainable = False  # Freeze base layers

# --------------------------
# Add Custom Layers
# --------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --------------------------
# Compile Model
# --------------------------
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------
# Train Model
# --------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# --------------------------
# Save Model
# --------------------------
model.save("corn_model.h5")
print("✅ Corn model saved as corn_model.h5")
