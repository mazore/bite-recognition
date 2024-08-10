import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'data/validate',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# # Load pre-trained MobileNetV2 model + higher level layers
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Class weights to handle imbalance
class_weights = {0: 1.0, 1: 14.1236842105}

# # Train the model
hist = model.fit(train_generator, epochs=10, validation_data=validation_generator, class_weight=class_weights)

# # Unfreeze some layers and fine-tune the model
# for layer in base_model.layers[-50:]:
#     layer.trainable = True

# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_generator, epochs=10, validation_data=validation_generator, class_weight=class_weights)
model = load_model('training2.h5')

# Evaluate the model
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

model.evaluate(test_generator)

# model.save('training2.h5')
