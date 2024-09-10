from tensorflow.keras.models import load_model
# import tensorflowjs as tfjs

model = load_model('training3.h5')
model.save('training3.keras')
# tfjs.converters.save_keras_model(model, 'tfjs_model')
