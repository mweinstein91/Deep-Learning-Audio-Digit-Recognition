from utils import prepare_data, prepare_model, plot_losses, evaluate_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import random

#Creates our training/val/testing splits
random.seed(9001)
X_train, X_val, X_test, y_train, y_val, y_test, input_output = prepare_data('./recordings/')

#Prepares a CNN with our desired architecture
CNN_best_model = prepare_model(input_output, modeltype='CNN', dropout=False, maxpooling=True, batch_n=False)

#Creates callbacks to save best model in training
callbacks = [ModelCheckpoint(filepath='models/cnn_best_model.h5', monitor='val_loss', save_best_only=True), TensorBoard(log_dir='./Graph', histogram_freq=1,
                                                  write_graph=False, write_images=False)]
#Fits model
history = CNN_best_model.fit(X_train, y_train, batch_size=32, epochs=50, verbose= 2, validation_data = [X_val, y_val],
                   callbacks=callbacks)

#Plots loss curve
plot_losses(history)

#Evaluate model on testing set
evaluate_model('models/cnn_best_model.h5', X_test, y_test)