import numpy as np
#Note to run this script you need to run librosa
import librosa
import os
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import wave
import pylab
import librosa.display
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from sklearn.metrics import classification_report


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def wav_to_mfcc(file_path, max_pad_len=20):
    wave, sr = librosa.load(file_path, mono=True, sr = None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr= 8000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def get_data(folder):
    print("Fetching wav data from" + folder)
    labels = []
    mfccs = []

    for f in os.listdir('./' + folder):
        if f.endswith('.wav'):
            # MFCC
            mfccs.append(wav_to_mfcc(folder + f))

            # List of labels
            label = f.split('_')[0]
            labels.append(label)

    #print(np.array([np.hstack(i) for i in mfccs]))
    return np.asarray(mfccs), to_categorical(labels)

def prepare_data(folder):
    print("Preparing Data")
    mfccs, labels = get_data(folder)

    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1
    classes = 10

    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels

    input_output = [(dim_1, dim_2, channels), classes]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)



    return X_train, X_val, X_test, y_train, y_val, y_test, input_output

def prepare_model(input_output, modeltype = 'CNN', dropout = True, batch_n = True, maxpooling = True, optimizer = keras.optimizers.Adam()):
    print("Creating Model")
    if modeltype == 'CNN':
        model = get_cnn_model(input_output[0], input_output[1], dropout, batch_n, maxpooling, optimizer)
        print(model.summary())
        plot_model(model, to_file='images/CNN_model.png')
    elif modeltype == 'FF':
        model = get_feedforward_model(input_output[0], input_output[1], dropout, batch_n, optimizer)
        print(model.summary())
        plot_model(model, to_file='images/FF_model.png')
    else:
        raise ValueError('Not An Acceptable Model Type Sorry!')

    return model

def generate_wave_graph(filename):
    file = './recordings/' + filename

    with wave.open(file, 'r') as wav_file:
        # Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.frombuffer(signal, 'Int16')

        # Split the data into channels
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index % len(channels)].append(datum)


        # Get time from indices
        fs = wav_file.getframerate()
        Time = np.linspace(0, len(signal) / len(channels) / fs, num= int(len(signal) / len(channels)))

        # Plot
        plt.figure(1)
        plt.title('Signal Wave')
        for channel in channels:
            plt.plot(Time, channel)
        plt.savefig('images/waveform' + filename + '.png')
        plt.show()


def generate_spectogram(filename):
    sound_info, frame_rate = get_wav_info('./recordings/' + filename)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('Spectrogram of %r' % filename)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('images/spectrogram' + filename + '.png')
    pylab.show()

def get_wav_info(filename):
    with wave.open(filename, 'r') as wav_file:
        frames = wav_file.readframes(-1)
        sound_info = pylab.fromstring(frames, 'int16')
        frame_rate = wav_file.getframerate()
    return sound_info, frame_rate

def generate_mfcc_graph(filename):
    (xf, sr) = librosa.load('./recordings/' + filename)
    mfccs = librosa.feature.mfcc(y=xf, sr=sr, n_mfcc=4)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.title('mfcc')
    plt.savefig('images/mfcc' + filename + '.png')
    plt.show()

def generate_graphs(filename):
    generate_wave_graph(filename)
    generate_mfcc_graph(filename)
    generate_spectogram(filename)

def get_cnn_model(input_shape, num_classes, dropout = True, batch_n = True, maxpooling = True, optimizer = keras.optimizers.Adam() ):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    if batch_n:
        model.add(BatchNormalization())

    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    if batch_n:
        model.add(BatchNormalization())
    if maxpooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if batch_n:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    if batch_n:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

def get_feedforward_model(input_shape, num_classes, dropout = True, batch_n = True,  optimizer = keras.optimizers.Adam() ):
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=input_shape))

    model.add(Flatten())
    if batch_n:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    if batch_n:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.4))

    model.add(Dense(64, activation='relu'))
    if batch_n:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer = optimizer,
                  metrics=['accuracy'])

    return model

def evaluate_model(model, test_X, test_y):
    model = keras.models.load_model(model)
    predictions = model.predict_classes(test_X)

    eval = model.evaluate(test_X, test_y)
    print('Test Score: '  + str(eval[1]))
    print(classification_report(test_y, to_categorical(predictions)))

def plot_losses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('images/loss_curve.png')
    plt.show()



if __name__ == '__main__':
     print("Generating Graphs")
     generate_wave_graph('0_jackson_0.wav')
     generate_spectogram('0_jackson_0.wav')
     generate_mfcc_graph('0_jackson_0.wav')

     print("Done")
     X_train, X_val, X_test, y_train, y_val, y_test, input_output = prepare_data('./recordings/')

     model = prepare_model(input_output)

     keras_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1,
                                                  write_graph=True, write_images=True)

     callbacks = [ModelCheckpoint(filepath='models/model.h5', monitor='val_loss', save_best_only=True), TensorBoard(log_dir='./Graph', histogram_freq=1,
                                                  write_graph=False, write_images=False)]

     history = model.fit(X_train, y_train, batch_size=32, epochs=50, verbose= 2, validation_data = [X_val, y_val],
                   callbacks=callbacks)

     # Plot training & validation accuracy values
     plot_losses(history)

     evaluate_model('models/model.h5', X_test, y_test)



