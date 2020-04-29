import numpy as np
#Note to run this script you need to run librosa
import librosa
import os
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import wave
import pylab
from matplotlib import cm
import pandas as pd
import librosa.display
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


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

def prepare_data_model(folder):
    print("Preparing Data")
    mfccs, labels = get_data(folder)

    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1
    classes = 10

    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels

    input_shape = (dim_1, dim_2, channels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    print("Creating Model")
    model = get_cnn_model(input_shape, classes)

    return X_train, X_val, X_test, y_train, y_val, y_test, model


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

def get_cnn_model(input_shape, num_classes, dropout = True, batch_n = True, MaxPooling = True, optimzer = keras.optimizers.Adam() ):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    if batch_n:
        model.add(BatchNormalization())

    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    if batch_n:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    if batch_n:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimzer, metrics=['accuracy'])

    return model

def get_feedforward_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model

def evaluate_model(model, test_X, test_y):
    model = keras.models.load_model('best_model.h5')
    print(model.evaluate(test_X, test_y))




if __name__ == '__main__':
     print("Generating Graphs")
     generate_wave_graph( '0_jackson_0.wav')
     generate_spectogram('0_jackson_0.wav')
     generate_mfcc_graph('0_jackson_0.wav')

     print("Done")
     X_train, X_val, X_test, y_train, y_val, y_test, model = prepare_data_model('./recordings/')

     print(model.summary())

     keras_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1,
                                                  write_graph=True, write_images=True)
     #EarlyStopping(monitor='val_loss', patience=2),
     callbacks = [
                  ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True), TensorBoard(log_dir='./Graph', histogram_freq=1,
                                                  write_graph=False, write_images=False)]

     model.fit(X_train, y_train, batch_size=250, epochs= 50, verbose= 2, validation_data = [X_val, y_val],
                   callbacks=callbacks)

     evaluate_model('best_model.h5', X_test, y_test)



