import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn") # estilo de gráficas

#%%  Etiquetas de las actividades

LABELS = ['backheel', 'laces_shot', 'volley_shot', 'scorpion_shot', 'toe_shot']


# El número de pasos dentro de un segmento de tiempo
TIME_PERIODS = 80

# Los pasos a dar de un segmento al siguiente; si este valor es igual a
# TIME_PERIODS, entonces no hay solapamiento entre los segmentos
STEP_DISTANCE = 40

# al haber solapamiento aprovechamos más los datos

#%% cargamos los datos

column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis', 'gyro-x', 'gyro-y', 'gyro-z', 'age', 'sex', 'foot']


df = pd.read_csv("football_dat.txt", header=None,
                     names=column_names)


print(df.info())

#%% Datos que tenemos

print(df.shape)


#%% convertimos a flotante

def convert_to_float(x):
    try:
        return float(x)
    except:
        return np.nan

for column in ['x-axis', 'y-axis', 'z-axis', 'gyro-x', 'gyro-y', 'gyro-z']:
    df[column] = df[column].apply(convert_to_float)


#%% Eliminamos entradas que contengan Nan --> ausencia de datos

df.dropna(axis=0, how='any', inplace=True)
#%% Mostramos los primeros datos

print(df.head())

#%% Mostramos los últimos

print(df.tail())

#%% Visualizamos la cantidad de datos que tenemos
# de cada actividad 

actividades = df['activity'].value_counts()
plt.bar(range(len(actividades)), actividades.values)
plt.xticks(range(len(actividades)), actividades.index)

#%% visualizamos 

def dibuja_datos_aceleracion(subset, actividad):
    plt.figure(figsize=(5,7))
    plt.subplot(311)
    plt.plot(subset["x-axis"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel X")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(actividad)
    plt.subplot(312)
    plt.plot(subset["y-axis"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Y")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(313)
    plt.plot(subset["z-axis"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Z")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

def dibuja_datos_gyro(subset, actividad):
    plt.figure(figsize=(5,7))
    plt.subplot(311)
    plt.plot(subset["gyro-x"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Gyro X")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(actividad)
    plt.subplot(312)
    plt.plot(subset["gyro-y"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Gyro Y")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(313)
    plt.plot(subset["gyro-z"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Gyro Z")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

for actividad in np.unique(df['activity']):
    subset = df[df['activity'] == actividad][:80]
    dibuja_datos_aceleracion(subset, actividad)
    dibuja_datos_gyro(subset, actividad)

#%% Codificamos la actividad de manera numérica

from sklearn import preprocessing

LABEL = 'ActivityEncoded'
# Transformar las etiquetas de String a Integer mediante LabelEncoder
le = preprocessing.LabelEncoder()

# Añadir una nueva columna al DataFrame existente con los valores codificados
df[LABEL] = le.fit_transform(df['activity'].values.ravel())

print(df.head())

#%% Normalizamos los datos

df["x-axis"] = (df["x-axis"] - min(df["x-axis"].values)) / (max(df["x-axis"].values) - min(df["x-axis"].values))
df["y-axis"] = (df["y-axis"] - min(df["y-axis"].values)) / (max(df["y-axis"].values) - min(df["y-axis"].values))
df["z-axis"] = (df["z-axis"] - min(df["z-axis"].values)) / (max(df["z-axis"].values) - min(df["z-axis"].values))

df["gyro-x"] = (df["gyro-x"] - min(df["gyro-x"].values)) / (max(df["gyro-x"].values) - min(df["gyro-x"].values))
df["gyro-y"] = (df["gyro-y"] - min(df["gyro-y"].values)) / (max(df["gyro-y"].values) - min(df["gyro-y"].values))
df["gyro-z"] = (df["gyro-z"] - min(df["gyro-z"].values)) / (max(df["gyro-z"].values) - min(df["gyro-z"].values))

#%% Representamos para ver que se ha hecho bien

plt.figure(figsize=(5,5))
plt.plot(df["x-axis"].values[:80])
plt.xlabel("Tiempo")
plt.ylabel("Acel X")

#%% Disión datos den entrenamiento y test
from sklearn.model_selection import train_test_split

df['user-id'] = df['user-id'].astype(int)

df_test = df[df['user-id'] < 2]
df_train = df[df['user-id'] >= 2]

#y=df.loc[:, "activity"].to_numpy()
#df_train, df_test = train_test_split(df, test_size=0.2,shuffle=True,stratify=y,random_state=5)

print("Entrenamiento", df_train.shape)
print("Test", df_test.shape)

#%% comprobamos cual ha sido la división

print("Entrenamiento", df_train.shape[0]/df.shape[0])
print("Test", df_test.shape[0]/df.shape[0])

#%% Creamos las secuencias

from scipy import stats


def create_segments_and_labels(df, time_steps, step, label_name):

    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        xg = df['gyro-x'].values[i: i + time_steps]
        yg = df['gyro-y'].values[i: i + time_steps]
        zg = df['gyro-z'].values[i: i + time_steps]
        # Lo etiquetamos como la actividad más frecuente 
        label = stats.mode(df[label_name][i: i + time_steps])[0]
        segments.append([xs, ys, zs, xg, yg, zg])
        labels.append(label)

    # Los pasamos a vector
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

x_test, y_test = create_segments_and_labels(df_test,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

#%% observamos la nueva forma de los datos (80, 6)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

#%% datos de entrada de la red neuronal

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

#%% transformamos los datos a flotantes

x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
#y_test = y_test.astype('float32')

#%% Realizamos el one-hote econding para los datos de salida

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
y_train_hot = cat_encoder.fit_transform(y_train.reshape(len(y_train),1))
y_train = y_train_hot.toarray()

#%% RED NEURONAL

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

model_m = Sequential()
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, 
                                                            num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())


#%% Guardamos el mejor modelo y utilizamos early stopping

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
]

#%% determinamos la función de pérdida, optimizador y métrica de funcionamiento 

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])


#%% Entrenamiento

BATCH_SIZE = 400
EPOCHS = 50

history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

#%% Visualización entrenamiento

from sklearn.metrics import classification_report

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()


#%% Evaluamos el modelo en los datos de test

# actualizar dependiendo del nombre del modelo guardado
#model = keras.models.load_model("best_model.98-1.26.h5")

y_test_hot = cat_encoder.fit_transform(y_test.reshape(len(y_test),1))
y_test = y_test_hot.toarray()

test_loss, test_acc = model_m.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
# Print confusion matrix for training data
y_pred_train = model_m.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
max_y_train = np.argmax(y_train, axis=1)
print(classification_report(max_y_train, max_y_pred_train))

#%%
import seaborn as sns
from sklearn import metrics

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

y_pred_test = model_m.predict(x_test)
# Toma la clase con la mayor probabilidad a partir de las predicciones de la prueba
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))