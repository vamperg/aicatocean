import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import os.path
import random
import sys
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import warnings
 
# load the image via load_img
# function
from os import walk
# Получить имена файлов в папке
def get_file_names(path):
  f = []
  for (dirpath, dirnames, filenames) in walk("ocean/" + path):

      f.extend(filenames)
      break
  return f





def img_to_arr_learn(file, pos):
  img = load_img('ocean/'+file, color_mode="grayscale")
  image = img.resize([24, 24])
  sqr_arr = []
  # convert the given image into  numpy array
  img_numpy_array = img_to_array(image)
  arr = []
  for x in img_numpy_array:
    for y in x:
      arr.append((y[0])/255)
  if pos == 0:    
    return (arr, [1,0,0,0,0,0,0])
  elif pos == 1:
    return (arr, [0,1,0,0,0,0,0])
  elif pos == 2:
    return (arr, [0,0,1,0,0,0,0])
  elif pos == 3:
    return (arr, [0,0,0,1,0,0,0])
  elif pos == 4:
    return (arr, [0,0,0,0,1,0,0])
  elif pos == 5:
    return (arr, [0,0,0,0,0,1,0])
  elif pos == 6:
    return (arr, [0,0,0,0,0,0,1])


  


all_arr = []

def load_data():
  corals = get_file_names("Corals")
  crabs = get_file_names("Crabs")
  dolphin = get_file_names("Dolphin")
  eel = get_file_names("Eel")
  jellyfish = get_file_names("Jelly Fish")
  lobster = get_file_names("Lobster")
  octopus = get_file_names("Octopus")
  penguin = get_file_names("Penguin")
  puffers = get_file_names("Puffers")
  sharks = get_file_names("Sharks")
  squid = get_file_names("Squid")
  seal = get_file_names("Seal")
  rays = get_file_names("Sea Rays")
  for i in range(len(crabs)):
    all_arr.append(img_to_arr_learn("Crabs/"+crabs[i],0))
  for i in range(len(dolphin)):
    all_arr.append((img_to_arr_learn("Dolphin/"+dolphin[i],1)))
  for i in range(len(jellyfish)):
    all_arr.append((img_to_arr_learn("Jelly Fish/"+jellyfish[i],2)))
  for i in range(len(octopus)):
    all_arr.append((img_to_arr_learn("Octopus/"+octopus[i],3)))
  for i in range(len(penguin)):
    all_arr.append(img_to_arr_learn("Penguin/"+penguin[i],4))
  for i in range(len(sharks)):
    all_arr.append(img_to_arr_learn("Sharks/"+sharks[i],5))
  for i in range(len(seal)):
    all_arr.append(img_to_arr_learn("Seal/"+seal[i],6))


if os.path.isfile("train_arr.npz"):
  print('был обнаружен массив входных данных, хотите загрузить? (y|n')
  do = input()
  if do == 'y' or do == 'Y':
    load = np.load('train_arr.npz',allow_pickle=True)
    all_arr = load['arr_0']
  else:
    load_data()
else:
  load_data()
    
np.savez('train_arr',all_arr)

class PartyNN(object):
    def save_weights(self):
        np.savez('save_weights',self.weights_0_1,self.weights_1_2)

    def load_weights(self):
        load = np.load('save_weights.npz')
        self.weights_0_1 = load['arr_0']
        self.weights_1_2 = load['arr_1']
        
    def __init__(self):
        path = "save_weights"
        self.is_train = False
        if os.path.isfile(path+".npz"):
            print('Загрузить веса или обучить с нуля?(y|n)')
            do = input()
            if do == 'y' or do == 'Y':
                self.is_train = True
        print('Введите число входящих нейронов: ')
        self.input_nodes = int(input())
        print('Введите число скрытых нейронов: ')
        self.hidden_nodes = int(input())
        print('Введите число выходных нейронов: ')
        self.out_nodes = int(input())
        
        self.learning_rate = 0.03
        self.weights_0_1 = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.weights_1_2 = (np.random.rand(self.out_nodes, self.hidden_nodes) - 0.5)
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        
    def creat_net(input_nodes, hidden_nodes, out_nodes,):
        # сознание массивов. -0.5 вычитаем что бы получить диапазон -0.5 +0.5 для весов
        input_hidden_w = (np.random.rand(hidden_nodes, input_nodes) - 0.5)
        hidden_out_w = (np.random.rand(out_nodes, hidden_nodes) - 0.5)
        return input_hidden_w, hidden_out_w

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid(inputs_2)
        return outputs_2

    def train(self, inputs, expected_predicts):
        inputs = np.array(inputs, ndmin=2).T
        expected_predicts = np.array(expected_predicts, ndmin=2).T

        if (True):
            inputs_1 = np.dot(self.weights_0_1, inputs)
            outputs_1 = self.sigmoid_mapper(inputs_1)

            inputs_2 = np.dot(self.weights_1_2, outputs_1)
            outputs_2 = self.sigmoid_mapper(inputs_2)
            actual_predict = outputs_2

            # Backward pass
            error_layer_2 = actual_predict - expected_predicts
            gradient_layer_2 = actual_predict * (1 - actual_predict)
            weights_delta_layer_2 = error_layer_2 * gradient_layer_2
            self.weights_1_2 -= self.learning_rate * np.dot(weights_delta_layer_2, outputs_1.T)

            error_layer_1 = np.dot(self.weights_1_2.T, weights_delta_layer_2)
            gradient_layer_1 = outputs_1 * (1 - outputs_1)
            weights_delta_layer_1 = error_layer_1 * gradient_layer_1
            self.weights_0_1 -= np.dot(weights_delta_layer_1, inputs.T) * self.learning_rate
#@title MSE
def MSE(y, Y):
    return np.mean((y-Y)**2)
train = all_arr
x = []
y_loss = []
max_x = len(x)

def start_train(network):
    max_x = len(x)
    print('Введите число эпох: ')
    epochs = int(input())
    print('Введите скорость обучения(0.05): ')
    network.learning_rate = float(input())
    for e in range(epochs):
        x.append(e + max_x)
        inputs_ = []
        correct_predictions = []
        count_train = 0
        for input_stat, correct_predict in train:
            network.train(np.array(input_stat), correct_predict)
            inputs_.append(np.array(input_stat))
            correct_predictions.append(np.array(correct_predict))
            count_train += 1
        train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions).T)
        y_loss.append(train_loss)
        sys.stdout.write("\rProgress: {}%,  {}|{} ({}/{}) Training loss:{}".format(str(100 * e/float(epochs))[:4],e,epochs,count_train,len(train), str(train_loss)[:5]))
    network.save_weights()

network = PartyNN()

if network.is_train == False:
    start_train(network)
    
    
else:
    network.load_weights()
    print("Хотите продолжить обучение сети? y|n")
    do = input()
    if do == "y" or do == "Y":
        start_train(network)

box = img_to_arr_learn("cat_1.jpg",0)
train =[box]
box = img_to_arr_learn("dog_1.jpg",1)
train.append(box)

for input_stat, correct_predict in train:
    pred = ["ХЗ","ХЗ"]
    if (network.predict(np.array(input_stat))[0]> 0.7):
      pred[0] = "Это КОТ"
    if (network.predict(np.array(input_stat))[1] > 0.7):
      pred[1] = "Это ПЕС"
    print()
    print("For input: {} the predict: {}, exprected: {}".format(
        str(1),
        str(network.predict(np.array(input_stat))),
        str(correct_predict)))
    
for input_stat, correct_predict in train:
    pred = ["ХЗ","ХЗ"]
    if (network.predict(np.array(input_stat))[0]> 0.7):
      pred[0] = "Это КОТ"
    if (network.predict(np.array(input_stat))[1] > 0.7):
      pred[1] = "Это ПЕС"
    print()
    print("For input: {} the predict: {}, exprected: {}".format(
        str(1),
        str(pred),
        str(correct_predict)))
    
