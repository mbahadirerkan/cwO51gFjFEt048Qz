"""
Simple binary classification neural network
github: mbahadirerkan
----- Some comments about the project -------
In order to be sure about accuracy I used random.seed() in range(0,5)
most of the cases exceeds the 0.73 limit in fitting process.
About bonuses: 
I thought about the question, I think that we can try different data sets 
in which the specific feature's value is fixed to 0 or 5, and compare the 
results with original one if there is no difference between them we can readily say that, that feature
-the question in survey- is unnecessary.
"""


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
import csv
import random
from keras.optimizers import Adam



#In case, you want to use callbacks
"""
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        if(logs["accuracy"] > 0.73):
            self.model.stop_training = True
            print("The training is going to be stopped!")
callback = myCallback()
"""

labels = list()
data = list() 
all = list()
with open('ACME-HappinessSurvey2020.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:            

        #Converting the input to the integer.
        if line_count != 0:
          line = list()
          for a in row:
              line.append(int(a))
          all.append(line)

        line_count = line_count + 1
    

#There was a saying, only god and me had known what was happening below, now only god knows :)
#I am just kidding, but I just found these layers by reading the keras document, by trying and in a heuristic way.

#selu layers multiplies layers inputs with a scale 
#relu layers are more powerful with greater shapes
model = Sequential()
#model.add(Dense(6, input_dim = 6, activation = "relu"))


model.add(Dense(6, input_dim = 6, activation="selu"))
model.add(Dense(144, activation= "relu"))
model.add(Dense(36, activation= "selu"))
model.add(Dense(72, activation= "relu"))
model.add(Dense(36, activation= "selu"))


#model.add(Dense(2, activation = "tanh"))
model.add(Dense(1, activation = "sigmoid"))

#meansquarederror because the greater numbers most probably give use 1 as a result.
#adam-sgd-rmsprop are belikened. All of them is using gradients to optimize. - maybe I could not understand the difference :)
model.compile(loss="mean_squared_error", optimizer= "adam" , metrics=["accuracy"])

#adjusting the training and testing data
#you can change the testsize
testsize = 5 
labelnum = len(labels) - testsize

#to choose specific random train and test
#random.seed(3)
#shuffling the data
random.shuffle(all)
line_count = 0

#Seperating labels and data
for row in all:

    count = 0
    line = list()
    for a in row:
        if count == 0:
          labels.append(a)
          count = count + 1
        else:
          line.append(a)
    data.append(line)




#train and test sets
traindata = data[:labelnum]
trainlabel = labels[:labelnum]

testdata = data[labelnum:]
testlabel = labels[labelnum:]

#print(testdata)




#if you want to use callback delete the comments sign below and delete the
#comment signs above of the callback class 
#model.fit(traindata, trainlabel, epochs=100, batch_size=1, callbacks=[callback])

model.fit(traindata, trainlabel, epochs=100, batch_size=1)
print("Helo")


_, accuracy = model.evaluate(testdata, testlabel)
print('Accuracy: %.2f' % (accuracy*100))