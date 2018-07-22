import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #to split out training and testing data 
from keras.models import Sequential
from keras.layers import Dense

#Creates a one layer model with two weights
def create_model():
	model = Sequential()
	model.add(Dense(1,kernel_initializer='random_uniform',use_bias=False,input_dim=2))
	model.summary()
	return model


def create_dummy_data():
	arr_size=100
	position=np.fromfunction(lambda i: i*2, (arr_size,), dtype=int)
	sensor_a=position+(np.random.rand(position.size)-0.5)*50 # unbiased noise
	sensor_b=position+(np.random.rand(position.size)-0.5)*20 # unbiased noise
	
	Y=np.expand_dims(position, axis=1)

	sensor_a=np.expand_dims(sensor_a, axis=1)
	sensor_b=np.expand_dims(sensor_b, axis=1)
	X=np.concatenate((sensor_a,sensor_b),axis=1)

	return (X,Y)

#Load Neural Network architecture
model=create_model()
X,Y=create_dummy_data()

#Split data into train and eval
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.10, random_state=1)

model.compile(loss='mse', optimizer='rmsprop')

#Train model
model.fit(X_train, Y_train,validation_data=( X_valid, Y_valid), batch_size=1, epochs=100,verbose=1)

for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    print("Weights: " + str(weights))


out=model.predict(X, batch_size=None,steps=1)
loss=np.abs(out-Y)
average_loss=np.mean(loss)
max_loss=np.max(loss)
print("Average loss :"+ str(average_loss))
print("Max loss :"+ str(max_loss))
