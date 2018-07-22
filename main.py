import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #to split out training and testing data 
from keras.models import Sequential
from keras.layers import Dense

#Setup constants
NUMBER_OF_SENSORS = 2
NUMBER_OF_OUTPUTS = 1

#Creates a one layer model with two weights
def create_model():
	model = Sequential()
	model.add(Dense(NUMBER_OF_OUTPUTS,kernel_initializer='random_uniform',use_bias=False,input_dim=NUMBER_OF_SENSORS))
	model.summary()
	return model

#Generate data to train the neural network
def create_dummy_data():
	#Number of points of data to generate
	arr_size = 100 
	
	#Ground truth
	test_function=lambda i: i*2
	position=np.fromfunction(test_function, (arr_size,), dtype=int)
	
	#Sensor A
	noise_magnitude_a = 50
	noise_sensor_a = ( np.random.rand(position.size) - 0.5 )*noise_magnitude_a # unbiased noise
	sensor_a = position + noise_sensor_a
	
	#Sensor B
	noise_magnitude_b = 20 
	noise_sensor_b = ( np.random.rand(position.size) - 0.5 )*noise_magnitude_b # unbiased noise
	sensor_b = position + noise_sensor_b
	
	Y=np.expand_dims(position, axis=1)

	sensor_a=np.expand_dims(sensor_a, axis=1)
	sensor_b=np.expand_dims(sensor_b, axis=1)
	X=np.concatenate((sensor_a,sensor_b),axis=1)

	return (X,Y)

#Load Neural Network architecture
model=create_model()

#Create dummy data and add noise to the sensors
X,Y=create_dummy_data()

#Split data into train and eval
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.10, random_state=1)

model.compile(loss='mse', optimizer='rmsprop')

#Train model
model.fit(X_train, Y_train,validation_data=( X_valid, Y_valid), batch_size=1, epochs=50,verbose=1)

#Prints weights from neural network 
for layer in model.layers:
    weights = layer.get_weights() 
    print("Weights: " + str(weights))


out=model.predict(X, batch_size=None,steps=1)
loss=np.abs(out-Y)
average_loss=np.mean(loss)
max_loss=np.max(loss)
print("Average loss :"+ str(average_loss))
print("Max loss :"+ str(max_loss))
