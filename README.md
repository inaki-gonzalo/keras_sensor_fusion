# Sensor Fusion using Neural Network
Optimal sensor fusion using a one layer neural netwrok.
The idea is based on sensor fusion using noise variances.
Estimate = A*sensor_a + B*sensor_b
where A= Var(noise_sensor_b)/(Var(noise_sensor_a)+Var(noise_sensor_b))
and B= Var(noise_sensor_a)/(Var(noise_sensor_a)+Var(noise_sensor_b))
We can numerically get the variances by trainning a Neural Netwrok. 


