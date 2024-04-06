# Hand-Gesture-Motor-Control
1) Firstly, run the master node for ROS framework using "roscore" command in terminal.
2) Now, run the "rosrun rosserial_python serial_node.py _port:=/dev/ttyACM1 _baud:=115200" to create a serial communication between ROS framework and Arduino UNO board connected to the designated port.
3) Now, run the Hand_gesture.py file so that camera window opens and you can give input through hand gestures. Note that this command must be run after going to the path of the Hand_gesture.py filr.
4) On receiving the inputs, model processing would start and Arduino UNO would be publising the PWM to the DC Motor according to the output of the model.
5) The DC Motor are connected through the MD10c motor driver to the Arduino UNO.
