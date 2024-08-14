compress the image will result in:
300*300 -> 150 * 150 , thus the number of CNN layers should reduced -> the final prediction of the model will change 
需要改变的地方：1、CNN第一个layer的input size要修改 150*150 2、training_generator 的target_size要修改成（150，150）

####第二门课 CNN
1、history = model.fit()
2、

