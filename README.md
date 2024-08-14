compress the image will result in:
300*300 -> 150 * 150 , thus the number of CNN layers should reduced -> the final prediction of the model will change 
需要改变的地方：1、CNN第一个layer的input size要修改 150*150 2、training_generator 的target_size要修改成（150，150）

####第二门课 CNN
1、history = model.fit()
2、图像增强之后的模型 如果和原来的模型训练一样的次数，那么训练集的准确性会低一点，图像增强不会改变每次训练集的图片数量，只是把现有的训练集的数据进行翻转 变换等等 操作，引入更多的随机性，训练时间变长是因为对图片进行这样的预处理更花时间
3、如果验证集的数据和训练集很像，或者没有随机性，即使做了图像增强也会导致验证集的准确率上下波动
4、CNN的input shape必须是 (batchsize,150,150,bites),所以在对一张图片进行预测的时候需要 扩大维数，(1,150,150,3)
#### Transfer learning
inceptionv3-> top = false(全连接层 不要)-> weights = false 舍弃CNN层中的参数 -> 把所有层冻结了 untrainable
getlayers.output 获取 最后一层想要利用的layer的output
通过 x = layers.Flatten()(last_output) 连接上CNN的自定义的最后一层
x = layers.Dropout(0.2)(x) +   model = Model(pre_trained_model.input, x) 进行拼接
