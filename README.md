compress the image will result in:
300*300 -> 150 * 150 , thus the number of CNN layers should reduced -> the final prediction of the model will change 
需要改变的地方：1、CNN第一个layer的input size要修改 150*150 2、training_generator 的target_size要修改成（150，150）

####第二门课 CNN
1、history = model.fit()
2、图像增强之后的模型 如果和原来的模型训练一样的次数，那么训练集的准确性会低一点，图像增强不会改变每次训练集的图片数量，只是把现有的训练集的数据进行翻转 变换等等 操作，引入更多的随机性，训练时间变长是因为对图片进行这样的预处理更花时间
3、如果验证集的数据和训练集很像，或者没有随机性，即使做了图像增强也会导致验证集的准确率上下波动
4、CNN的input shape必须是 (batchsize,150,150,bites),所以在对一张图片进行预测的时候需要 扩大维数，(1,150,150,3)
#### Transfer learning
inceptionv3（需指定引用该模型的input size是多少）-> top = false(全连接层 不要)-> weights = None 舍弃CNN层中的参数 -> 把所有层冻结了 untrainable
get_layers('layers名字').output 获取 最后一层想要利用的layer的output
通过 x = layers.Flatten()(last_output) 连接上CNN的自定义的最后一层
###用dropout来避免过拟合
x = layers.Dropout(0.2)(x) +   model = Model(pre_trained_model.input, x) #### model函数进行拼接
当然可以在transfer learning 的时候进行data augumentation，因为最底下的 全连接层 是需要训练的
###
多元分类
改动的地方：
1、train_generator 里面class mode要修改
2、model.compile里面的loss function要修改
3、最后的输出函数改成 softmax+ 神经元数量修改
####loss function
4、注意多元分类时，如果你的标签是整数形式（即每个标签是一个单独的类别编号，如 [0, 1, 2, ...]），但模型期望的是 one-hot 编码的标签（即 [0, 0, 1, ...] 形式的向量），这时loss function要用 sparse_categorical_crossentropy 
5、用flow方法加载数据，见assignment C2W4
##################
第三门课 NLP
1、padding：针对一个大的list包含几句单词不一样长的话，有三种方式进行填充：前填充、后填充、不填充
2、












