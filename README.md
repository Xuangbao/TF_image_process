![image](https://github.com/user-attachments/assets/7b75879d-6dea-4ac1-b816-b3393f6c2bdb)compress the image will result in:
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
1、padding：针对一个大的list包含几句单词不一样长的话，有三种方式进行填充（必须得到的list of list of numbers是相同长度的）：前填充、后填充、不填充
默认是后填充：sequences_post = vectorize_layer(sentences)
也可以前填充：tf_dataset= tf.data.Dataset.from_tensor_slices(sentences)
            df = tf_dataset.map(vectorize_layer) ##得到不一样长的语句的vector的编码
            sequences_pre = tf.keras.utils.pad_sequences(df, padding='pre')
也可以不填充 省略...
2、在自然语言处理（NLP）模型中，词汇表外（Out-Of-Vocabulary, OOV）的词通常是用一个特殊的标记来表示的，比如 <UNK>（unknown, 未知）。这样，当模型遇到不在词汇表中的单词时，就会用这个特殊的标记来替代，而不是用0或者是其他替代方法。
3、普通张量是一个具有固定形状的多维数组。例如，一个形状为 [2, 3] 的二维张量可以表示为：
[[1,2,3],[1,2,3]
RaggedTensor 允许每行（或更高维度的子数组）具有不同的长度
[[1, 2],
 [3, 4, 5],
 [6]]
 #################
def padding_func(sequences):
  '''Generates padded sequences from a tf.data.Dataset'''
  #只有ragged_batch才能存在
  # Put all elements in a single ragged batch
  sequences = sequences.ragged_batch(batch_size=sequences.cardinality())
    #ragged_batch：将 tf.data.Dataset 中的所有序列合并成一个批次，并且保留每个序列的原始长度。
  print(type(sequences))
  # Output a tensor from the single batch
  sequences = sequences.get_single_element()
  print(type(sequences))
  # Pad the sequences 函数的input是numpy数组
  padded_sequences = tf.keras.utils.pad_sequences(sequences.numpy(), 
                                                  maxlen=MAX_LENGTH, 
                                                  truncating=TRUNC_TYPE, 
                                                  padding=PADDING_TYPE
                                                 )
  print(type(padded_sequences))
  # Convert back to a tf.data.Dataset
  padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)

  return padded_sequences
  ############################
如果你直接对 tf.data.Dataset 进行填充而不先使用 ragged_batch()，会遇到以下问题：

无法直接处理不规则形状的张量：tf.data.Dataset 的每个元素在默认情况下必须具有相同的形状。如果数据集中有不规则长度的序列，TensorFlow 将无法直接将它们组合到一起进行填充。
处理复杂性：使用 ragged_batch() 可以有效地将不规则的序列合并并保留其长度，之后再将它们转为 numpy 数组，这样可以方便地使用 pad_sequences 进行填充。
 ########################
4、batch和tensor区别：
如果单个样本是一个形状为 [28, 28, 3] 的张量（表示一张 28x28 像素的 RGB 图像），那么一个包含 32 个样本的批次将表示为形状为 [32, 28, 28, 3] 的张量。
一个批次本质上是一个张量，但它多了一维来表示批次大小
5、<class 'tensorflow.python.data.ops.batch_op._BatchDataset'> 是 TensorFlow 中的一个数据结构，它表示 tf.data.Dataset 的一种特殊类型，称为 BatchDataset。这个数据结构是在使用 batch() 方法将数据集划分为批次时生成的
6、train_dataset_final = (train_dataset_vectorized
                       .cache()
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .prefetch(PREFETCH_BUFFER_SIZE)
                       .batch(BATCH_SIZE)
                       )
随机选择过程：当你调用 shuffle(1000) 时，TensorFlow 会将前 1000 张图像加载到缓冲区中，然后在这些图像中随机选择一张输出给下一步的处理。在输出一张图像后，缓冲区会从数据集中再读取一张新的图像填充进来，以保持缓冲区中有 1000 张图像供随机选择。
打乱过程：前 1000 张图像被加载到缓冲区中，TensorFlow 在这些图像中随机选择一个输出。
批次生成：batch(32) 会从打乱后的输出中每次取出 32 张图像，形成一个批次。由于这些图像的顺序是经过 shuffle 处理的，因此它们是随机组合的。
7、在模型训练中：这种将输入数据和标签数据打包成元组的形式是训练机器学习模型（尤其是深度学习模型）时的标准做法。在模型训练时，fit 方法期望接收的数据集格式通常是 (input, label) 的形式。
8、tf.data.Dataset 更像是一个生成器或迭代器，而不是一个静态的数据结构。它定义了数据生成的过程，而不是存储实际数据。因此，tf.data.Dataset 需要通过 map() 等操作逐个处理数据，而不能直接像列表或数组那样批量传递给 vectorize_layer。
9、用 sub_wordtocken的好处在于可以把一个单词切分成字母，所以在解码没见过的单词的时候可以认出来更多未知的单词。因而在decode一个string的时候也会让sequence的长度更长
10、需要处理掉每句话中的stopwords，目的：This should improve the performance of your classifier by removing frequently used words that don't add information to determine the topic of the news
11、model input_shape的问题
如果你有不同长度的句子，每个词被嵌入为一个 128 维的向量，那么 input_shape 可以是 (None, 128)，这里 None 表示句子的词数不固定，而 128 是词嵌入的维度。
表示一个固定长度的序列输入，其中 sequence_length 是序列的长度，并且只有1维
12、在初始化处理text层的时候，可以 tf.keras.layers.TextVectorization(standardize = func,output_sequence_length= ) 加入这两个参数，第一个参数表示这个层在处理序列中的每个语句的时候都调用该函数（可以是去没用词的时候），另外一个参数保证了 使用Vectorization（text）的时候保证每句话的输出 是经过填充的该size的一个向量
13、train_batch = next(train_proc_dataset.as_numpy_iterator())
validation_batch = next(validation_proc_dataset.as_numpy_iterator())
print(f"Shape of the train batch: {train_batch[0].shape}")
print(f"Shape of the validation batch: {validation_batch[0].shape}")
答案：（32，120）
next()：用于从迭代器中获取下一个元素。在这个上下文中，它返回的是 train_proc_dataset 中的一个批次的数据。
14、好好复习一下C3W2_assignment的细节，有很多细节
15、concatenate函数可以直接用于两个tf.dataset的（还可以用于很多其他相同格式的两个东西，类似于list的相加）拼接 A.concatenate(B)
16、在初始化Textvectorlize_layer的时候设置最大可以给多少个单词打标签 在填充（padding）的时候决定每句话保留多少个单词
17、Conv1D用于text的分组任务（同理用GlobalMaxPooling1D），因为每一个input text都是一维向量，而比如（120,）,注意写input size的时候不要加上batch_size这个维度，但是最后的model.summary()是有none即batch这个维度在最前面的
18、embedding层里面的output（120，16），16只是用来看的，不会给这个句子增加维度，最后的主要目标还是给句子进行分组
19、conv1d(120,5,activation='relu')的意思是120个convolution，然后每个取的小区域都是5*1的，所以头尾要去掉两个维度最后
20\当一个tf.dataset被分成批次之后，len(train_dataset) 返回的是一共 有多少批，不是数据的总长度
21、有两个细节：
对于一个包含很多句子的语料库，他是遍布每个句子，然后从每个句子转成的number sequence中生成素材（然后对每个素材进行切分成label和text），这样最大化利用了每个句子作为素材






