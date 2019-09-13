###### 

<center><img src='https://miro.medium.com/max/4928/1*-QTg-_71YF0SVshMEaKZ_g.png'> </center>
# TF 2.0 cheatsheet! 

<a href='https://github.com/patrickphatnguyen/Tensorflow-2-cheatsheet'><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width=50></a>A **minimal** doc on **Tensorflow 2.0**  (let's call it a cheatsheet!) that you don't need to wander around and  start being confused of all modules available in Tensorflow (If you're a  newbie), because I already put all the things in one place, only **necessary things**  are mentioned in the cheatsheet that is just enough to make you  dangerous. So you can later learn by yourself more complex modules in  Tensorflow.

`Author` <a href='https://github.com/patrickphatnguyen'><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width=40></a> <a href='https://www.linkedin.com/in/tphat/'><img src='https://cdn1.iconfinder.com/data/icons/logotypes/32/square-linkedin-512.png' width=40></a> Patrick Nguyen (University of Information Technology, Vietnam) 

`References` from Tensorflow docs and bunch of places on the internet!

# Working with Tensors

Tensors are the most lowest-level units that you have to work on if you want to build any kind of Neural Network using Tensorflow.

## tf.constant()

## tf.variable()

## tf.convert_to_tensors()

## tf.random

Outputs random values from a distribution.

### .normal()

```python
# Create a normally distributed tensor with the shape of (12,48,48)
normal_tensor = tf.random.normal((12,48,48), mean=1, stddev=1.0) 
```

### .uniform()

```python
# Create a uniformly distributed tensor with the shape of (12,48,48)
uniform_tensor = tf.random.uniform((12,48,48))
```

### .gamma()
```python
# Create a u gamma-ly distributed tensor with the shape of (12,48,48)
uniform_tensor = tf.random.gamma((12,48,48))
```



## tf.reshape()

<img src='https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/basic/transpose_reshape/reshape.png' width=500>

Given `tensor`, this operation returns a tensor that has the same values as `tensor` with shape `shape`

```python
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]

reshape(t, [3, 3]) # [[1, 2, 3],
                   #  [4, 5, 6],
                   #  [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]

reshape(t, [2, 4]) #  [[1, 1, 2, 2],
                   #   [3, 3, 4, 4]]
```



## tf.expand_dims()

Inserts a dimension of 1 into a tensor's shape.

```python
# 't' is a tensor of shape [2, 3, 5]
(tf.expand_dims(t, 0)).shape  # [1, 2, 3, 5]
(tf.expand_dims(t, 2)).shape  # [2, 3, 1, 5]
(tf.expand_dims(t, 3)).shape  # [2, 3, 5, 1]
```

## tf.arg_max()

Returns the index with the largest value across dimensions of a tensor.

```python
import tensorflow as tf
a = tf.convert_to_tensor([[1, 10, 7], 
                           [3, 4, 9]])

tf.argmax(a,axis=1) # [1,2]
```

## tf.arg_min()

Returns the index with the largest value across dimensions of a tensor.

```python
import tensorflow as tf
a = tf.convert_to_tensor([[1, 10, 7], 
                           [3, 4, 9]])
tf.argmin(a,axis=1) # [0,0]
```

## tf.reduce_sum()

Return the sum of all elements in the tensor or sum of the tensor along specified axis

``` python
a = tf.convert_to_tensor([[1, 10, 7], 
                           [3, 4, 9]])
tf.reduce_sum(a) # 34
tf.reduce_sum(a,axis = 0) # [4,14,16]
tf.reduce_sum(a,axis = 1) # [18,16]
```

## tf.reduce_prod()

Return the product of all elements in the tensor or product of the tensor along specified axis

```python
a = tf.convert_to_tensor([[1, 10, 7], 
                           [3, 4, 9]])
tf.reduce_prod(a) # 7560
tf.reduce_prod(a,axis = 0) # [3,40,63]
tf.reduce_prod(a,axis = 1) # [70,108]
```

## tf.reduce_mean()
Return the product of all elements in the tensor or product of the tensor along specified axis.

```python
a = tf.convert_to_tensor([[1, 10, 7], 
                           [3, 4, 9]])
tf.reduce_mean(a) # 7560
tf.reduce_mean(a,axis = 0) # [3,40,63]
tf.reduce_(a,axis = 1) # [70,108]
```

## tf.reduce_max()
Return the max of all elements in the tensor or product of the tensor along specified axis.

```python
a = tf.convert_to_tensor([[1, 10, 7], 
                           [3, 4, 9]])
tf.reduce_max(a) # 10
tf.reduce_max(a,axis = 0) # [3,10,9]
tf.reduce_max(a,axis = 1) # [70,108]
```

## tf.reduce_min()
Return the min of all elements in the tensor or product of the tensor along specified axis.

```python
a = tf.convert_to_tensor([[1, 10, 7], 
                           [3, 4, 9]])
tf.reduce_min(a) # 1
tf.reduce_min(a,axis = 0) # [1,4,7]
tf.reduce_min(a,axis = 1) # [1,3]
```




# Loading and preprocessing data

## tf.dataset 

a module for manage and load batches of dataset at training & test time

### .Dataset

#### .from_tensor_slices()

Creates a `Dataset` whose elements are slices of the given tensors.

```python
image_dataset = tf.data.Dataset.from_tensor_slices(unique_image_urls)
```

#### .map()

After create a dataset from_tensor_slices(), you can then transform this dataset using a mapping function. For e.g:

```python
# Mapping function
def load_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image,channels = 3)
  image = tf.image.resize(image,(299,299))
  image = tf.keras.applications.inception_v3.preprocess_input(image)
  return image, image_path
# Transform dataset
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
# Then you can iterate
for img, path in image_dataset:
  batch_features = feature_extractor_model(img)
    ....

```

## tf.keras.utils

### .get_file()

This module allows you to download a file via a link and put that to a specified folder. You can extract the file if you wish!

```python
data_path = os.path.abspath('.') + "/my_data_folder/"
train_zip_url = 'http://images.cocodataset.org/zips/train2014.zip'
name_of_zip = "my_train_data.zip"

train_zip = tf.keras.utils.get_file(name_of_zip,
                                      cache_subdir=data_path,
                                      origin = train_zip_url ,
                                      extract = True)

```



# Modelling

## tf.keras

### .layers

#### .Conv1D

#### .Conv2D

#### .Conv3D

#### .GlobalAveragePooling1D

#### .GlobalAveragePooling2D

#### .GlobalAveragePooling3D

#### .GlobalMaxPooling1D

`shortcut` GlobalMaxPool1D



#### .GlobalMaxPooling2D

`shortcut`

#### .GlobalMaxPooling3D

#### 

#### 

#### .Dense

#### .Dropout

#### .Embedding

#### .Flatten

#### .LSTM

#### .GRU

Turns positive integers (indexes) into dense vectors of fixed size.

Inherits From: [`Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)

e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

```python
vocab_size = 5000
embedding_dim = 256
my_input = tf.Variable([Tokenizer.word_index['dog']] *3) # value: [64,64,64] shape: [3]
Embedding_Layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
Embedding_Layer(my_input) # shape: [3,256]
```



### .Sequential

Create a model that is linearly stacked with layers

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))
```

## Functional API

## Create custom layers

If you want to create your own custom layers, you can inherit from `tf.keras.layers.Layer` class then create your own using atomic tensorflow operations.

```python
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable("kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
```



## Create custom models

So if you want to be able to call `.fit()`, `.evaluate()`, or `.predict()` on those blocks or you want to be able to save and load those blocks separately or something you should use the Model class. A model should contain multiple layers while layer contains only 1 pass-through from input to output (theoretically speaking).

```python
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    .....
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    .....

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

```



# Automatic Differentiation

## tf.GradientTape

In tensorflow 2, Gradient Tape is born for the sake of **automatic differentiation** - computing the gradient of a computation with respect to its input variables. Tensorflow "records" all operations executed inside the context of a `tf.GradientTape` onto a "tape". 

```python
def f(w1, w2):
     return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)
 
gradients = tape.gradient(z, [w1, w2])
```

You can also using `tf.GradientTape()` to compute gradient corresponded to a chain of computation with respected to some variables, and then apply these gradients to variables via a declared optimizer.

```python
# Init loss
  loss = 0
  
  with tf.GradientTape() as tape:
    # Put through CNN encoder
    features = encoder(image_tensor) 
    ..... 

    for i in range(target.shape[1])
      
      # Get output and hidden from decoder
      x, hidden, _  = decoder(x, features, hidden)
      
      # Compute loss function
      loss += loss_func(target[i], x)
      ....
    
  .....
  
  train_params =  encoder.trainable_variables + decoder.trainable_variables # Grab all trainable variables
  
  gradients = tape.gradient(loss,train_params) # calculate gradient respected to param
  
  optimizer.apply_gradients(zip(gradients,train_params)) # apply gradients
```

## tf.keras

### .optimizers 

#### .Adam

Optimizer that implements the Adam algorithm.

#### .Adagrad

Optimizer that implements the Adagrad algorithm.

#### .Adadelta

Optimizer that implements the Adadelta algorithm.

#### .Adamax

Optimizer that implements the Adamax algorithm.

#### .Ftrl

Optimizer that implements the FTRL algorithm.

#### .Nadam 

Optimizer that implements the NAdam algorithm.

#### .RMSprop

Optimizer that implements the RMSprop algorithm.

#### .SGD

Stochastic gradient descent and momentum optimizer.




# References 

**Tensorflow's doc** https://www.tensorflow.org/tutorials/eager/