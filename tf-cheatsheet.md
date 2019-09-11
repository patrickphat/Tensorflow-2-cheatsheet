

# Create custom layer

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

# Create custom model

So if you want to be able to call `.fit()`, `.evaluate()`, or `.predict()` on those blocks or you want to be able to save and load those blocks separately or something you should use the Model class. A model should contain multiple layers while layer contains only 1 pass-through from input to output (theoretically speaking :D).

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



#  tf.GradientTape()

In tensorflow 2, Gradient Tape is born for the sake of automatic differentiation - computing the gradient of a computation with respect to its input variables. Tensorflow "records" all operations executed inside the context of a [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape) onto a "tape". 

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

    for i in range(target.shape[1]):
      
      # Get output and hidden from decoder
      x, hidden, _  = decoder(x, features, hidden)
      
      # Compute loss function
      loss += loss_func(target[:,i], x)
      ....
    
  .....
  
  train_params =  encoder.trainable_variables + decoder.trainable_variables # Grab all trainable variables
  
  gradients = tape.gradient(loss,train_params) # calculate gradient respected to param
  
  optimizer.apply_gradients(zip(gradients,train_params)) # apply gradients
```



# tf.keras

## .optimizers 

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

## .layers



### .Embedding

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



# tf.dataset

a module for manage and load batches of dataset at training & test time

## .Dataset

### .from_tensor_slices()

Creates a `Dataset` whose elements are slices of the given tensors.


```python
image_dataset = tf.data.Dataset.from_tensor_slices(unique_image_urls)
```
### .map()

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

# tf.reshape()



# tf.expand_dims()

Inserts a dimension of 1 into a tensor's shape.

```python
# 't' is a tensor of shape [2, 3, 5]
(tf.expand_dims(t, 0)).shape  # [1, 2, 3, 5]
(tf.expand_dims(t, 2)).shape  # [2, 3, 1, 5]
(tf.expand_dims(t, 3)).shape  # [2, 3, 5, 1]
```

# tf.arg_max()

Returns the index with the largest value across dimensions of a tensor.

```python

```

# tf.arg_min()

Returns the index with the smallest value across dimensions of a tensor.

```python

```

# References 

**Tensorflow's doc** https://www.tensorflow.org/tutorials/eager/