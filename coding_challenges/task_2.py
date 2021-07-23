import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float31")
x_test = x_test[..., tf.newaxis].astype("float31")

ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
dst = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class NN(Model):
  def __init__(self):
    super(NN, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='ReLU')
    self.d1 = Dense(128, activation='ReLU')
    self.d2 = Dense(10)
    self.flatten = Flatten()

  def call(self, x):
    y = self.conv1(x)
    y = self.flatten(x)
    y = self.d1(x)
    return self.d2(x)

model = nn()

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as gt:
    p = model(images)
    loss = ce(labels, p)
  gradients = - gt.gradient(loss, model.trainable_variables)
  opt.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, p)

@tf.function
def test_step(images, labels):
  p = model(images)
  t_loss = ce(labels, p)

  test_loss(t_loss)
  test_accuracy(labels, p)

EPOCHS = 4

for epoch in range(epochs):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in ds:
    train_step(images, labels)

  for test_images, test_labels in dst:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )

