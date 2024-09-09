import tensorflow as tf # type: ignore
import numpy as np 
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', #order is important
               'Dress', 'Coat','Sandal', 'Shirt', 
               'Sneaker', 'Bag', 'Ankle boot']

def plot_image(index, prediction_array,answers,image):
    prediction=prediction_array[i]
    true_label=answers[i]
    img=image[i]
    plt.grid=(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img,cmap=plt.cm.binary)
    predicted_label= np.argmax(prediction)

    if predicted_label == true_label:
        color='blue'
    else: color='red'

    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(prediction):2.0f}% ({class_names[true_label]})",
                                color=color)
    
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array[i], color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array[i])

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#modeling 784 to 128 to 10 with reLU
model= tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])

#creating the lost function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=1)
#converting output layer to probability distribution of 10 categories
probability_model=tf.keras.Sequential([model,tf.keras.layers.Softmax()])
prediction=probability_model.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, prediction, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, prediction, test_labels)
plt.tight_layout()
plt.show()
