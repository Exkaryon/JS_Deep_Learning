const tf = require('@tensorflow/tfjs');


// Упрощенная модель, которая используется в браузере.
const model = tf.sequential({
    layers: [
        tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 16, activation: 'relu'}),
        tf.layers.maxPooling2d({poolSize: 2, strides: 2}),
        tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}),
        tf.layers.maxPooling2d({poolSize: 2, strides: 2}),
        tf.layers.flatten(),
        tf.layers.dense({units: 64, activation: 'relu'}),
        tf.layers.dense({units: 10, activation: 'softmax'})
    ]
});


// Расширенная модель для обучения с помощью Node.js.
/*
const model = tf.sequential({
    layers: [
        tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
        }),
        tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
        }),
        tf.layers.maxPooling2d({poolSize: [2, 2]}),
        tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }),
        tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }),
        tf.layers.maxPooling2d({poolSize: [2, 2]}),
        tf.layers.flatten(),
        tf.layers.dropout({rate: 0.25}),                    // Слои дропаута снижают риск переобучения модели. 0.25 - означает, что 25% элементов входного тензора будет случайным образом обнулено, тем самым создавая шум, который помогает избежать случаайно возникающие паттерны, несущественные относительно истинных паттернов данных.
        tf.layers.dense({
            units: 512,
            activation: 'relu'
        }),
        tf.layers.dropout({rate: 0.5}),
        tf.layers.dense({
            units: 10,
            activation: 'softmax'
        })
    ]
});
*/

model.compile({
  optimizer: 'rmsprop',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

console.log('Модель создана:');

module.exports = model;
