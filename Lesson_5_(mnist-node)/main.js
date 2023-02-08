const data = require('./data');
const model = require('./model');
const tf = require('@tensorflow/tfjs-node');      // Для ускоренного обучения. На процессорах без инструкций AVX работать не будет, выдаст ошибку: A dynamic link library (DLL) initialization routine failed.
//const tf = require('@tensorflow/tfjs');             // Альтернатива, если tfjs-node не работает, будет медленно обучать, но работать будет на любом оборудовании.



let batchTime = {start: 0, end: 0};


const EPOCHS = 20;
const BATCH_SIZE = 128; 
const MODEL_SAVE_PATH = null;



async function run(epochs, batchSize, modelSavePath) {
    await data.loadData();
    const {images: trainImages, labels: trainLabels} = data.getTrainData();
    console.log('Обучающие данные подготовленны!');
    model.summary();


    console.log('Обучение модели...');
    const validationSplit = 0.15;                                                           // Проверочные данные - 15% 
    const numTrainExamplesPerEpoch = trainImages.shape[0] * (1 - validationSplit);          // Число примеров за эпоху  10000 * (1 - 0.15) = 8500
    const numTrainBatchesPerEpoch = Math.ceil(numTrainExamplesPerEpoch / batchSize);        // Число батчей за эпоху
    await model.fit(trainImages, trainLabels, {
        epochs,
        batchSize,
        validationSplit,
/*         callbacks: {
            onBatchBegin: () => {
                batchTime.start = Date.now();
            },
            onBatchEnd: (batch, logs) => {
                batchTime.end = Date.now();
                console.log('Батч: '+batch, ' | Время обработки: '+ ((batchTime.end - batchTime.start) / 1000)+ ' секунд');
                console.log(logs);
            }
        } */
    });
    console.log('Обучение завершено!');

    const {images: testImages, labels: testLabels} = data.getTestData();
    console.log('Контрольные данные подготовленны!');

    const evalOutput = model.evaluate(testImages, testLabels);
    console.log(
        `\nРезультат на контрорльных данных:\n` +
        `Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

    if (modelSavePath != null) {
        await model.save(`file://${modelSavePath}`);
        console.log(`Модель сохранена в: ${modelSavePath}`);
    }

}



run(EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH);