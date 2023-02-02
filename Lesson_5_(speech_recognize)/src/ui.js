const tfvis = require('@tensorflow/tfjs-vis');

import { speechData } from './data';
import {runButtonCallback} from './index.js';
import {createTensors} from './index.js';

const getDataBox = document.querySelector('div.getdata-box');
const fftSizeOpts = document.getElementById('fft_size_option');
const bufferSizeOpts = document.getElementById('buffer_size_option');
const deleteHFOpt = document.getElementById('delete_high_freq');
const productivityString = document.getElementById('productivity');
const predElements = document.getElementById('pred-container');

// Параметры для спектрограмм, передаются в SpectrogramCreator.
export const preparingDataOptions = {
    fftSize: 0,
    bufferSize: 0,
    deleteHF: false,
};

///////////////////////////////////////////////////////////////////////
///// Метод выводит инфу о прогрессе загрузки и обработки  данных /////
///////////////////////////////////////////////////////////////////////
export function spectroDataProgress(progress, startButton){
    if(startButton){
        const spectroProgress = document.createElement('div');
        spectroProgress.id = 'spectro-progress';
        getDataBox.innerHTML = '';
        getDataBox.append(spectroProgress);
        spectroProgress.insertAdjacentHTML('beforeend', '<div></div>');
        preparingDataOptions.fftSize = fftSizeOpts.value;
        preparingDataOptions.bufferSize = bufferSizeOpts.value;
        preparingDataOptions.deleteHF = deleteHFOpt.checked;
    }
    const progressBar = document.querySelector('#spectro-progress div');
    progressBar.style.width = progress+'%';
}


//////////////////////////////////////////////////////////////////////////////////////////
///// Метод выполняет действия связанные с готовностью данных для создания тензоров. /////
//////////////////////////////////////////////////////////////////////////////////////////
export function flattenDataIsReady(fromJSONFile){
    getDataBox.innerHTML = `<div class="complete">✓&nbsp;Данные загружены и подготовленны!<br>(${speechData.dataInfo.loadedElements} из ${speechData.dataInfo.elemsNum} элементов)</div>`;
    getDataBox.innerHTML += `<button id="create_tensors">Создать тензоры</button>`;
    let saveDataButton, createTensorsButton;
    
    if(!fromJSONFile){
        getDataBox.innerHTML += `<button id="save_data">Сохранить подготовленные данные</button>`;
        saveDataButton = document.getElementById('save_data');
        saveDataButton.addEventListener('click', () => {
            const res = speechData.safeFlatDataInFile();
            saveDataButton.insertAdjacentHTML('afterend', '<a href="'+res+'" download="msc_ready_data.json" target="_blank"><b>Сохранить данные на диск</b></a>');
            saveDataButton.remove();
            // Нужно узнать, можно ли тут как-то ноду использовать для сохранения файла.
        });
    }

    createTensorsButton = document.getElementById('create_tensors');
    createTensorsButton.addEventListener('click', () => {
        createTensors();
        createTensorsButton.remove();
        if(!fromJSONFile) saveDataButton.remove();
    });


}



/////////////////////////////////////////////
///// Получение параметров из интерфеса /////
/////////////////////////////////////////////
function getParamsFromUI(){
    return {
        modelType: document.querySelector('.control select').value,
        epochs: +document.querySelector('.control input[name="epochs"]').value,
    }
}


/////////////////////////////////////////////////////////////
///// Метод действий после завершения создания тензоров /////
/////////////////////////////////////////////////////////////
export function tensorsComplete(){
    const complete = document.querySelector('div.complete');
    complete.innerHTML += '<br>✓&nbsp;Тензоры подготовлены!';
    const runButton = document.querySelector('.control button');
    runButton.removeAttribute('disabled')
    runButton.addEventListener('click', (ev) => {
        const params = getParamsFromUI();
        runButtonCallback(params);
        ev.target.remove();
    }, {once:true})
}



export function showModelParams (params){
    const trainInfo = document.querySelector('div.train-info');
    trainInfo.innerHTML = `<table>
        <tr><td>Размер батча (элементов):</td><td>${params.batchSize}</td></tr>
        <tr><td>Всего батчей:</td><td>${params.totalNumBatches}</td></tr>
        <tr><td>Всего эпох:</td><td>${params.epochs}</td></tr>
        <tr><td>Батчей за эпоху:</td><td>${params.totalNumBatches / params.epochs}</td></tr>
        <tr><td>Размер спектрограмм:</td><td>${params.shape}</td></tr>
        <tr><td>Тренировочных данных:</td><td>${params.trainData}</td></tr>
        <tr><td>Проверочных данных:</td><td>${params.testData}</td></tr>
        </table>
    `;
}


///////////////////////////////////////////////
///// Вывод логов жизненного цикла модели /////
///////////////////////////////////////////////
const trainLog = document.getElementById('logs');
export function logStatus(log, type){
    if(type == 'train'){
         trainLog.innerHTML = `
            &nbsp; ${log.mess}<br>
            &nbsp; Эпоха: ${log.epoch + 1} <br>`;
    }else if(type == 'final'){
        trainLog.innerHTML += `<br>${log.mess}`;
    }
}



////////////////////////////////////
///// Отрисовка графика потерь /////
////////////////////////////////////
const seriesValues = [[], []];                              // Данные обучения по Батчам, данные обучения по эпохам
let firstLossOnEpoch = 0;                                   // Первичное значение потерь в каждой эпохе для первой точки графика потерь на проверочных данных. 
export function plotLoss(logs, progressType){
    if(progressType == 'batchEnd'){
        if(!seriesValues[0].length) firstLossOnEpoch = logs.loss;                       // Если это первая итераация, то запишем в первый элемент - точку для линии по эпохам - равную первому значению первого батча.
        seriesValues[0].push({x: seriesValues[0].length, y: logs.loss});                // Массив точек для линии потерь по батчам.
        seriesValues[1].push({x: seriesValues[1].length, y: firstLossOnEpoch});         // Массив для линии потерь на проверочных данных может отображаться только за эпоху, но linechart не может работать с двумя массивами разной длины, поэтому пока наполним его любыми значениями.
    }else if (progressType == 'epochEnd'){
        const startEpochVal = firstLossOnEpoch;
        const endEpochVal = logs.val_loss;
        let currenVal = startEpochVal;
        const batchSizePerEpoch = logs.batchSize / logs.totalEpochs;
        const step = (startEpochVal - endEpochVal) / batchSizePerEpoch;                                             // У нас есть только две точки за эпоху (начало и конец), но нам нужно наполнить промежуточные элементы, которые созданы за батчи с неактуальным значением, так, чтобы получился ровный отрезок.
        for(let i = logs.currentEpoch * batchSizePerEpoch; i < batchSizePerEpoch * (logs.currentEpoch + 1); i++){   // Пробежимся по массиву и запишем в него новые значения, получился ровный отрезок.
            currenVal -= step;
            seriesValues[1][i].y = currenVal;
        }
        firstLossOnEpoch = currenVal;                                                                               // Укажем новую точку начала отрезка линни для следующей эпохи.
    }

    tfvis.render.linechart(
        document.getElementById('plotLoss'),
        {   
            values: seriesValues,
            series: ['train per Batch', 'validation per Epoch']
        },
        {
            xLabel: 'Batch #',
            yLabel: 'Loss',
            width: 450,
            height: 320,
        }
    );
}



//////////////////////////////////////
///// Отрисовка графика точности /////
//////////////////////////////////////
const seriesValuesAcc = [[], []];
let firstAccOnEpoch = 0;
export function plotAccuracy(logs, progressType){
    if(progressType == 'batchEnd'){
        if(!seriesValuesAcc[0].length) firstAccOnEpoch = logs.acc;
        seriesValuesAcc[0].push({x: seriesValuesAcc[0].length, y: logs.acc});
        seriesValuesAcc[1].push({x: seriesValuesAcc[1].length, y: firstAccOnEpoch});
    }else if (progressType == 'epochEnd'){
        const startEpochVal = firstAccOnEpoch;
        const endEpochVal = logs.val_acc;
        let currenVal = startEpochVal;
        const batchSizePerEpoch = logs.batchSize / logs.totalEpochs;
        const step = (startEpochVal - endEpochVal) / batchSizePerEpoch; 
        for(let i = logs.currentEpoch * batchSizePerEpoch; i < batchSizePerEpoch * (logs.currentEpoch + 1); i++){
            currenVal -= step;
            seriesValuesAcc[1][i].y = currenVal;
        }
        firstAccOnEpoch = currenVal;
    }

    tfvis.render.linechart(
        document.getElementById('plotAccuracy'),
        {   
            values: seriesValuesAcc,
            series: ['train per Batch', 'validation per Epoch']
        },
        {
            xLabel: 'Batch #',
            yLabel: 'Accuracy',
            width: 450,
            height: 320,
        }
    );
}



export function showTestResults(examples, predictions, labels){
    let productivity = 0;
    const testExamples = examples.dataForVisual.shape[0];              // exampleBatch.xs.shape = [100, 28, 28, 1]
    predElements.innerHTML = '<div><div>Real</div><div>Pred</div></div>';
    for (let i = 0; i < testExamples; i++) {
        const prediction = predictions[i];
        const label = labels[i];
        const correct = prediction === label;
        const pred = document.createElement('div');
        pred.className = correct ? 'correct' : 'incorrect';
        pred.innerHTML = `<div>${speechData.dataInfo.classes[label]}</div><div>${speechData.dataInfo.classes[prediction]}</div>`;
        predElements.appendChild(pred);
        if(correct) productivity++;
    }

    productivityString.innerHTML = `Продуктивность модели на данной выборке: <b>${(productivity / testExamples * 100).toFixed(1)}%</b>`;
}