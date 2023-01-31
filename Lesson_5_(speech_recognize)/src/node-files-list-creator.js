
const fs = require('fs');
const path = './dataset/mini_speech_commands';

 
const readDir = async (path) => {
    return new Promise((resolve, reject) => fs.readdir(path, (err , data) => {
        if(err) return reject(err.message);
        resolve(data);
    }))
};



const writeFile = async (path, data) => {
    return new Promise((resolve, reject) => fs.writeFile(path, data, (err) => {
        if(err) return reject(err.message);
        resolve();
    }))
};



async function getFilesByClasses(classes, acc){
    const curClass = classes.shift();
    const files = await readDir(path+'/'+curClass);
    acc[curClass] = files; 
    if(classes.length){
        await getFilesByClasses(classes, acc);
    }else{
        return acc;
    }
}


async function createList(){
    const clasess =  await readDir(path);
    const filesList = {};

    await getFilesByClasses(clasess, filesList);

    const fileData = JSON.stringify(filesList);

    await writeFile(path+'/../msc_dataset.json', fileData);
    console.log('Файл датасетов успешно создан!');
}

createList();