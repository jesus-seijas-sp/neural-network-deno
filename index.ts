import { NeuralNetwork } from './src/neural-network.ts';

const corpus = JSON.parse(Deno.readTextFileSync('./corpus-en.json'));
const numRuns = 1000;

function arrToObject(arr: any) {
  const result: any = {};
  arr.forEach((word: string) => {
    result[word] = 1;
  });
  return result;
}

function normalize(text: string) {
  return text
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
}

function tokenize(text: string) {
  return normalize(text).split(/[\s,.!?;:([\]'"¡¿)/]+/).filter((x) => x);
}

function formatUtterance(utterance: string, intent: string) {
  return {
    input: arrToObject(tokenize(utterance)),
    output: arrToObject([intent])
  }
}

function buildTrainData() {
  const result: any[] = [];
  corpus.data.forEach((item: any) => {
    item.utterances.forEach((utterance: string) => {
      result.push(formatUtterance(utterance, item.intent));
    });
  });
  return result;
}

function formatOutput(output: any) {
  const result: any[] = [];
  Object.keys(output).forEach(key => {
    result.push({ label: key, score: output[key] });
  });
  return result.sort((a, b) => b.score - a.score);
}


const trainData = buildTrainData();
const net = new NeuralNetwork({ log: false });
net.train(trainData);
let total = 0;
let good = 0;
const hrstart = performance.now();
for (let run = 0; run < numRuns; run += 1) {
  corpus.data.forEach((item: any) => {
    item.tests.forEach((test: string) => {
      const validationData = formatUtterance(test, item.intent);
      const output = net.run(validationData.input);
      const classifications = formatOutput(output);
      total += 1;
      if (classifications[0].label === item.intent) {
        good += 1;
      }
    });
  });
}
const elapsed = performance.now() - hrstart;
const timePerUtterance = elapsed / total;
const utterancesPerSecond = 1000 / timePerUtterance;
console.log(`Total runs: ${numRuns}`);
console.log(`${good / numRuns} good of a total of ${total / numRuns} (${good*100/total}%)`);
console.log(
  `Milliseconds per utterance: ${timePerUtterance}`
);
console.log(
  `Utterances per second: ${utterancesPerSecond}`
);

console.log(`${good} good of a total of ${total} (${good * 100 / total}%)`);