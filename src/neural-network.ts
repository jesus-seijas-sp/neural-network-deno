import { CorpusLookup } from './corpus-lookup.ts';

const defaultSettings = {
  iterations: 20000,
  errorThresh: 0.00005,
  fixedError: false,
  deltaErrorThresh: 0.000001,
  learningRate: 0.6,
  momentum: 0.5,
  alpha: 0.07,
  log: false,
};

export class NeuralNetwork {
  settings: any;
  logFn: Function | undefined;
  perceptronsByName: any;
  perceptrons: any;
  outputs: any;
  numPerceptrons: number = 0;
  lookup: CorpusLookup | undefined;
  decayLearningRate: number = 0.6;
  status: any;
  constructor(settings = {}) {
    this.settings = settings;
    this.applySettings(this.settings, defaultSettings);
    if (this.settings.log === true) {
      this.logFn = (status: any, time: number) =>
        console.log(
          `Epoch ${status.iterations} loss ${status.error} time ${time}ms`
        );
    } else if (typeof this.settings.log === 'function') {
      this.logFn = this.settings.log;
    }
  }

  applySettings(obj: any = {}, settings: any = {}) {
    Object.keys(settings).forEach((key) => {
      if (obj[key] === undefined) {
        obj[key] = settings[key];
      }
    });
    return obj;
  }

  initialize(numInputs: number, outputNames: Array<string>) {
    this.perceptronsByName = {};
    this.perceptrons = [];
    this.outputs = {};
    this.numPerceptrons = outputNames.length;
    for (let i = 0; i < outputNames.length; i += 1) {
      const name = outputNames[i];
      this.outputs[name] = 0;
      const perceptron = {
        name,
        id: i,
        weights: new Float32Array(numInputs),
        changes: new Float32Array(numInputs),
        bias: 0,
      };
      this.perceptrons.push(perceptron);
      this.perceptronsByName[name] = perceptron;
    }
  }

  runInputPerceptron(perceptron: any, input: any) {
    const sum = input.keys.reduce(
      (prev: number, key: number) => prev + input.data[key] * perceptron.weights[key],
      perceptron.bias
    );
    return sum <= 0 ? 0 : this.settings.alpha * sum;
  }

  runInput(input: any) {
    for (let i = 0; i < this.numPerceptrons; i += 1) {
      this.outputs[this.perceptrons[i].name] = this.runInputPerceptron(
        this.perceptrons[i],
        input
      );
    }
    return this.outputs;
  }

  get isRunnable() {
    return !!this.numPerceptrons;
  }

  run(input: any) {
    return this.lookup
      ? this.runInput(this.lookup.transformInput(input))
      : undefined;
  }

  prepareCorpus(corpus: any) {
    this.lookup = new CorpusLookup(undefined, undefined);
    return this.lookup.build(corpus);
  }

  verifyIsInitialized() {
    if (!this.perceptrons && this.lookup && this.lookup.outputLookup) {
      this.initialize(this.lookup.numInputs, this.lookup.outputLookup.items);
    }
  }

  trainPerceptron(perceptron: any, data: any) {
    const { alpha, momentum } = this.settings;
    const { changes, weights } = perceptron;
    let error = 0;
    for (let i = 0; i < data.length; i += 1) {
      const { input, output } = data[i];
      const actualOutput = this.runInputPerceptron(perceptron, input);
      const expectedOutput = output.data[perceptron.id] || 0;
      const currentError = expectedOutput - actualOutput;
      if (currentError) {
        error += currentError ** 2;
        const delta =
          (actualOutput > 0 ? 1 : alpha) *
          currentError *
          this.decayLearningRate;
        for (let j = 0; j < input.keys.length; j += 1) {
          const key = input.keys[j];
          const change = delta * input.data[key] + momentum * changes[key];
          changes[key] = change;
          weights[key] += change;
        }
        perceptron.bias += delta;
      }
    }
    return error;
  }

  train(corpus: any) {
    if (!corpus || !corpus.length) {
      return {};
    }
    const useNoneFeature =
      corpus[corpus.length - 1].input.nonefeature !== undefined;
    if (useNoneFeature) {
      const intents: any = {};
      for (let i = 0; i < corpus.length - 1; i += 1) {
        const tokens = Object.keys(corpus[i].output);
        for (let j = 0; j < tokens.length; j += 1) {
          if (!intents[tokens[j]]) {
            intents[tokens[j]] = 1;
          }
        }
      }
      const current = corpus[corpus.length - 1];
      const keys = Object.keys(intents);
      for (let i = 0; i < keys.length; i += 1) {
        current.output[keys[i]] = 0.0000001;
      }
    }
    const data = this.prepareCorpus(corpus);
    if (!this.status) {
      this.status = { error: Infinity, deltaError: Infinity, iterations: 0 };
    }
    this.verifyIsInitialized();
    const minError = this.settings.errorThresh;
    const minDelta = this.settings.deltaErrorThresh;
    while (
      this.status.iterations < this.settings.iterations &&
      this.status.error > minError &&
      this.status.deltaError > minDelta
    ) {
      const hrstart = new Date();
      this.status.iterations += 1;
      this.decayLearningRate =
        this.settings.learningRate / (1 + 0.001 * this.status.iterations);
      const lastError = this.status.error;
      this.status.error = 0;
      for (let i = 0; i < this.numPerceptrons; i += 1) {
        this.status.error += this.trainPerceptron(this.perceptrons[i], data);
      }
      this.status.error /= this.numPerceptrons * data.length;
      this.status.deltaError = Math.abs(this.status.error - lastError);
      const hrend = new Date();
      if (this.logFn) {
        this.logFn(this.status, hrend.getTime() - hrstart.getTime());
      }
    }
    return this.status;
  }
}
