import { Lookup } from './lookup.ts';

export class CorpusLookup {
  inputLookup: Lookup | undefined;
  outputLookup: Lookup | undefined;
  numInputs: number = 0;
  numOutputs: number = 0;
  constructor(features: Array<string> | undefined, intents: Array<string> | undefined) {
    if (features && intents) {
      this.inputLookup = new Lookup(undefined);
      this.outputLookup = new Lookup(undefined);
      for (let i = 0; i < features.length; i += 1) {
        this.inputLookup.add(features[i]);
      }
      for (let i = 0; i < intents.length; i += 1) {
        this.outputLookup.add(intents[i]);
      }
      this.numInputs = this.inputLookup.items.length;
      this.numOutputs = this.outputLookup.items.length;
    }
  }

  build(corpus: any) {
    this.inputLookup = new Lookup(corpus, 'input');
    this.outputLookup = new Lookup(corpus, 'output');
    this.numInputs = this.inputLookup.items.length;
    this.numOutputs = this.outputLookup.items.length;
    const result = [];
    for (let i = 0; i < corpus.length; i += 1) {
      const { input, output } = corpus[i];
      result.push({
        input: this.inputLookup.prepare(input),
        output: this.outputLookup.prepare(output),
      });
    }
    return result;
  }

  transformInput(input: any) {
    return this.inputLookup ? this.inputLookup.prepare(input) : undefined;
  }
}
