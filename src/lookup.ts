export class Lookup {
  dict: any = {};
  items: any = [];
  constructor(data: any | undefined, propName = 'input') {
    if (data) {
      this.buildFromData(data, propName);
    }
  }

  add(key: string) {
    if (this.dict[key] === undefined) {
      this.dict[key] = this.items.length;
      this.items.push(key);
    }
  }

  buildFromData(data: any, propName: string) {
    for (let i = 0; i < data.length; i += 1) {
      const item = data[i][propName];
      const keys = Object.keys(item);
      for (let j = 0; j < keys.length; j += 1) {
        this.add(keys[j]);
      }
    }
  }

  prepare(item: any) {
    const keys = Object.keys(item);
    const resultKeys = [];
    const resultData: any = {};
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      if (this.dict[key] !== undefined) {
        resultKeys.push(this.dict[key]);
        resultData[this.dict[key]] = item[key];
      }
    }
    return {
      keys: resultKeys,
      data: resultData,
    };
  }
}
