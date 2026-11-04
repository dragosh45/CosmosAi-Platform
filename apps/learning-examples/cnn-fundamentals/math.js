/* Numerical teaching operations: dense channel mixing, no groups or dilation. */
(function (root) {
  'use strict';
  function integer(n, min, max, name) {
    if (!Number.isInteger(n) || n < min || n > max) throw new Error(`${name}: use an integer between ${min} and ${max}.`);
  }
  function shape(h, w, k, p, s, cin, cout) {
    for (const [v, lo, hi, label] of [[h,1,4096,'H'],[w,1,4096,'W'],[k,1,255,'K'],[p,0,255,'P'],[s,1,255,'S'],[cin,1,4096,'Input channels'],[cout,1,4096,'Filters']]) integer(v,lo,hi,label);
    const oh = Math.floor((h + 2*p - k) / s) + 1;
    const ow = Math.floor((w + 2*p - k) / s) + 1;
    if (oh < 1 || ow < 1) throw new Error('The kernel does not fit the padded input.');
    return {h:oh, w:ow, params:cout*(k*k*cin+1), weights:cout*k*k*cin, biases:cout};
  }
  const inputs = [[[0,1,2],[1,2,3],[2,3,4]], [[1,2,3],[2,3,4],[3,4,5]]];
  const kernels = [[[0,1],[1,2]], [[1,2],[2,3]]];
  function convolve(channels, bias) {
    integer(channels,1,2,'Channels');
    if (!Number.isFinite(bias) || Math.abs(bias)>1000) throw new Error('Bias must be finite, between -1000 and 1000.');
    return [0,1].map(r => [0,1].map(c => {
      const sums = [], products = [];
      for (let ch=0; ch<channels; ch++) {
        let sum=0;
        for(let u=0;u<2;u++) for(let v=0;v<2;v++) {
          const x=inputs[ch][r+u][c+v], weight=kernels[ch][u][v];
          products.push({ch,u,v,x,weight,value:x*weight}); sum+=x*weight;
        }
        sums.push(sum);
      }
      return {sums, products, value:sums.reduce((a,b)=>a+b, bias)};
    }));
  }
  function pool(values, mode) {
    if (!values.length || !values.every(Number.isFinite)) throw new Error('Pooling needs finite values.');
    if(mode==='max') return Math.max(...values);
    if(mode==='average') return values.reduce((a,b)=>a+b,0)/values.length;
    throw new Error('Unknown pooling mode.');
  }
  function receptive(layers) {
    let field=1, jump=1;
    for(const [k,s] of layers) { integer(k,1,255,'K'); integer(s,1,255,'S'); field+=(k-1)*jump; jump*=s; }
    return {field,jump};
  }
  const api={shape,convolve,pool,receptive,inputs,kernels};
  if(typeof module!=='undefined' && module.exports) module.exports=api;
  else root.CNNMath=api;
})(typeof globalThis!=='undefined'?globalThis:this);
