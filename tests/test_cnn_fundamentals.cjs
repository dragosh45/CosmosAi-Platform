const {test}=require('node:test');
const assert=require('node:assert/strict');
const m=require('../apps/learning-examples/cnn-fundamentals/math.js');
test('shape floors incomplete windows and counts shared parameters once',()=>{
  assert.deepEqual(m.shape(3,3,3,1,1,3,4),{h:3,w:3,params:112,weights:108,biases:4});
  assert.equal(m.shape(8,8,3,0,2,3,4).h,3);
  assert.equal(m.shape(7,7,3,1,2,1,1).w,4);
  assert.equal(m.shape(224,224,11,1,4,3,96).h,54);
  assert.equal(m.shape(227,227,11,2,4,3,96).h,56);
  assert.equal(m.shape(227,227,11,0,4,3,96).h,55);
  assert.equal(m.shape(32,64,3,1,1,3,4).params,112);
});
test('channel contributions sum before adding one bias at each position',()=>{
  assert.deepEqual(m.convolve(1,0).map(r=>r.map(x=>x.value)),[[6,10],[10,14]]);
  assert.deepEqual(m.convolve(2,0).map(r=>r.map(x=>x.value)),[[24,36],[36,48]]);
  assert.equal(m.convolve(2,2)[1][1].value,50);
  assert.equal(m.convolve(2,0)[0][0].products.length,8);
});
test('pooling and receptive fields include negative values and stride jumps',()=>{
  assert.equal(m.pool([-1,-3,-2,-8],'max'),-1);
  assert.equal(m.pool([1,3,2,8],'average'),3.5);
  assert.deepEqual(m.receptive([[3,1],[3,1]]),{field:5,jump:1});
  assert.deepEqual(m.receptive([[3,1],[3,1],[3,1]]),{field:7,jump:1});
  assert.deepEqual(m.receptive([[3,2],[3,2],[3,2]]),{field:15,jump:8});
  assert.deepEqual(m.receptive([[3,2],[3,1]]),{field:7,jump:2});
});
test('invalid shapes and nonfinite input are rejected',()=>{
  for(const args of [[1,1,3,0,1,1,1],[3,3,3,0,0,1,1],[3.5,3,2,0,1,1,1]]) assert.throws(()=>m.shape(...args));
  assert.throws(()=>m.convolve(2,NaN));assert.throws(()=>m.pool([],'max'));
});
