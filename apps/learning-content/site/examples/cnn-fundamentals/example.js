'use strict';
const M=globalThis.CNNMath;
let selected=0;
const byId=id=>document.getElementById(id);
function table(values,caption,highlight,clickable=false) {
  const wrap=document.createElement('div'), t=document.createElement('table');
  const cap=document.createElement('caption');cap.textContent=caption;t.append(cap);
  values.forEach((row,r)=>{const tr=document.createElement('tr');row.forEach((v,c)=>{
    const td=document.createElement('td');td.className=highlight(r,c)?(clickable?'chosen':'patch'):'';
    if(clickable){const b=document.createElement('button');b.type='button';b.textContent=v;b.setAttribute('aria-label',`Output (${r},${c}): ${v}`);b.setAttribute('aria-pressed',highlight(r,c));b.onclick=()=>{selected=r*2+c;renderPatch();};td.append(b);}else td.textContent=v;
    tr.append(td);});t.append(tr);});wrap.append(t);return wrap;
}
function renderPatch(){
  const out=byId('products');
  try {
    const channels=Number(byId('channels').value),bias=byId('bias').valueAsNumber;
    const result=M.convolve(channels,bias),r=Math.floor(selected/2),c=selected%2;
    const relu=byId('relu').checked, shown=result.map(row=>row.map(x=>relu?Math.max(0,x.value):x.value));
    const maps=byId('maps');maps.replaceChildren();
    for(let ch=0;ch<channels;ch++){
      maps.append(table(M.inputs[ch],`Input, channel ${ch}`,(u,v)=>u>=r&&u<r+2&&v>=c&&v<c+2));
      maps.append(table(M.kernels[ch],`Kernel, channel ${ch}`,()=>false));
    }
    maps.append(table(shown,relu?'Output after ReLU':'Output before ReLU',(u,v)=>u===r&&v===c,true));
    const cell=result[r][c];out.replaceChildren();
    for(let ch=0;ch<channels;ch++) {const p=document.createElement('p');p.textContent=`Channel ${ch}: `+cell.products.filter(x=>x.ch===ch).map(x=>`${x.x}×${x.weight}`).join(' + ')+` = ${cell.sums[ch]}`;out.append(p);}
    const p=document.createElement('p');p.textContent=`Position (${r},${c}): ${cell.sums.join(' + ')} + bias ${bias} = ${cell.value}`+(relu?`; ReLU = ${shown[r][c]}`:'');out.append(p);
    const flat=shown.flat();byId('pooled').textContent=`Global pooling of the displayed map: max = ${M.pool(flat,'max')}; average = (${flat.join(' + ')}) / 4 = ${M.pool(flat,'average')}.`;
  }catch(e){byId('maps').replaceChildren();out.textContent=e.message;byId('pooled').textContent='';}
}
function renderShape(){
  const form=byId('shape-form'),keys=['h','w','k','p','s','cin','cout'],v=keys.map(k=>form.elements.namedItem(k).valueAsNumber);
  try{const [h,w,k,p,s,cin,cout]=v, x=M.shape(...v);byId('shape-result').textContent=`H: floor((${h}+2×${p}−${k})/${s})+1 = ${x.h}; W: floor((${w}+2×${p}−${k})/${s})+1 = ${x.w}. Output: ${x.h}×${x.w}×${cout}. Parameters: ${cout}×(${k}²×${cin}+1) = ${x.params} (${x.weights} weights + ${x.biases} biases).`;}
  catch(e){byId('shape-result').textContent=e.message;}
}
byId('patch-form').onsubmit=e=>{e.preventDefault();renderPatch();};
byId('channels').onchange=renderPatch;byId('relu').onchange=renderPatch;
byId('next').onclick=()=>{selected=(selected+1)%4;renderPatch();};
byId('shape-form').onsubmit=e=>{e.preventDefault();renderShape();};
byId('preset').onchange=e=>{const values={tiny:[3,3,3,1,1,3,4],stride:[7,7,3,1,2,1,1],alex224:[224,224,11,1,4,3,96],alex227:[227,227,11,2,4,3,96]}[e.target.value];['h','w','k','p','s','cin','cout'].forEach((k,i)=>byId('shape-form').elements.namedItem(k).value=values[i]);renderShape();};
renderPatch();renderShape();

function renderReceptiveField() {
  const depth=Number(byId('rf-depth').value), stride=Number(byId('rf-stride').value);
  const stages=Array.from({length:depth},()=>[3,stride]);
  const value=M.receptive(stages);
  const fields=stages.map((_,i)=>M.receptive(stages.slice(0,i+1)).field);
  byId('rf-result').textContent=`Layer-by-layer receptive-field widths: ${fields.join(' → ')}. One final response can depend on a ${value.field}×${value.field} original-input region. Neighbouring final responses are spaced ${value.jump} original-input cells apart. This is dependency coverage, not output-map size.`;
}
byId('rf-depth').onchange=renderReceptiveField;
byId('rf-stride').onchange=renderReceptiveField;
renderReceptiveField();
