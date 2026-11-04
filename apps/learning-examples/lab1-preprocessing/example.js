'use strict';
const fmt = n => Number(n.toFixed(6)).toString();
function fillGap(event) {
  if (event) event.preventDefault();
  const form = document.querySelector('#gap-form');
  const left = Number(form.elements.left.value), right = Number(form.elements.right.value), missing = Number(form.elements.missing.value);
  const target = document.querySelector('#gap-result');
  if (![left, right].every(Number.isFinite) || !Number.isInteger(missing) || missing < 1 || missing > 20) { target.textContent = 'Use finite endpoints and 1–20 missing readings.'; return; }
  const step = (right-left)/(missing+1);
  const filled = Array.from({length: missing}, (_,i) => left+(i+1)*step);
  target.textContent = `Step = (${fmt(right)} − ${fmt(left)}) / ${missing+1} = ${fmt(step)} A.\nBefore: [${fmt(left)}, ${Array(missing).fill('NaN').join(', ')}, ${fmt(right)}]\nAfter: [${[left,...filled,right].map(fmt).join(', ')}]`;
  target.style.whiteSpace = 'pre-wrap';
}
function normalize(event) {
  if (event) event.preventDefault();
  const form = document.querySelector('#norm-form');
  const parts = form.elements.training.value.split(',').map(v=>v.trim());
  const train = parts.map(Number), reading = Number(form.elements.reading.value);
  const target = document.querySelector('#norm-result');
  if (parts.some(v=>!v) || train.length<2 || train.length>1000 || ![...train,reading].every(Number.isFinite)) { target.textContent = 'Enter 2–1000 finite training readings separated by commas.'; return; }
  const mu = train.reduce((a,b)=>a+b,0)/train.length;
  const deviations = train.map(x=>(x-mu)**2);
  const sigma = Math.sqrt(deviations.reduce((a,b)=>a+b,0)/(train.length-1));
  if (!Number.isFinite(sigma) || sigma===0) { target.textContent = 'The training readings need a finite, nonzero spread.'; return; }
  target.textContent = `Training mean = (${train.map(fmt).join(' + ')}) / ${train.length} = ${fmt(mu)} A.\nSquared distances from mean: [${deviations.map(fmt).join(', ')}].\nStandard deviation = √(${fmt(deviations.reduce((a,b)=>a+b,0))} / ${train.length-1}) = ${fmt(sigma)} A.\nNormalised reading = (${fmt(reading)} − ${fmt(mu)}) / ${fmt(sigma)} = ${fmt((reading-mu)/sigma)}.\nThe mean and spread come only from the training readings.`;
  target.style.whiteSpace = 'pre-wrap';
}
document.querySelector('#gap-form').addEventListener('submit', fillGap);
document.querySelector('#norm-form').addEventListener('submit', normalize);
fillGap(); normalize();
