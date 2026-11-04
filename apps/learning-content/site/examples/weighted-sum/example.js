/* A deterministic forward calculation; no API calls, training or external assets. */
(() => {
  'use strict';
  const form = document.getElementById('neuron-form');
  const output = document.getElementById('calculation');
  const format = value => Number(value.toPrecision(8)).toString();
  const calculate = () => {
    if (!form.reportValidity()) return;
    const read = name => Number(form.elements.namedItem(name).value);
    const inputs = [0, 1, 2].map(i => read(`x${i}`));
    const weights = [0, 1, 2].map(i => read(`w${i}`));
    const bias = read('bias');
    const products = inputs.map((x, i) => x * weights[i]);
    const z = products.reduce((total, value) => total + value, bias);
    if (![...inputs, ...weights, bias, ...products, z].every(Number.isFinite)) {
      output.textContent = 'Use finite numbers small enough to calculate.';
      return;
    }
    const lines = [
      `z = ${inputs.map((x, i) => `(${format(x)} × ${format(weights[i])})`).join(' + ')} + (${format(bias)})`,
      `z = ${products.map(value => `(${format(value)})`).join(' + ')} + (${format(bias)}) = ${format(z)}`,
      `a = ReLU(z) = max(0, ${format(z)}) = ${format(Math.max(0, z))}`,
    ];
    output.replaceChildren(...lines.map(text => {
      const p = document.createElement('p'); p.textContent = text; return p;
    }));
    output.dataset.z = String(z);
    output.dataset.activation = String(Math.max(0, z));
  };
  form.addEventListener('submit', event => { event.preventDefault(); calculate(); });
  form.addEventListener('reset', () => setTimeout(calculate, 0));
  fetch('trace.json').then(response => {
    if (!response.ok) throw new Error('Default trace unavailable');
    return response.json();
  }).then(trace => {
    trace.inputs.forEach((value, i) => { form.elements.namedItem(`x${i}`).value = value; });
    trace.weights.forEach((value, i) => { form.elements.namedItem(`w${i}`).value = value; });
    form.elements.namedItem('bias').value = trace.bias;
    calculate();
  }).catch(() => { calculate(); });
})();
