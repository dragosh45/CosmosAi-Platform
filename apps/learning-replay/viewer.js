/* CosmosAI M50-L2: read-only viewer for the measured format-version-2 trace. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const labels = ["elliptical", "spiral", "lenticular", "irregular"];
  const channelNames = ["red", "green", "blue"];
  const maxTraceBytes = 2000000;
  const returnKey = "cosmosai-replay-return";
  const selectionKeys = ["updateIndex", "imageIndex", "filterIndex", "channelIndex", "outputRow", "outputColumn", "classIndex", "selectedParameter"];
  const state = {
    trace: null,
    updateIndex: 0,
    imageIndex: 0,
    filterIndex: 0,
    channelIndex: 0,
    outputRow: 1,
    outputColumn: 1,
    classIndex: 0,
    selectedParameter: "features.0.weight|0,0,1,1",
    playing: false,
    timer: null,
  };

  function setStatus(message, isError) {
    const element = $("trace-status");
    element.textContent = message;
    element.className = `status ${isError ? "status-error" : "status-ok"}`;
  }

  function formatNumber(value, digits = 6) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "unavailable";
    const number = Number(value);
    if (Object.is(number, -0)) return "0.000000";
    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
  }

  function formatPlain(value, digits = 6) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "unavailable";
    return Number(value).toFixed(digits);
  }

  function shapeOf(value) {
    const shape = [];
    let cursor = value;
    while (Array.isArray(cursor)) {
      shape.push(cursor.length);
      cursor = cursor[0];
    }
    return shape;
  }

  function getAt(value, index) {
    return index.reduce((current, position) => current[position], value);
  }

  function flatten(value, output = []) {
    if (Array.isArray(value)) value.forEach((item) => flatten(item, output));
    else output.push(value);
    return output;
  }

  function valueClass(value) {
    if (Math.abs(Number(value)) < 0.0000005) return "zero";
    return Number(value) > 0 ? "positive" : "negative";
  }

  function matrixMarkup(values, digits = 6) {
    if (!Array.isArray(values) || !Array.isArray(values[0])) return `<p class="notice">Matrix unavailable</p>`;
    return `<table class="matrix compact"><tbody>${values.map((row) => `<tr>${row.map((value) => `<td class="${valueClass(value)}">${formatPlain(value, digits)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  function nestedSelectMarkup(values, title, digits = 6) {
    return `<div class="plane"><h4>${title}</h4>${matrixMarkup(values, digits)}</div>`;
  }

  function currentUpdate() {
    return state.trace.updates[state.updateIndex];
  }

  function parameterKey(name, index) {
    return `${name}|${index.join(",")}`;
  }

  function parseParameterKey(key) {
    const [name, rawIndex] = key.split("|");
    return { name, index: rawIndex.split(",").filter(Boolean).map(Number) };
  }

  function parameterValue(update, name, index, field = "before") {
    const parameter = update.parameters[name];
    return parameter ? getAt(parameter[field], index) : null;
  }

  function parameterMeaning(name, index) {
    if (name === "features.0.weight") {
      const [filter, channel, row, column] = index;
      return `${name}[${index.join(",")}] is one convolution weight: filter/output channel ${filter}, input channel ${channel} (${channelNames[channel]}), kernel row ${row}, kernel column ${column}. It is one of 108 convolution weights. The last two indices are kernel coordinates. An output map position such as (1,1) is separate: it shows where this same weight is reused, not which weight is selected.`;
    }
    if (name === "features.0.bias") return `${name}[${index[0]}] is the bias added to every spatial response produced by filter ${index[0]}. It is one of 4 convolution biases.`;
    if (name === "classifier.weight") return `${name}[${index.join(",")}] is one classifier weight: class ${index[0]} (${labels[index[0]]}) connected to pooled feature a[${index[1]}]. It is one of 16 feature-to-class weights.`;
    if (name === "classifier.bias") return `${name}[${index[0]}] is the bias for class ${index[0]} (${labels[index[0]]}). It is one of 4 classifier biases.`;
    return `${name}[${index.join(",")}] is one stored scalar parameter. Its gradient and SGD update are shown below.`;
  }

  function validateTrace(trace) {
    if (!trace || trace.format_version !== 2) throw new Error("This viewer needs observer trace format_version 2.");
    if (!Array.isArray(trace.updates) || trace.updates.length < 1 || trace.updates.length > 20) throw new Error("Trace must contain between 1 and 20 updates.");
    if (trace.parameter_count !== 132) throw new Error("This educational viewer expects the 132-parameter TinyGalaxyCNN.");
    trace.updates.forEach((update, index) => {
      if (!Array.isArray(update.images) || !update.parameters || !update.pre_relu_maps_before || !update.selected_convolution_gradient) {
        throw new Error(`Update ${index + 1} is missing required measured fields.`);
      }
      if (!Array.isArray(update.image_ids) || update.image_ids.length !== update.images.length) throw new Error(`Update ${index + 1} has inconsistent image metadata.`);
    });
    return trace;
  }

  function buildSelect(select, items, selected) {
    select.innerHTML = items.map((item) => `<option value="${item.value}">${item.label}</option>`).join("");
    if (selected !== undefined) select.value = selected;
  }

  function populateControls() {
    const update = currentUpdate();
    buildSelect($("image-select"), update.image_ids.map((id, index) => ({ value: index, label: `${String.fromCharCode(65 + index)}: ${id} (${labels[update.labels[index]]})` })), state.imageIndex);
    buildSelect($("filter-select"), [0, 1, 2, 3].map((value) => ({ value, label: `Filter ${value}` })), state.filterIndex);
    buildSelect($("row-select"), [0, 1, 2].map((value) => ({ value, label: String(value) })), state.outputRow);
    buildSelect($("column-select"), [0, 1, 2].map((value) => ({ value, label: String(value) })), state.outputColumn);
    buildSelect($("class-select"), labels.map((label, index) => ({ value: index, label: `${index}: ${label}` })), state.classIndex);

    const parameterOptions = [];
    Object.entries(update.parameters).forEach(([name, parameter]) => {
      for (const index of indexTuples(parameter.shape)) {
        parameterOptions.push({
          value: parameterKey(name, index),
          label: `${name}[${index.join(",")}]`,
        });
      }
    });
    buildSelect($("parameter-select"), parameterOptions, state.selectedParameter);
    if (!$("parameter-select").value) state.selectedParameter = parameterOptions[0].value;
  }

  function indexTuples(shape) {
    const result = [];
    function visit(prefix, depth) {
      if (depth === shape.length) { result.push(prefix); return; }
      for (let index = 0; index < shape[depth]; index += 1) visit(prefix.concat(index), depth + 1);
    }
    visit([], 0);
    return result;
  }

  function renderPlaneSection(update) {
    const image = update.images[state.imageIndex];
    const filter = getAt(update.parameters["features.0.weight"].before, [state.filterIndex]);
    $("image-meta").textContent = `${String.fromCharCode(65 + state.imageIndex)} / ${update.image_ids[state.imageIndex]} / ${labels[update.labels[state.imageIndex]]}`;
    $("map-meta").textContent = `Filter ${state.filterIndex}, output (${state.outputRow}, ${state.outputColumn})`;
    $("input-plane-values").innerHTML = image.map((plane, index) => nestedSelectMarkup(plane, `${channelNames[index]} plane`)).join("");
    $("kernel-slices").innerHTML = filter.map((plane, index) => nestedSelectMarkup(plane, `${channelNames[index]} kernel`)).join("");
    const preMap = update.pre_relu_maps_before[state.imageIndex][state.filterIndex];
    const postMap = update.post_relu_maps_before[state.imageIndex][state.filterIndex];
    const pooledFeatures = update.features_before[state.imageIndex];
    const postReluMaps = update.post_relu_maps_before[state.imageIndex];
    const poolingCards = pooledFeatures.map((value, index) => {
      const map = postReluMaps[index];
      const sum = map.reduce((total, row) => total + row.reduce((rowTotal, cell) => rowTotal + cell, 0), 0);
      const mapRows = map.map((row) => `[ ${row.map((cell) => formatPlain(cell)).join("  ")} ]`).join(" + ");
      const selectedNote = index === state.filterIndex
        ? `Selected response ${formatNumber(map[state.outputRow][state.outputColumn])} is included in this filter's sum.`
        : "";
      return `<span class="pooled-chip"><code>a[${index}]</code> from filter ${index}<span class="pooled-map-label">After-ReLU 3x3 values:</span><span class="pooled-map-values">${mapRows}</span><span class="pooled-equation">sum = ${formatPlain(sum)}; a[${index}] = ${formatPlain(sum)} / 9 = <strong>${formatNumber(value)}</strong></span>${selectedNote ? `<span class="pooled-selected-note">${selectedNote}</span>` : ""}</span>`;
    }).join("");
    $("pooled-summary").innerHTML = `<span class="pooled-summary-label">Global average pooling: each filter's 3x3 After-ReLU map is summed and divided by 9:</span>${poolingCards}`;
    $("pre-map").innerHTML = matrixMarkup(preMap);
    $("post-map").innerHTML = matrixMarkup(postMap);
    const response = preMap[state.outputRow][state.outputColumn];
    const pooled = update.features_before[state.imageIndex][state.filterIndex];
    $("response-explanation").textContent = `Selected convolution response z_conv[${state.outputRow},${state.outputColumn}] = ${formatNumber(response)}. ReLU gives ${formatNumber(postMap[state.outputRow][state.outputColumn])}; average pooling gives a[${state.filterIndex}] = ${formatNumber(pooled)}. This a[${state.filterIndex}] is an input to the classifier below.`;
    renderWindowExplanation(image, paddingFor(update));
    renderProducts(update, image, filter);
  }

  function paddingFor(update) {
    return Number(update.selected_convolution_gradient.padding || 0);
  }

  function renderWindowExplanation(image, padding) {
    const height = image[0].length;
    const width = image[0][0].length;
    const firstRow = state.outputRow - padding;
    const lastRow = firstRow + 2;
    const firstColumn = state.outputColumn - padding;
    const lastColumn = firstColumn + 2;
    const rowRange = `${firstRow}..${lastRow}`;
    const columnRange = `${firstColumn}..${lastColumn}`;
    const invalidRows = [];
    const invalidColumns = [];
    for (let value = firstRow; value <= lastRow; value += 1) if (value < 0 || value >= height) invalidRows.push(value);
    for (let value = firstColumn; value <= lastColumn; value += 1) if (value < 0 || value >= width) invalidColumns.push(value);
    const paddingText = invalidRows.length || invalidColumns.length
      ? `Coordinates outside the real ${height}x${width} image are zero-padding${invalidRows.length ? ` (row${invalidRows.length > 1 ? "s" : ""} ${invalidRows.join(", ")})` : ""}${invalidColumns.length ? ` (column${invalidColumns.length > 1 ? "s" : ""} ${invalidColumns.join(", ")})` : ""}.`
      : "Every coordinate in this window is a real input pixel.";
    const explanation = `Output z[${state.outputRow},${state.outputColumn}] is a location, not an input row. The unchanged 3x3 filter is placed there, so it reads input rows ${rowRange} and columns ${columnRange}. For kernel offset (k,j), input = (output row + k - padding, output column + j - padding). ${paddingText}`;
    $("window-explanation").textContent = explanation;
    $("window-formula").textContent = `Why the table changes: output (${state.outputRow},${state.outputColumn}) moves the same filter window to a new location. Row ${state.outputRow} uses input rows ${rowRange}; column ${state.outputColumn} uses input columns ${columnRange}. The kernel weights stay fixed, but each weight is now paired with the pixel at its new coordinate.`;
  }

  function renderProducts(update, image, filter) {
    const padding = paddingFor(update);
    const rows = [];
    const subtotals = [0, 0, 0];
    let example = null;
    for (let channel = 0; channel < 3; channel += 1) {
      for (let kernelRow = 0; kernelRow < 3; kernelRow += 1) {
        for (let kernelColumn = 0; kernelColumn < 3; kernelColumn += 1) {
          const inputRow = state.outputRow + kernelRow - padding;
          const inputColumn = state.outputColumn + kernelColumn - padding;
          const input = inputRow < 0 || inputColumn < 0 || inputRow >= image[0].length || inputColumn >= image[0][0].length ? 0 : image[channel][inputRow][inputColumn];
          const weight = filter[channel][kernelRow][kernelColumn];
          const product = input * weight;
          subtotals[channel] += product;
          if (example === null && input !== 0) {
            example = { channel, inputRow, inputColumn, kernelRow, kernelColumn, input, weight, product };
          }
          rows.push(`<tr><td>${channelNames[channel]}</td><td>${inputRow}, ${inputColumn}</td><td>${formatNumber(input)}</td><td>${formatNumber(weight)}</td><td>${formatNumber(product)}</td></tr>`);
        }
      }
    }
    const bias = update.parameters["features.0.bias"].before[state.filterIndex];
    $("position-meta").textContent = `row ${state.outputRow}, column ${state.outputColumn}`;
    const response = update.pre_relu_maps_before[state.imageIndex][state.filterIndex][state.outputRow][state.outputColumn];
    $("position-formula").textContent = `z_conv[${state.outputRow},${state.outputColumn}] = 27 input x weight products + bias ${formatNumber(bias)} = ${formatNumber(response)}`;
    const exampleText = example
      ? `Example of one product: ${channelNames[example.channel]} input at (${example.inputRow},${example.inputColumn}) ${formatNumber(example.input)} x ${channelNames[example.channel]} kernel weight at offset (${example.kernelRow},${example.kernelColumn}) ${formatNumber(example.weight)} = ${formatNumber(example.product)}. This is one of the 27 products.`
      : "This window contains only zero-padding values, so every product is zero; the bias still contributes.";
    $("product-explanation").textContent = `${exampleText} General rule: product = input value x matching kernel weight; each channel subtotal = its nine products added together; response = red subtotal + green subtotal + blue subtotal + bias.`;
    $("products-table").querySelector("tbody").innerHTML = rows.join("");
    const fullResponse = `${formatPlain(subtotals[0])} + ${formatPlain(subtotals[1])} + (${formatPlain(subtotals[2])}) + ${formatPlain(bias)} = ${formatPlain(response)}`;
    $("subtotal-strip").innerHTML = subtotals.map((value, index) => `<div class="subtotal"><span>${channelNames[index]} products subtotal</span><strong>${formatNumber(value)}</strong></div>`).concat(`<div class="subtotal"><span>red + green + blue + bias</span><strong>${fullResponse}</strong></div>`).join("");
  }

  function renderNetwork(update) {
    const features = update.features_before[state.imageIndex];
    const weights = update.parameters["classifier.weight"].before;
    const biases = update.parameters["classifier.bias"].before;
    const logits = update.logits_before[state.imageIndex];
    const probabilities = update.probabilities_before[state.imageIndex];
    const width = 600; const height = 300;
    const featureY = [45, 115, 185, 255]; const classY = [45, 115, 185, 255];
    let svg = `<svg class="network-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Four pooled features connected to four class logits">`;
    for (let feature = 0; feature < 4; feature += 1) {
      for (let classIndex = 0; classIndex < 4; classIndex += 1) {
        const weight = weights[classIndex][feature];
        const selected = classIndex === state.classIndex;
        svg += `<line class="${selected ? "selected" : ""}" x1="140" y1="${featureY[feature]}" x2="445" y2="${classY[classIndex]}" stroke="${weight >= 0 ? "#247a55" : "#a75b1d"}" stroke-width="${1 + Math.min(3, Math.abs(weight) * 5)}" />`;
        if (selected) {
          const labelX = 292;
          const labelY = (featureY[feature] + classY[classIndex]) / 2 - 3;
          svg += `<text class="weight-label" x="${labelX}" y="${labelY}" text-anchor="middle">${formatNumber(weight)}</text>`;
        }
      }
    }
    for (let feature = 0; feature < 4; feature += 1) svg += `<circle cx="140" cy="${featureY[feature]}" r="22" fill="#d8f2e1" /><text x="140" y="${featureY[feature] - 3}" text-anchor="middle">a[${feature}]</text><text class="small" x="140" y="${featureY[feature] + 11}" text-anchor="middle">${formatPlain(features[feature], 4)}</text>`;
    for (let classIndex = 0; classIndex < 4; classIndex += 1) svg += `<circle cx="445" cy="${classY[classIndex]}" r="22" fill="#e7defb" /><text x="445" y="${classY[classIndex] - 3}" text-anchor="middle">${classIndex}</text><text class="small" x="445" y="${classY[classIndex] + 11}" text-anchor="middle">${labels[classIndex]}</text><text class="small" x="480" y="${classY[classIndex] - 3}" text-anchor="start">z=${formatPlain(logits[classIndex], 4)}</text><text class="small" x="480" y="${classY[classIndex] + 11}" text-anchor="start">p=${formatPlain(probabilities[classIndex], 4)}</text>`;
    svg += `<text class="small" x="140" y="292" text-anchor="middle">pooled features a[j]</text><text class="small" x="445" y="292" text-anchor="middle">class logits z[i] -> softmax p[i]</text></svg>`;
    $("network-diagram").innerHTML = svg;
    $("classifier-meta").textContent = `${String.fromCharCode(65 + state.imageIndex)} / ${update.image_ids[state.imageIndex]}`;
    $("features-list").innerHTML = features.map((value, index) => `<div class="value-chip"><span>a[${index}]</span><strong>${formatNumber(value)}</strong></div>`).join("");
    const matrixRows = weights.map((row) => `<div>[ ${row.map((value) => formatPlain(value)).join("  ")} ]</div>`).join("");
    const featureVector = features.map((value) => `<div>${formatPlain(value)}</div>`).join("");
    const biasVector = biases.map((value) => `<div>${formatPlain(value)}</div>`).join("");
    const logitVector = logits.map((value) => `<div>${formatPlain(value)}</div>`).join("");
    const probabilityVector = probabilities.map((value) => `<div>${formatPlain(value, 6)}</div>`).join("");
    $("classifier-schematic").innerHTML = `<div class="schematic-flow"><div class="schematic-block"><strong>W</strong><small>4 x 4 weights</small><div class="schematic-matrix">${matrixRows}</div></div><span class="schematic-symbol">x</span><div class="schematic-block"><strong>a</strong><small>4 x 1 features</small><div class="schematic-vector">${featureVector}</div></div><span class="schematic-symbol">+</span><div class="schematic-block"><strong>b</strong><small>4 x 1 bias</small><div class="schematic-vector">${biasVector}</div></div><span class="schematic-symbol">=</span><div class="schematic-block"><strong>z</strong><small>4 x 1 logits</small><div class="schematic-vector">${logitVector}</div></div><span class="schematic-arrow">-&gt;</span><div class="schematic-block probability-block"><strong>softmax(z)</strong><small>4 x 1 probabilities</small><div class="schematic-vector">${probabilityVector}</div></div></div><div class="schematic-equation"><code>W x a + b = z</code><span>then</span><code>softmax(z) = p</code></div><p class="schematic-caption">This uses column vectors: each row of W takes all four pooled features, adds one bias, and produces one class logit. Softmax compares all four logits and turns them into probabilities.</p>`;
    $("classifier-matrix").innerHTML = `<table class="matrix"><thead><tr><th>class</th>${[0,1,2,3].map((i) => `<th>a[${i}]</th>`).join("")}</tr></thead><tbody>${weights.map((row, classIndex) => `<tr><th>${classIndex}: ${labels[classIndex]}</th>${row.map((value) => `<td class="${valueClass(value)}">${formatPlain(value)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    $("classifier-general-formula").textContent = `General form: z[i] = sum over j of (W[i,j] x a[j]) + b[i]. Here W has shape [4 classes x 4 pooled features], a has shape [4], and z has one logit per class. Each matrix cell is one learned feature-to-class weight.`;
    $("logit-list").innerHTML = logits.map((value, index) => `<div class="logit-chip"><span>${index}: ${labels[index]}</span><strong>z = ${formatNumber(value)}<br>p = ${formatPlain(probabilities[index], 6)}</strong></div>`).join("");
    const row = weights[state.classIndex];
    const contributionText = row.map((weight, index) => `${formatPlain(features[index])} x ${formatPlain(weight)} = ${formatNumber(features[index] * weight)}`).join("; ");
    $("classifier-connection-example").textContent = `${labels[state.classIndex]} example: ${contributionText}; bias ${formatPlain(biases[state.classIndex])} -> z[${state.classIndex}] = ${formatNumber(logits[state.classIndex])} -> p[${state.classIndex}] = ${formatPlain(probabilities[state.classIndex], 6)}. The four line contributions and bias are summed before softmax compares this class with the other three.`;
    $("classifier-order").textContent = `Order: first calculate all four class logits z[i] from the four pooled features. Only after every z[i] exists does softmax convert the logits into probabilities p[i].`;
    $("classifier-formula").textContent = `Class ${state.classIndex} logit = ${row.map((weight, index) => `(${formatPlain(features[index])} x ${formatPlain(weight)})`).join(" + ")} + bias ${formatPlain(biases[state.classIndex])} = ${formatPlain(logits[state.classIndex])}. Softmax is applied after all four logits exist.`;
    const denominator = logits.map((value) => `exp(${formatNumber(value)})`).join(" + ");
    $("softmax-explanation").textContent = `Softmax for ${state.classIndex}: p[${state.classIndex}] = exp(z[${state.classIndex}]) / sum_k exp(z[k]) = exp(${formatNumber(logits[state.classIndex])}) / (${denominator}) = ${formatPlain(probabilities[state.classIndex], 6)}. The same denominator is used for every class, so all four probabilities sum to 1.`;
  }

  function renderParameterPanel(update) {
    const parsed = parseParameterKey(state.selectedParameter);
    const parameter = update.parameters[parsed.name];
    if (!parameter) return;
    const before = parameterValue(update, parsed.name, parsed.index, "before");
    const gradient = parameterValue(update, parsed.name, parsed.index, "gradient");
    const after = parameterValue(update, parsed.name, parsed.index, "after");
    const delta = parameterValue(update, parsed.name, parsed.index, "delta");
    $("parameter-shape").textContent = `shape [${parameter.shape.join(", ")}]`;
    $("parameter-meaning").textContent = parameterMeaning(parsed.name, parsed.index);
    $("parameter-equation").innerHTML = `<div class="equation-title">SGD update: use the gradient to change this stored parameter</div><div>gradient already calculated: dL/dw = <strong>${formatNumber(gradient)}</strong></div><div>delta = -learning_rate x gradient = -${formatPlain(state.trace.learning_rate)} x (${formatNumber(gradient)}) = <strong>${formatNumber(delta)}</strong></div><div>new parameter = old parameter + delta</div><div>${formatNumber(after)} = ${formatNumber(before)} + (${formatNumber(delta)})</div><div>old parameter = value before SGD; new parameter = value after SGD</div>`;
    if (parsed.name === "features.0.weight" && parsed.index.join(",") === "0,0,1,1") renderConvolutionGradient(update);
    else if (parsed.name === "classifier.weight" || parsed.name === "classifier.bias") renderClassifierGradient(update, parsed);
    else $("gradient-source").innerHTML = `<p class="notice">${parameterMeaning(parsed.name, parsed.index)} The full trace records this scalar's gradient, old value, new value and delta. The detailed contribution table is expanded for the worked shared convolution weight <code>features.0.weight[0,0,1,1]</code> and classifier parameters; other convolution weights are summarized so the page stays readable.</p>`;
    renderGradientDiagram(update, parsed, before, gradient, after, delta);
    renderInventory(update);
  }

  function renderGradientDiagram(update, parsed, before, gradient, after, delta) {
    const isConvolution = parsed.name === "features.0.weight";
    const selected = `${parsed.name}[${parsed.index.join(",")}]`;
    const sourceTitle = isConvolution ? "response cells" : "batch examples";
    const sourceDetail = isConvolution ? "A: 9 map positions + B: 9 map positions" : "each example contributes to this parameter";
    const rule = isConvolution ? "sum(dL/dz x aligned input)" : "sum(upstream error x input)";
    const detail = isConvolution
      ? "The selected weight is reused at each map position. Each source contributes a product; all 18 products become one gradient for this scalar."
      : "Each source contributes one product to this scalar's gradient; other parameters have their own gradients.";
    $("gradient-diagram").innerHTML = `<div class="gradient-diagram-title">How this selected scalar changes</div><div class="gradient-flow"><div class="gradient-stage"><span>${sourceTitle}</span><div class="gradient-node">${sourceDetail}<small>${isConvolution ? "dL/dz x input" : "upstream error x input"}</small></div></div><div class="gradient-arrow">-&gt;</div><div class="gradient-stage"><span>accumulate</span><div class="gradient-node gradient-node-focus"><strong>dL/dw</strong><small>${rule}</small><b>${formatNumber(gradient)}</b></div></div><div class="gradient-arrow">-&gt;</div><div class="gradient-stage"><span>SGD update</span><div class="gradient-node gradient-update-node"><small>old parameter</small><b>${formatNumber(before)}</b><small>delta = -${formatPlain(state.trace.learning_rate)} x (${formatNumber(gradient)})</small><b>${formatNumber(delta)}</b><small>new parameter = old + delta</small><b>${formatNumber(after)}</b></div></div></div><p class="gradient-diagram-note"><code>${selected}</code>: ${detail}</p>`;
  }

  function renderConvolutionGradient(update) {
    const trace = update.selected_convolution_gradient;
    const focus = trace.contributions.find((item) => item.image_index === state.imageIndex && item.output_row === state.outputRow && item.output_column === state.outputColumn);
    const rows = trace.contributions.map((item) => {
      const focused = item === focus;
      return `<tr class="${focused ? "gradient-focus" : ""}"><td>${item.image_index === 0 ? "A" : "B"}</td><td>${item.output_row}, ${item.output_column}</td><td>${formatNumber(item.aligned_input)}</td><td>${formatNumber(item.dL_dz)}</td><td>${formatNumber(item.product)}</td></tr>`;
    }).join("");
    const focusNote = focus
      ? `Selected row: image ${focus.image_index === 0 ? "A" : "B"}, map position (${focus.output_row},${focus.output_column}) means the convolution response z_conv[${focus.output_row},${focus.output_column}] at that spatial location. It is not a weight. The aligned input is the pixel value paired with the selected kernel weight at that location: ${formatNumber(focus.aligned_input)}. dL/dz = ${formatNumber(focus.dL_dz)} is the measured local derivative for that response (shown rounded), and its contribution is ${formatNumber(focus.dL_dz)} x ${formatNumber(focus.aligned_input)} = ${formatNumber(focus.product)}.`
      : "Select a matching image and map position to highlight one contribution.";
    $("gradient-source").innerHTML = `<p class="formula-note"><strong>Gradient calculation for this shared convolution weight</strong><br>${trace.equation}<br>${focusNote}<br>Each table row is one spatial use of the same weight. The final gradient is not one row: it is the sum of the Gradient contribution column across all rows.<br>Image A subtotal = sum of image A's contribution products: ${formatNumber(trace.image_subtotals[0])}. This is not a sum of image pixels.<br>Image B subtotal = ${formatNumber(trace.image_subtotals[1])}. Therefore ${formatNumber(trace.image_subtotals[0])} + ${formatNumber(trace.image_subtotals[1])} = gradient ${formatNumber(trace.sum_of_contributions)}<br>PyTorch gradient ${formatNumber(trace.autograd_gradient)}; matches: ${trace.matches_autograd}</p><div class="table-scroll"><table><thead><tr><th title="A or B identifies the training image">Image</th><th title="A location in the 3x3 convolution response map, not a weight index">Map position</th><th title="The pixel paired with the selected kernel weight at this map location">Aligned input</th><th title="Derivative of the loss with respect to this convolution response">dL/dz</th><th title="dL/dz multiplied by aligned input">Gradient contribution</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function renderClassifierGradient(update, parsed) {
    const classIndex = parsed.index[0];
    const featureIndex = parsed.name === "classifier.weight" ? parsed.index[1] : null;
    const rows = update.dL_d_logits.map((error, imageIndex) => {
      const input = featureIndex === null ? 1 : update.features_before[imageIndex][featureIndex];
      const product = error[classIndex] * input;
      return `<tr><td>${imageIndex === 0 ? "A" : "B"}</td><td>${formatNumber(error[classIndex])}</td><td>${formatNumber(input)}</td><td>${formatNumber(product)}</td></tr>`;
    }).join("");
    const label = featureIndex === null ? `bias[${classIndex}]` : `W[${classIndex},${featureIndex}]`;
    $("gradient-source").innerHTML = `<p class="formula-note">For ${label}, dL/dw = sum(dL/dlogit x input). The error already includes the mean batch division.</p><div class="table-scroll"><table><thead><tr><th>Image</th><th>dL/dlogit</th><th>Input</th><th>Product</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function renderInventory(update) {
    const entries = Object.entries(update.parameters);
    const rows = entries.map(([name, parameter]) => `<tr><td>${name}</td><td>[${parameter.shape.join(", ")}]</td><td>${parameter.changed_count}</td><td>${flatten(parameter.gradient).filter((value) => value === 0).length}</td></tr>`).join("");
    $("parameter-inventory").innerHTML = `<div class="table-scroll"><table class="inventory-table"><thead><tr><th>Tensor</th><th>Shape</th><th>Changed scalars</th><th>Zero gradients</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    renderUpdateOverview(update, entries);
  }

  function renderUpdateOverview(update, entries) {
    const selected = parseParameterKey(state.selectedParameter);
    const selectedParameter = update.parameters[selected.name];
    const selectedBefore = parameterValue(update, selected.name, selected.index, "before");
    const selectedGradient = parameterValue(update, selected.name, selected.index, "gradient");
    const selectedDelta = parameterValue(update, selected.name, selected.index, "delta");
    const selectedAfter = parameterValue(update, selected.name, selected.index, "after");
    const selectedName = `${selected.name}[${selected.index.join(",")}]`;
    const totalScalars = entries.reduce((sum, [, parameter]) => sum + flatten(parameter.before).length, 0);
    const changedScalars = entries.reduce((sum, [, parameter]) => sum + parameter.changed_count, 0);
    const groups = [
      ["features.0.weight", "convolution weights", "108 scalars", "one gradient per kernel weight"],
      ["features.0.bias", "convolution biases", "4 scalars", "one gradient per filter"],
      ["classifier.weight", "classifier weights", "16 scalars", "one gradient per class connection"],
      ["classifier.bias", "classifier biases", "4 scalars", "one gradient per class"],
    ];
    const groupCards = groups.map(([name, label, count, detail]) => {
      const parameter = update.parameters[name];
      const zeroGradients = flatten(parameter.gradient).filter((value) => value === 0).length;
      const selectedGroup = name === selected.name;
      return `<div class="update-group ${selectedGroup ? "update-group-selected" : ""}"><strong>${label}</strong><span>${count}</span><small>${parameter.changed_count} changed; ${zeroGradients} zero gradients</small><em>${detail}</em></div>`;
    }).join("");
    const selectedRow = selectedParameter
      ? `<div class="update-scalar-grid"><div class="update-scalar-cell update-scalar-selected"><span>selected scalar</span><code>${selectedName}</code></div><div class="update-scalar-cell"><span>old</span><b>${formatNumber(selectedBefore)}</b></div><div class="update-scalar-cell"><span>gradient</span><b class="update-gradient-value">${formatNumber(selectedGradient)}</b></div><div class="update-scalar-cell"><span>delta</span><b class="update-delta-value">${formatNumber(selectedDelta)}</b></div><div class="update-scalar-cell"><span>new</span><b>${formatNumber(selectedAfter)}</b></div></div>`
      : "";
    const filterMatrix = selected.name === "features.0.weight" ? renderFilterUpdateMatrix(update, selected.index[0]) : "";
    $("update-overview").innerHTML = `<div class="update-overview-title">Whole model update: every stored scalar gets its own gradient</div><div class="model-update-flow"><div class="model-update-node"><strong>batch loss L</strong><small>one loss for this batch</small></div><span class="model-update-arrow">&darr;</span><div class="model-update-node model-update-node-focus"><strong>backward</strong><small>compute dL/dparameter</small></div><span class="model-update-arrow">&darr;</span><div class="update-groups">${groupCards}</div><span class="model-update-arrow">&darr;</span><div class="model-update-node model-update-node-sgd"><strong>SGD</strong><small>${changedScalars} / ${totalScalars} changed</small><code>new = old - lr x gradient</code></div></div>${selectedRow}${filterMatrix}<p class="update-overview-note">The selected scalar is one member of the highlighted group. Its delta changes only that scalar; the other ${totalScalars - 1} scalars use their own gradients. After all updates, the changed weights affect the next forward pass and loss.</p>`;
  }

  function renderFilterUpdateMatrix(update, filterIndex) {
    const parameter = update.parameters["features.0.weight"];
    const slices = parameter.before[filterIndex].map((channel, channelIndex) => {
      const channelGradient = flatten(parameter.gradient[filterIndex][channelIndex]).reduce((sum, value) => sum + Number(value), 0);
      const channelDelta = flatten(parameter.delta[filterIndex][channelIndex]).reduce((sum, value) => sum + Number(value), 0);
      const cells = channel.map((row, rowIndex) => `<tr>${row.map((before, columnIndex) => {
        const index = [filterIndex, channelIndex, rowIndex, columnIndex];
        const after = parameterValue(update, "features.0.weight", index, "after");
        const delta = parameterValue(update, "features.0.weight", index, "delta");
        return `<td><span>${formatNumber(before, 4)}</span><span class="filter-after">&rarr; ${formatNumber(after, 4)}</span><small>&Delta; ${formatNumber(delta, 4)}</small></td>`;
      }).join("")}</tr>`).join("");
      return `<div class="filter-slice"><strong>${channelNames[channelIndex]} channel <small>9 weights</small></strong><div class="filter-channel-total"><span>&Sigma; gradient ${formatNumber(channelGradient, 4)}</span><span>&Sigma; &Delta; ${formatNumber(channelDelta, 4)}</span></div><table><tbody>${cells}</tbody></table></div>`;
    }).join("");
    return `<div class="filter-update-matrix"><div class="filter-update-title">Full update for filter ${filterIndex}: all 3 x 3 x 3 convolution weights</div><p>Each slice is one input channel. Every cell is a separate weight. Cell format: old, then &rarr; new, then &Delta;. The channel totals sum the nine gradients or nine deltas; they are summaries, not convolution outputs.</p><div class="filter-slices">${slices}</div></div>`;
  }

  function probabilityMarkup(values) {
    return values.map((value, index) => `<div class="probability-bar"><span>${labels[index]}</span><div><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div><output>${formatPlain(value)}</output></div>`).join("");
  }

  function renderMetrics(update) {
    const changed = Object.values(update.parameters).reduce((sum, parameter) => sum + parameter.changed_count, 0);
    $("metric-cards").innerHTML = [
      ["Loss before step", formatPlain(update.loss_before)],
      ["Same batch after step", formatPlain(update.loss_after_same_batch)],
      ["Changed parameters", `${changed} / ${state.trace.parameter_count}`],
      ["Weights unchanged after backward", update.weights_unchanged_after_backward ? "true" : "false"],
    ].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
    const lossItems = update.labels.map((target, imageIndex) => {
      const targetProbability = update.probabilities_before[imageIndex][target];
      const sampleLoss = -Math.log(Math.max(targetProbability, Number.MIN_VALUE));
      return `<div class="loss-item"><strong>Image ${String.fromCharCode(65 + imageIndex)}</strong><span>target: ${labels[target]} (class ${target})</span><span>p[target] = ${formatPlain(targetProbability)}</span><code>L = -ln(${formatPlain(targetProbability)}) = ${formatPlain(sampleLoss)}</code></div>`;
    });
    const sampleLosses = update.labels.map((target, imageIndex) => -Math.log(Math.max(update.probabilities_before[imageIndex][target], Number.MIN_VALUE)));
    const lossSum = sampleLosses.reduce((sum, value) => sum + value, 0);
    const batchSize = update.labels.length;
    $("loss-walkthrough").innerHTML = `<div class="loss-title">Cross-entropy loss before this optimizer step</div><div class="loss-items">${lossItems.join("")}</div><div class="loss-equation">batch loss = (${sampleLosses.map((value) => formatPlain(value)).join(" + ")}) / ${batchSize} = ${formatPlain(lossSum)} / ${batchSize} = <strong>${formatPlain(update.loss_before)}</strong></div><div class="loss-gradient-link">This single batch loss feeds backward. For each image: <code>dL/dz = (p - one_hot(target)) / ${batchSize}</code>. Those logit gradients then flow into classifier, pooling, ReLU, and convolution gradients.</div>`;
    $("probabilities-before").innerHTML = probabilityMarkup(update.probabilities_before[state.imageIndex]);
    $("probabilities-after").innerHTML = probabilityMarkup(update.probabilities_after[state.imageIndex]);
  }

  function render() {
    if (!state.trace) return;
    const update = currentUpdate();
    $("update-slider").max = String(state.trace.updates.length - 1);
    $("update-slider").value = String(state.updateIndex);
    $("update-label").textContent = `${state.updateIndex + 1} / ${state.trace.updates.length}`;
    $("state-label").textContent = `Update ${update.update}: forward -> backward -> SGD`;
    $("state-detail").textContent = `Epoch ${update.epoch}, batch ${update.batch_in_epoch}; ${update.image_ids.length} images`;
    $("learning-rate").textContent = formatPlain(state.trace.learning_rate);
    $("parameter-count").textContent = String(state.trace.parameter_count);
    populateControls();
    renderPlaneSection(update);
    renderNetwork(update);
    renderParameterPanel(update);
    renderMetrics(update);
  }

  function stopPlayback() {
    state.playing = false;
    if (state.timer) window.clearInterval(state.timer);
    state.timer = null;
    $("play-toggle").textContent = "Play";
  }

  function startPlayback() {
    stopPlayback();
    state.playing = true;
    $("play-toggle").textContent = "Pause";
    state.timer = window.setInterval(() => {
      if (state.updateIndex >= state.trace.updates.length - 1) { stopPlayback(); return; }
      state.updateIndex += 1;
      render();
    }, Number($("speed-select").value));
  }

  function loadTrace(trace, sourceName) {
    stopPlayback();
    state.trace = validateTrace(trace);
    state.updateIndex = 0; state.imageIndex = 0; state.filterIndex = 0; state.classIndex = 0; state.outputRow = 1; state.outputColumn = 1; state.selectedParameter = "features.0.weight|0,0,1,1";
    $("trace-source").textContent = sourceName;
    setStatus(`Trace loaded: ${state.trace.updates.length} updates`, false);
    render();
  }

  function rememberReplay() {
    if (!state.trace) return;
    stopPlayback();
    try {
      sessionStorage.setItem(returnKey, JSON.stringify({
        trace: state.trace,
        sourceName: $("trace-source").textContent,
        selection: Object.fromEntries(selectionKeys.map((key) => [key, state[key]])),
        scrollY: window.scrollY,
      }));
    } catch (_) {
      // Storage may be disabled or full. Do not offer a stale return snapshot.
      try { sessionStorage.removeItem(returnKey); } catch (_) { /* Storage unavailable. */ }
    }
  }

  function restoreReplay() {
    if (new URLSearchParams(location.search).get("resume") !== "1") return false;
    const raw = sessionStorage.getItem(returnKey);
    if (!raw || raw.length > maxTraceBytes * 2) throw new Error("Saved replay unavailable; the sample trace is shown.");
    const saved = JSON.parse(raw);
    const trace = validateTrace(saved.trace);
    const selected = saved.selection || {};
    const limits = { updateIndex: trace.updates.length, filterIndex: 4, channelIndex: 3, outputRow: 3, outputColumn: 3, classIndex: 4 };
    Object.entries(limits).forEach(([key, limit]) => {
      if (!Number.isInteger(selected[key]) || selected[key] < 0 || selected[key] >= limit) throw new Error("Saved replay selection is invalid; the sample trace is shown.");
    });
    const update = trace.updates[selected.updateIndex];
    if (!Number.isInteger(selected.imageIndex) || selected.imageIndex < 0 || selected.imageIndex >= update.images.length) throw new Error("Saved image selection is invalid.");
    const validParameter = Object.entries(update.parameters).some(([name, parameter]) =>
      indexTuples(parameter.shape).some((index) => parameterKey(name, index) === selected.selectedParameter));
    if (!validParameter) throw new Error("Saved parameter selection is invalid.");
    loadTrace(trace, typeof saved.sourceName === "string" ? saved.sourceName : "Restored trace");
    selectionKeys.forEach((key) => { state[key] = selected[key]; });
    render();
    if (Number.isFinite(saved.scrollY)) requestAnimationFrame(() => requestAnimationFrame(() => window.scrollTo(0, Math.max(0, saved.scrollY))));
    return true;
  }

  function bindControls() {
    $("previous-update").addEventListener("click", () => { stopPlayback(); state.updateIndex = Math.max(0, state.updateIndex - 1); render(); });
    $("next-update").addEventListener("click", () => { stopPlayback(); state.updateIndex = Math.min(state.trace.updates.length - 1, state.updateIndex + 1); render(); });
    $("play-toggle").addEventListener("click", () => state.playing ? stopPlayback() : startPlayback());
    $("update-slider").addEventListener("input", (event) => { stopPlayback(); state.updateIndex = Number(event.target.value); render(); });
    $("speed-select").addEventListener("change", () => { if (state.playing) startPlayback(); });
    $("image-select").addEventListener("change", (event) => { state.imageIndex = Number(event.target.value); render(); });
    $("filter-select").addEventListener("change", (event) => { state.filterIndex = Number(event.target.value); render(); });
    $("row-select").addEventListener("change", (event) => { state.outputRow = Number(event.target.value); render(); });
    $("column-select").addEventListener("change", (event) => { state.outputColumn = Number(event.target.value); render(); });
    $("class-select").addEventListener("change", (event) => { state.classIndex = Number(event.target.value); render(); });
    $("parameter-select").addEventListener("change", (event) => { state.selectedParameter = event.target.value; render(); });
    $("trace-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (!file) return;
      if (file.size > maxTraceBytes) { setStatus("Trace rejected: file is over 2 MB", true); return; }
      const reader = new FileReader();
      reader.onload = () => {
        try { loadTrace(JSON.parse(reader.result), file.name); }
        catch (error) { setStatus(error.message, true); }
      };
      reader.onerror = () => setStatus("Could not read the selected trace", true);
      reader.readAsText(file);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    // Standalone file exports have no hosted classifier to return to.
    const hosted = ["http:", "https:"].includes(window.location.protocol);
    $("site-nav").hidden = !hosted;
    document.querySelectorAll(".concept-links").forEach((element) => { element.hidden = !hosted; });
    if (hosted) document.querySelectorAll('a[href^="/learn/concepts/"], a[href^="/learn/diagrams/"]').forEach((link) => link.addEventListener("click", rememberReplay));
    bindControls();
    try {
      loadTrace(window.COSMOSAI_SAMPLE_TRACE, "sample_trace.json");
    } catch (error) {
      setStatus(error.message, true);
      $("trace-source").textContent = "No valid sample trace";
    }
    if (hosted) {
      try { restoreReplay(); }
      catch (_) { loadTrace(window.COSMOSAI_SAMPLE_TRACE, "sample_trace.json"); setStatus("Saved replay unavailable; the sample trace is shown.", true); }
    }
  });
}());
