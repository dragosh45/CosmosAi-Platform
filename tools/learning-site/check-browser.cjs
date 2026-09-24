/* Run against a started gateway: node tools/learning-site/check-browser.cjs [URL]. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const requireTools = createRequire(path.join(process.env.COSMOSAI_LEARNING_TOOLS || __dirname, 'package.json'));
const {chromium} = requireTools('playwright');
const base = process.argv[2] || 'http://127.0.0.1:8080';
const root = path.resolve(__dirname, '../..');

(async () => {
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [], external = [], modelCalls = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', request => {
      if (!request.url().startsWith(base) && /^https?:/.test(request.url())) external.push(request.url());
      if (/\/(model|classify\/image)$/.test(request.url())) modelCalls.push(request.url());
    });
    await page.goto(base + '/learn/');
    const custom = await page.evaluate(() => { const trace = structuredClone(window.COSMOSAI_SAMPLE_TRACE); trace.learning_rate = 0.123; return trace; });
    await page.locator('#trace-file').setInputFiles({name: 'custom-example.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(custom))});
    await page.locator('#next-update').click();
    await page.locator('#filter-select').selectOption('2');
    await page.locator('#image-select').selectOption('1');
    await page.locator('#parameter-select').selectOption('classifier.weight|1,2');
    await page.locator('.concept-links a[href$="#concept-backpropagation"]').click();
    await page.waitForURL('**/concepts/#concept-backpropagation');
    assert.equal(await page.locator('#concept-backpropagation').isVisible(), true);
    await page.locator('#concept-search').fill('gradient');
    assert.match(await page.locator('#search-status').innerText(), /matching sections/);
    await page.locator('#concept-search').fill('does-not-exist-990123');
    assert.equal(await page.locator('#search-status').innerText(), '0 matching sections');
    await page.locator('.return-replay').first().click();
    await page.waitForURL('**/learn/?resume=1');
    assert.equal(await page.locator('#trace-source').innerText(), 'custom-example.json');
    assert.equal(await page.locator('#learning-rate').innerText(), '0.123000');
    assert.equal(await page.locator('#filter-select').inputValue(), '2');
    assert.equal(await page.locator('#image-select').inputValue(), '1');
    assert.equal(await page.locator('#update-slider').inputValue(), '1');
    assert.equal(await page.locator('#parameter-select').inputValue(), 'classifier.weight|1,2');
    await page.goto(base + '/learn/diagrams/#galaxy_gradient_backprop_weight_update_flow');
    await page.waitForFunction(() => document.querySelector('#diagram-object').contentDocument?.querySelector('svg text'));
    assert.match(await page.locator('#diagram-title').innerText(), /gradient|weight/i);
    const previousWidth = await page.locator('#diagram-object').evaluate(el => el.clientWidth);
    await page.locator('#zoom-in').click();
    assert.ok(await page.locator('#diagram-object').evaluate(el => el.clientWidth) > previousWidth);
    await page.locator('#fit-diagram').click();
    const catalog = await (await page.request.get(base + '/learn/diagrams/catalog.json')).json();
    // Rasterize each self-contained SVG and count non-white pixels, not just HTTP 200.
    for (const item of catalog) {
      const url = base + '/learn/diagrams/svg/' + item.id + '.svg';
      const pixels = await page.evaluate(async url => {
        const svg = await (await fetch(url)).text();
        const blob = URL.createObjectURL(new Blob([svg], {type: 'image/svg+xml'}));
        try {
          const img = new Image(); img.src = blob; await img.decode();
          const canvas = document.createElement('canvas'); canvas.width = 600; canvas.height = Math.round(600 * img.height / img.width);
          const ctx = canvas.getContext('2d'); ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
          let marked = 0;
          for (let i = 0; i < data.length; i += 4) if (data[i+3] > 100 && Math.min(data[i], data[i+1], data[i+2]) < 220) marked++;
          return marked;
        } finally { URL.revokeObjectURL(blob); }
      }, url);
      assert.ok(pixels > 1000, `${item.id} is blank: ${pixels}`);
    }
    assert.equal(modelCalls.length, 0, 'Learning pages must not run inference');
    const screenshotDir = process.env.COSMOSAI_SCREENSHOTS || '/tmp/cosmosai-learning-check';
    fs.mkdirSync(screenshotDir, {recursive: true});
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({width, height: 1000});
      for (const [name, url] of [['classifier', '/'], ['replay', '/learn/'], ['concepts', '/learn/concepts/#concept-backpropagation'], ['diagrams', '/learn/diagrams/#galaxy_gradient_backprop_weight_update_flow']]) {
        await page.goto(base + url);
        if (name === 'diagrams') await page.waitForFunction(() => document.querySelector('#diagram-object').contentDocument?.querySelector('svg'));
        const size = await page.evaluate(() => ({width: innerWidth, scroll: document.documentElement.scrollWidth}));
        assert.ok(size.scroll <= size.width + 1, `${name} overflows at ${width}: ${JSON.stringify(size)}`);
        await page.screenshot({path: path.join(screenshotDir, `${name}-${width}.png`)});
      }
    }
    await page.goto(base + '/learn/?resume=1');
    await page.evaluate(() => sessionStorage.setItem('cosmosai-replay-return', '{invalid'));
    await page.reload();
    assert.match(await page.locator('#trace-status').innerText(), /Saved replay unavailable/);
    await page.goto('file://' + root + '/apps/learning-replay/index.html');
    assert.equal(await page.locator('#site-nav').isVisible(), false);
    assert.equal(await page.locator('.concept-links').first().isVisible(), false);
    assert.match(await page.locator('#trace-status').innerText(), /Trace loaded/);
    assert.deepEqual(external, []);
    assert.deepEqual(errors, []);
    console.log(`PASS: replay return (custom trace), concepts search, ${catalog.length} SVG pixel checks, four pages at four widths, standalone replay. Screenshots: ${screenshotDir}`);
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
