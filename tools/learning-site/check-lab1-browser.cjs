/* Local study preview check: COSMOSAI_LEARNING_TOOLS=/path/to/node/tools node tools/learning-site/check-lab1-browser.cjs */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { createRequire } = require('node:module');
const requireTools = createRequire(path.join(process.env.COSMOSAI_LEARNING_TOOLS || __dirname, 'package.json'));
const { chromium } = requireTools('playwright');
const base = process.argv[2] || 'http://127.0.0.1:8080';
const root = path.resolve(__dirname, '../..');
(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.COSMOSAI_CHROME ? {executablePath: process.env.COSMOSAI_CHROME} : {}) });
  const errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({viewport: {width, height: 1000}});
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(base + '/learn/courses/AIA/');
      assert.equal(await page.locator('#AIA-LAB01 li a').count(), 9);
      for (const name of ['L01_MATLAB_basics_and_data_preprocessing.pdf','Lab1 report.pdf','Lab1 script.m']) {
        const result = await page.request.get(base+'/learn/courses/materials/AIA/'+encodeURIComponent(name));
        assert.equal(result.status(), 200);
        const source = name.startsWith('L01_') ? 'docs/courses/AIA/labs/L01/' + name : 'docs/courses/AIA/labs/L01/submission/'+name;
        const hash = bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
        assert.equal(hash(await result.body()), hash(fs.readFileSync(path.join(root,source))));
      }
      await page.goto(base+'/learn/concepts/?course=AIA#aia-lab1-normalization');
      assert(await page.locator('#aia-lab1-normalization').isVisible());
      const figureLinks = page.locator('article[data-topic^="aia-lab1-"] img');
      assert.equal(await figureLinks.count(),7);
      for (const img of await figureLinks.all()) {
        const anchor=await img.evaluate(el=>el.closest('article').dataset.topic);
        await page.evaluate(anchor=>{location.hash=anchor;},anchor);
        await img.scrollIntoViewIfNeeded();
        await img.evaluate(el=>el.decode());
        assert(await img.evaluate(el=>el.naturalWidth>0));
      }
      await page.goto(base+'/learn/examples/lab1-preprocessing/');
      assert.match(await page.locator('#gap-result').innerText(), /After: \[2, 3, 4, 5\]/);
      assert.match(await page.locator('#norm-result').innerText(), /Normalised reading = \(5 − 2\) \/ 1 = 3/);
      await page.locator('[name=reading]').fill('-1');
      await page.getByRole('button',{name:'Show the calculation'}).click();
      assert.match(await page.locator('#norm-result').innerText(), /\(-1 − 2\) \/ 1 = -3/);
      await page.locator('[name=training]').fill('2,2,2');
      await page.getByRole('button',{name:'Show the calculation'}).click();
      assert.match(await page.locator('#norm-result').innerText(), /nonzero spread/);
      await page.locator('[name=training]').fill('1,2,3');
      await page.getByRole('button',{name:'Show the calculation'}).click();
      await page.screenshot({path:`/tmp/cosmosai-lab1-practice-${width}.png`,fullPage:true});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.close();
      console.log(`${width}px: lab lesson, figures, PDF/script checksums, interactive arithmetic passed`);
    }
    assert.deepEqual(errors,[]);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
