/* Browser navigation over generated content; no model calls or remote assets. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const normal = text => text.normalize('NFKD').toLowerCase();
  window.lucide?.createIcons();
  try {
    if (sessionStorage.getItem('cosmosai-replay-return')) {
      document.querySelectorAll('.return-replay').forEach(a => { a.hidden = false; });
    }
  } catch (_) { /* Navigation still works when browser storage is disabled. */ }

  if (document.body.dataset.page === 'concepts') {
    const articles = [...document.querySelectorAll('article[data-topic]')];
    const links = [...$('concept-topics').querySelectorAll('a')];
    const text = new Map(articles.map(a => [a.dataset.topic, normal(a.textContent)]));
    const courseFilter = $('course-filter');
    if (courseFilter) {
      const requested = new URLSearchParams(location.search).get('course');
      if ([...courseFilter.options].some(option => option.value === requested)) courseFilter.value = requested;
    }
    const courseTopics = () => courseFilter?.value
      ? JSON.parse(courseFilter.selectedOptions[0].dataset.topics) : null;
    const search = () => {
      const query = normal($('concept-search').value.trim());
      let count = 0;
      const allowed = courseTopics();
      links.forEach(link => {
        link.hidden = (allowed && !allowed.includes(link.dataset.topic)) ||
          (!!query && !normal(link.textContent).includes(query) && !text.get(link.dataset.topic).includes(query));
        if (!link.hidden) count++;
      });
      $('search-status').textContent = query ? `${count} matching sections` : `${articles.length} topics`;
    };
    const show = () => {
      let target;
      try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (_) {}
      let selected = target?.closest('article[data-topic]') || articles[0];
      const allowed = courseTopics();
      if (allowed && !allowed.includes(selected?.dataset.topic)) {
        selected = articles.find(article => allowed.includes(article.dataset.topic));
      }
      if ($('course-empty')) $('course-empty').hidden = !!selected;
      articles.forEach(article => { article.hidden = article !== selected; });
      const selectedAnchor = target?.closest('article[data-topic]') === selected ? target?.id : selected?.dataset.topic;
      links.forEach(link => {
        if (selected && link.hash.slice(1) === selectedAnchor) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
      if (target && selected?.contains(target) && location.hash) requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
      document.title = `${selected?.querySelector('h2').textContent || 'Course concepts'} | CosmosAI Concepts`;
    };
    $('concept-search').addEventListener('input', search);
    if (courseFilter) courseFilter.addEventListener('change', () => {
      const url = new URL(location.href);
      if (courseFilter.value) url.searchParams.set('course', courseFilter.value);
      else url.searchParams.delete('course');
      history.replaceState(null, '', url);
      search();
      const first = links.find(link => !link.hidden);
      if (first) location.hash = first.hash;
      show();
    });
    window.addEventListener('hashchange', show);
    search();
    if (!location.hash && courseFilter?.value) {
      const first = links.find(link => !link.hidden);
      if (first) history.replaceState(null, '', location.pathname + location.search + first.hash);
    }
    show();
  }

  if (document.body.dataset.page === 'diagrams') {
    let catalog, selected, zoom = 1;
    const viewport = $('diagram-viewport');
    const resize = value => {
      if (!selected) return;
      zoom = Math.max(.05, Math.min(2, value));
      $('diagram-object').style.width = `${Math.ceil(selected.width * zoom)}px`;
      $('diagram-object').style.height = `${Math.ceil(selected.height * zoom)}px`;
      $('diagram-zoom').value = String(Math.round(zoom * 100));
      $('zoom-value').textContent = `${Math.round(zoom * 100)}%`;
    };
    const fit = () => resize(Math.min(1, (viewport.clientWidth - 18) / selected.width));
    const show = () => {
      selected = catalog.find(d => '#' + d.id === location.hash) || catalog[0];
      $('diagram-select').value = selected.id;
      $('diagram-title').textContent = selected.title;
      $('diagram-status').textContent = 'Loading diagram';
      const url = `/learn/diagrams/svg/${selected.id}.svg`;
      $('diagram-object').data = url;
      $('diagram-object').setAttribute('aria-label', selected.title);
      $('download-diagram').href = url;
      $('diagram-source').href = `/learn/diagrams/scenes/${selected.id}.excalidraw`;
      $('diagram-concepts').replaceChildren(...selected.related.map(href => {
        const a = document.createElement('a'); a.href = href;
        a.textContent = href.split('#')[1].replace(/^concept-/, '').replaceAll('-', ' ');
        return a;
      }));
      viewport.scrollTo(0, 0);
      if (viewport.clientWidth < 700) resize(.75);
      else fit();
    };
    $('diagram-object').addEventListener('load', () => { $('diagram-status').textContent = 'Excalidraw illustration'; });
    $('diagram-object').addEventListener('error', () => { $('diagram-status').textContent = 'Diagram unavailable. Rebuild the learning exports.'; });
    $('diagram-select').addEventListener('change', event => { location.hash = event.target.value; });
    $('diagram-zoom').addEventListener('input', event => resize(Number(event.target.value) / 100));
    $('zoom-in').addEventListener('click', () => resize(zoom * 1.25));
    $('zoom-out').addEventListener('click', () => resize(zoom / 1.25));
    $('fit-diagram').addEventListener('click', fit);
    window.addEventListener('hashchange', () => { if (catalog) show(); });
    fetch('/learn/diagrams/catalog.json').then(response => {
      if (!response.ok) throw new Error('Diagram catalog unavailable'); return response.json();
    }).then(data => {
      if (!data.length || data.some(d => !(d.width > 0 && d.height > 0))) throw new Error('Diagram SVG exports have not been built');
      catalog = data; show();
    }).catch(error => { $('diagram-status').textContent = error.message; });
  }
})();
