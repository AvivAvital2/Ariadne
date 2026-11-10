/* Render answer text without interpreting model output as HTML. */
(function () {
  const esc = value => String(value ?? '').replace(/[&<>"]/g,
    c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  let renderer;
  let nextDiagram = 0;

  function loadRenderer() {
    if (!renderer) {
      renderer = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = '/static/vendor/mermaid-12.2.0/mermaid.min.js';
        const failed = () => {
          renderer = undefined;
          script.remove();
          reject(new Error('Diagram renderer unavailable'));
        };
        script.onload = () => {
          try {
            mermaid.initialize({startOnLoad: false, securityLevel: 'strict',
              suppressErrorRendering: true, htmlLabels: false,
              fontFamily: 'Arial, sans-serif', theme: 'default'});
            resolve(mermaid);
          } catch { failed(); }
        };
        script.onerror = failed;
        document.head.append(script);
      });
    }
    return renderer;
  }

  // Recognize fences before inline formatting so code stays literal. Fence
  // lengths and closing lines matter: embedded triple backticks are valid code.
  function blocks(text) {
    const lines = String(text ?? '').replace(/\r\n?/g, '\n').split('\n');
    const result = [];
    let prose = [];
    const flush = () => {
      if (prose.length) result.push({text: prose.join('\n')});
      prose = [];
    };
    for (let i = 0; i < lines.length; i++) {
      const opening = lines[i].match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
      if (!opening || (opening[1][0] === '`' && opening[2].includes('`'))) {
        prose.push(lines[i]);
        continue;
      }
      flush();
      const closing = new RegExp('^ {0,3}' + opening[1][0] + '{' + opening[1].length + ',}[ \\t]*$');
      const code = [];
      while (++i < lines.length && !closing.test(lines[i])) code.push(lines[i]);
      result.push({code: code.join('\n'), closed: i < lines.length,
        language: opening[2].trim().split(/\s+/)[0].toLowerCase()});
    }
    flush();
    return result;
  }

  function codeBlock(text) {
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.textContent = text;
    pre.append(code);
    return pre;
  }

  function showDiagramError({diagram, details}) {
    const message = document.createElement('p');
    message.className = 'diagram-error';
    message.textContent = 'Diagram could not be rendered. Its source is shown below.';
    diagram.replaceChildren(message);
    details.open = true;
  }

  window.renderAnswerContent = async function (container, text) {
    const diagrams = [];
    container.replaceChildren();
    for (const block of blocks(text)) {
      if (!Object.hasOwn(block, 'code')) {
        const prose = esc(block.text).split(/\n\s*\n/).filter(p => p.trim()).map(p =>
          '<p>' + p.replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>') + '</p>'
        ).join('');
        container.insertAdjacentHTML('beforeend', prose);
      } else if (block.language !== 'mermaid' || !block.closed) {
        container.append(codeBlock(block.code));
      } else {
        const figure = document.createElement('figure');
        figure.className = 'answer-diagram-block';
        const diagram = document.createElement('div');
        diagram.className = 'answer-diagram';
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Diagram source';
        details.append(summary, codeBlock(block.code));
        figure.append(diagram, details);
        container.append(figure);
        diagrams.push({diagram, details, source: block.code});
      }
    }
    if (!diagrams.length) return;
    let api;
    try { api = await loadRenderer(); }
    catch { diagrams.forEach(showDiagramError); return; }
    for (const block of diagrams) {
      try {
        // Model output cannot configure the renderer. Mermaid's newer shape
        // metadata (@{...}) can fetch image URLs before SVG sanitization, even
        // in strict mode, so leave those blocks as source too.
        if (/%%\s*\{|^\s*---(?:\n|$)|@\s*\{/.test(block.source)) {
          showDiagramError(block);
          continue;
        }
        const {svg} = await api.render('answer-diagram-' + (++nextDiagram), block.source);
        // Mermaid sanitizes the SVG in strict mode; raw model text is never HTML.
        if (block.diagram.isConnected) {
          block.diagram.innerHTML = svg;
          const drawing = block.diagram.querySelector('svg');
          // Keep labels at their native size; the container scrolls on small
          // screens instead of shrinking a long diagram into unreadable text.
          if (drawing?.viewBox.baseVal.width) {
            drawing.style.minWidth = drawing.viewBox.baseVal.width + 'px';
          }
        }
      } catch { showDiagramError(block); }
    }
  };
})();
