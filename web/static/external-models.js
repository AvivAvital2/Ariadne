/* User-managed model servers are checked only after an explicit form submit. */
(function () {
  'use strict';
  window.createExternalModels = function (container, onSaved, onBusy = () => {}) {
    container.classList.add('external-models');
    container.innerHTML = `
      <details>
        <summary>Connect servers you manage</summary>
        <p>Choose Gemma 4 12B or Qwen3.8-27B Q4 generation with Qwen3-Embedding-8B Q4 embeddings. You manage model installation and server capacity.</p>
        <form>
          <div class="external-model-fields">
            <label for="externalModelProfile">Model pair
              <select id="externalModelProfile" name="model_profile">
                <option value="qwen-mac-27b-q4-thinking-v1">Qwen 27B + Qwen 8B embeddings</option>
                <option value="gemma4-12b-qwen8b-mac-q4-v1">Gemma 4 12B + Qwen 8B embeddings</option>
              </select>
            </label>
            <label for="externalChatUrl">Generation server URL
              <input id="externalChatUrl" name="chat_url" type="url" required placeholder="http://localhost:18080/v1" autocomplete="off" spellcheck="false">
            </label>
            <label for="externalEmbeddingUrl">Embedding server URL
              <input id="externalEmbeddingUrl" name="embedding_url" type="url" required placeholder="http://localhost:18081/v1" autocomplete="off" spellcheck="false">
            </label>
            <label for="externalChatKeyEnv">Generation credential variable (optional)
              <input id="externalChatKeyEnv" name="chat_api_key_env" pattern="[A-Za-z_][A-Za-z0-9_]*" title="Enter an environment variable name, such as QWEN_CHAT_KEY" placeholder="QWEN_CHAT_KEY" autocomplete="off" spellcheck="false">
            </label>
            <label for="externalEmbeddingKeyEnv">Embedding credential variable (optional)
              <input id="externalEmbeddingKeyEnv" name="embedding_api_key_env" pattern="[A-Za-z_][A-Za-z0-9_]*" title="Enter an environment variable name, such as QWEN_EMBED_KEY" placeholder="QWEN_EMBED_KEY" autocomplete="off" spellcheck="false">
            </label>
            <label for="externalRequestTimeoutMinutes">Request timeout in minutes (optional)
              <input id="externalRequestTimeoutMinutes" name="request_timeout_minutes" type="number" min="10" max="240" step="1" placeholder="10–240" aria-describedby="externalModelHelp">
            </label>
          </div>
          <p>Use URLs ending in <code>/v1</code>. Credential fields take names of environment variables set on the Ariadne server; do not enter secret keys.</p>
          <p id="externalModelHelp">Test and save makes a short generation request and embedding requests, which may incur charges on your server. Leave the timeout blank to keep defaults, including a connection check of up to two minutes. CPU-only servers may need longer: set 10–240 whole minutes for server requests and the overall connection check. Existing projects keep their saved servers.</p>
          <button type="submit" class="btn ghost" aria-describedby="externalModelHelp">Test and save</button>
          <p id="externalModelStatus" role="status" aria-live="polite"></p>
        </form>
      </details>`;
    const form = container.querySelector('form');
    const button = form.querySelector('button');
    const status = container.querySelector('[role="status"]');
    const inputs = [...form.querySelectorAll('input, select')];
    const urls = inputs.filter(input => input.type === 'url');
    let busy = false;
    urls.forEach(input => input.addEventListener('input', () => input.setCustomValidity('')));
    form.addEventListener('input', () => {
      if (!busy && status.textContent) {
        status.classList.remove('error');
        status.textContent = 'Changed settings have not been tested or saved.';
      }
    });
    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (busy) return;
      for (const input of urls) {
        let valid = false;
        try {
          const url = new URL(input.value.trim());
          valid = ['http:', 'https:'].includes(url.protocol) && url.hostname &&
            !url.username && !url.password && !url.search && !url.hash &&
            url.pathname.replace(/\/+$/, '').endsWith('/v1');
        } catch (_) { /* The form reports the invalid field below. */ }
        input.setCustomValidity(valid ? '' : 'Enter an HTTP(S) server URL ending in /v1, without credentials, query or fragment.');
      }
      if (!form.reportValidity()) return;
      const body = Object.fromEntries(new FormData(form));
      for (const key of Object.keys(body)) {
        body[key] = body[key].trim();
        if (!body[key] && key.endsWith('_api_key_env')) delete body[key];
      }
      const timeoutMinutes = body.request_timeout_minutes;
      delete body.request_timeout_minutes;
      // Omitted selection retains the original Qwen request and saved identity.
      if (body.model_profile === 'qwen-mac-27b-q4-thinking-v1') delete body.model_profile;
      if (timeoutMinutes) body.request_timeout_seconds = Number(timeoutMinutes) * 60;
      busy = true;
      button.disabled = true;
      inputs.forEach(input => {input.disabled = true;});
      form.setAttribute('aria-busy', 'true');
      onBusy(true);
      status.classList.remove('error');
      status.textContent = 'Testing generation and embeddings…';
      try {
        const response = await fetch('/api/external-models', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
        });
        let result;
        try {
          result = await response.json();
        } catch (_) {
          throw new Error('The server did not return a valid connection-test result. Try again.');
        }
        if (!response.ok) throw new Error(result.error || 'Connection test failed.');
        if (!Array.isArray(result.models) || !result.models.some(model => model.id === result.profile_id)) {
          throw new Error('The server did not return a saved model profile.');
        }
        await onSaved(result);
        status.textContent = 'Connection checked and profile saved. Document accuracy and hardware fit have not been verified.';
      } catch (error) {
        status.classList.add('error');
        status.textContent = error.message || 'Connection test failed. Try again.';
      } finally {
        busy = false;
        button.disabled = false;
        inputs.forEach(input => {input.disabled = false;});
        form.setAttribute('aria-busy', 'false');
        onBusy(false);
      }
    });
  };
})();
