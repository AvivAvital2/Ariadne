/// <reference types="cypress" />

const hosted = {id:'configured', model:'gpt-6-astra', provider:'openai',
  embedding_model:'text-embedding-3-large', label:'Configured OpenAI',
  available:true, preview_available:true};
const external = {id:'external-0123456789abcdef01234567',
  model:'bartowski/Qwen3.8-27B-GGUF', provider:'local',
  embedding_model:'Qwen/Qwen3-Embedding-8B-GGUF',
  label:'Qwen servers you manage', available:true, preview_available:true,
  qualification:'unqualified'};
const project = '0123456789abcdef01234567';

function estimate(choice) {
  const local = choice.provider === 'local';
  return {source:'demo', model:choice.model, provider:choice.provider,
    batch_supported:!local, max_concurrency:local?1:null,
    input_per_million:local?0:0.2, output_per_million:local?0:2.5,
    estimate_note:local?'Provider token charges: $0. Hardware/runtime cost: unestimated.':'',
    file_count:1, input_tokens:100, output_tokens:100, embedding_tokens:100,
    total_cost_usd:local?0:1, total_cost_batched_usd:local?0:0.5,
    embedding_cost_usd:0, languages:[], by_doc_type:[], by_directory:[]};
}

describe('User-managed model endpoints', () => {
  let base;
  beforeEach(() => {
    cy.task('fixedStaticUrl').then(url => {base = url;});
    cy.intercept('GET', '/api/onboarding-models', {models:[hosted], default:hosted.id});
    cy.intercept('POST', '/api/sources', {sources:[]});
    cy.intercept('POST', '/api/source_add', {source:'demo', created:true});
    cy.intercept('POST', '/api/discover', {source:'demo', file_count:1, languages:[], indexers:[]}).as('discover');
    cy.intercept('POST', '/api/estimate', req => {
      req.reply(estimate(req.body.onboarding_model === external.id ? external : hosted));
    }).as('estimate');
    cy.intercept('POST', '/api/local-models', () => {
      throw new Error('User-managed servers must not invoke the managed Mac runtime');
    });
  });

  function preview() {
    cy.visit(base + '/');
    cy.get('#srcName').type('demo');
    cy.get('#srcPath').type('/workspace/demo');
    cy.get('#nextBtn').click();
    cy.wait('@discover');
    cy.get('#nextBtn').click();
    cy.wait('@estimate');
    cy.get('#externalModels summary').click();
  }

  function enterEndpoints() {
    cy.get('#externalChatUrl').type('https://chat.example.test/v1');
    cy.get('#externalEmbeddingUrl').type('https://embed.example.test/v1');
  }

  it('selects Gemma generation with Qwen embeddings and saves the tested pair', () => {
    const gemma = {...external, model:'unsloth/gemma-4-12b-it-GGUF', user_managed:true,
      label:'Gemma 4 12B + Qwen 8B embeddings'};
    cy.intercept('POST', '/api/estimate', req => {
      req.reply(estimate(req.body.onboarding_model === gemma.id ? gemma : hosted));
    }).as('estimate');
    cy.intercept('POST', '/api/external-models', req => {
      expect(req.body).to.deep.equal({
        model_profile:'gemma4-12b-qwen8b-mac-q4-v1',
        chat_url:'https://chat.example.test/v1',
        embedding_url:'https://embed.example.test/v1',
      });
      req.reply({delay:300, statusCode:201, body:{profile_id:gemma.id, models:[hosted, gemma],
        checks:{chat:{completed:true}, embeddings:{dimensions:3072}}, qualification:'unqualified'}});
    }).as('gemmaCheck');
    preview();
    cy.get('#externalModelProfile').select('gemma4-12b-qwen8b-mac-q4-v1');
    enterEndpoints();
    cy.get('#externalModels button[type="submit"]').click();
    cy.get('#externalModelProfile').should('be.disabled');
    cy.wait('@gemmaCheck');
    cy.wait('@estimate').its('request.body.onboarding_model').should('eq',gemma.id);
    cy.get('#externalModelProfile').should('be.enabled').and('have.value','gemma4-12b-qwen8b-mac-q4-v1');
    cy.get('#modeldd .ddbtn').should('contain','Gemma');
    cy.get('#externalModelStatus').should('contain','profile saved');
  });

  it('checks only on request, selects the saved profile, and builds with that profile', () => {
    let checks = 0;
    cy.intercept('POST', '/api/external-models', req => {
      checks++;
      expect(req.body).to.deep.equal({
        chat_url:'https://chat.example.test/v1',
        embedding_url:'https://embed.example.test/v1',
        chat_api_key_env:'QWEN_CHAT_KEY', embedding_api_key_env:'QWEN_EMBED_KEY',
      });
      req.reply({statusCode:201, body:{profile_id:external.id, models:[hosted, external],
        checks:{chat:{complete:true}, embeddings:{dimensions:3072}}, qualification:'unqualified'}});
    }).as('check');
    cy.intercept('POST', '/api/onboard', {job_id:'external-build', project_id:project}).as('build');
    cy.intercept('GET', '/api/onboard/events*', {headers:{'content-type':'text/event-stream'}, body:
      'data: {"type":"done","result":{"docs_written":1,"themes_found":0,"coverage_percent":100,"files_indexed":1}}\n\n'});
    preview();
    cy.then(() => {expect(checks).to.eq(0);});
    cy.get('#externalModels').should('contain','Qwen3.8-27B').and('contain','Qwen3-Embedding-8B');
    cy.get('#externalModels').should('contain','short generation').and('contain','embedding');
    enterEndpoints();
    cy.get('#externalChatKeyEnv').type('QWEN_CHAT_KEY');
    cy.get('#externalEmbeddingKeyEnv').type('QWEN_EMBED_KEY');
    cy.get('#externalModels button[type="submit"]').click();
    cy.wait('@check');
    cy.wait('@estimate').its('request.body.onboarding_model').should('eq',external.id);
    cy.get('#externalModelStatus').should('contain','saved').and('contain','accuracy');
    cy.get('#ddnm').should('have.text',external.model);
    cy.get('#modeldd .ddbtn').click();
    cy.get('#modelSeg [data-model="configured"]').should('exist');
    cy.get('#modelSeg [data-model="' + external.id + '"]').click();
    cy.wait('@estimate');
    cy.get('#nextBtn').click();
    cy.get('#externalModels').should('be.visible');
    cy.get('#nextBtn').click();
    cy.wait('@estimate');
    cy.get('#nextBtn').click();
    cy.wait('@build').then(({request}) => {
      expect(request.body.onboarding_model).to.eq(external.id);
      expect(request.body).not.to.have.property('model');
      expect(request.body).not.to.have.property('chat_url');
    });
    cy.then(() => {expect(checks).to.eq(1);});
  });

  it('converts the minimum timeout to seconds and locks settings until one check finishes', () => {
    let checks = 0;
    cy.intercept('POST', '/api/external-models', req => {
      checks++;
      expect(req.body).to.deep.equal({chat_url:'https://chat.example.test/v1',
        embedding_url:'https://embed.example.test/v1', request_timeout_seconds:600});
      req.reply({delay:1800, statusCode:201, body:{profile_id:external.id,
        models:[hosted, external], checks:{}, qualification:'unqualified'}});
    }).as('check');
    preview(); enterEndpoints();
    cy.get('#externalRequestTimeoutMinutes').type('10');
    cy.get('#externalModels button[type="submit"]').click().should('be.disabled');
    cy.get('#externalModels input').each(input => {cy.wrap(input).should('be.disabled');});
    cy.get('#nextBtn').should('be.disabled');
    cy.get('#externalModels form').trigger('submit');
    cy.get('#externalModelStatus').should('contain','Testing');
    cy.wait('@check'); cy.wait('@estimate');
    cy.get('#externalModels input').each(input => {cy.wrap(input).should('be.enabled');});
    cy.get('#externalModels button[type="submit"]').should('be.enabled');
    cy.get('#nextBtn').should('be.enabled');
    cy.then(() => {expect(checks).to.eq(1);});
  });

  it('rejects invalid timeout minutes and accepts the maximum without leaking the form field', () => {
    let checks = 0;
    cy.intercept('POST', '/api/external-models', req => {
      checks++;
      expect(req.body).to.deep.equal({chat_url:'https://chat.example.test/v1',
        embedding_url:'https://embed.example.test/v1', request_timeout_seconds:14400});
      req.reply({statusCode:201, body:{profile_id:external.id,
        models:[hosted, external], checks:{}, qualification:'unqualified'}});
    }).as('check');
    preview(); enterEndpoints();
    cy.get('#externalModelHelp').should('contain','two minutes').and('contain','CPU');
    ['9', '241', '10.5'].forEach(value => {
      cy.get('#externalRequestTimeoutMinutes').clear().type(value);
      cy.get('#externalModels button[type="submit"]').click();
      cy.get('#externalRequestTimeoutMinutes').then(input => {
        expect(input[0].validity.valid, value + ' minutes').to.eq(false);
      });
    });
    cy.then(() => {expect(checks).to.eq(0);});
    cy.get('#externalRequestTimeoutMinutes').clear().type('240');
    cy.get('#externalModels button[type="submit"]').click();
    cy.wait('@check'); cy.wait('@estimate');
    cy.get('#externalModelStatus').should('contain','saved');
    cy.then(() => {expect(checks).to.eq(1);});
  });

  it('rejects incomplete URLs and credential values before sending a request', () => {
    cy.intercept('POST', '/api/external-models', () => {
      throw new Error('Invalid form values must not start a connection check');
    });
    preview();
    cy.get('#externalModels button[type="submit"]').click();
    cy.get('#externalChatUrl').then(input => {expect(input[0].validity.valid).to.eq(false);});
    enterEndpoints();
    cy.get('#externalChatUrl').clear().type('https://chat.example.test/incompatible');
    cy.get('#externalModels button[type="submit"]').click();
    cy.get('#externalChatUrl').then(input => {expect(input[0].validity.valid).to.eq(false);});
    cy.get('#externalChatUrl').clear().type('https://chat.example.test/v1');
    cy.get('#externalChatKeyEnv').type('secret-with-dashes');
    cy.get('#externalModels button[type="submit"]').click();
    cy.get('#externalChatKeyEnv').then(input => {expect(input[0].validity.valid).to.eq(false);});
    cy.get('#ddnm').should('have.text',hosted.model);
  });

  it('keeps the selected pair on failure, renders errors as text, and permits retry', () => {
    let checks = 0;
    cy.intercept('POST', '/api/external-models', req => {
      checks++;
      req.reply(checks === 1 ? {statusCode:400, body:{error:'Generation failed <img src=x onerror=alert(1)>'}} :
        {statusCode:201, body:{profile_id:external.id, models:[hosted, external], checks:{}, qualification:'unqualified'}});
    }).as('check');
    preview(); enterEndpoints();
    cy.get('#externalModels button[type="submit"]').click(); cy.wait('@check');
    cy.get('#externalModelStatus').should('contain','Generation failed <img');
    cy.get('#externalModels img').should('not.exist');
    cy.get('#externalChatUrl').should('have.value','https://chat.example.test/v1');
    cy.get('#ddnm').should('have.text',hosted.model);
    cy.get('#nextBtn').should('be.enabled');
    cy.get('#externalModels button[type="submit"]').should('be.enabled').click();
    cy.wait('@check'); cy.wait('@estimate');
    cy.get('#externalModelStatus').should('contain','saved');
    cy.get('#ddnm').should('have.text',external.model);
  });

  it('does not change the selected profile for an incomplete success response', () => {
    cy.intercept('POST', '/api/external-models', {statusCode:201, body:{profile_id:external.id, models:[hosted]}}).as('check');
    preview(); enterEndpoints();
    cy.get('#externalModels button[type="submit"]').click(); cy.wait('@check');
    cy.get('#externalModelStatus').should('contain','did not return a saved model profile');
    cy.get('#ddnm').should('have.text',hosted.model);
  });

  it('marks edited settings as untested while retaining the saved selection', () => {
    let checks = 0;
    cy.intercept('POST', '/api/external-models', req => {
      checks++;
      req.reply({statusCode:201, body:{profile_id:external.id, models:[hosted, external], checks:{}, qualification:'unqualified'}});
    }).as('check');
    preview(); enterEndpoints();
    cy.get('#externalModels button[type="submit"]').click(); cy.wait('@check'); cy.wait('@estimate');
    cy.get('#externalModelStatus').should('contain','Connection checked');
    cy.get('#externalChatUrl').clear().type('https://replacement.example.test/v1');
    cy.get('#externalModelStatus').should('contain','not been tested or saved').and('not.contain','Connection checked');
    cy.get('#ddnm').should('have.text',external.model);
    cy.then(() => {expect(checks).to.eq(1);});
  });

  it('recovers controls and explains an invalid server response', () => {
    cy.intercept('POST', '/api/external-models', {statusCode:502,
      headers:{'content-type':'text/html'}, body:'<h1>Unavailable</h1>'}).as('check');
    preview(); enterEndpoints();
    cy.get('#externalModels button[type="submit"]').click(); cy.wait('@check');
    cy.get('#externalModelStatus').should('contain','valid connection-test result').and('contain','Try again');
    cy.get('#externalModels button[type="submit"]').should('be.enabled');
    cy.get('#nextBtn').should('be.enabled');
    cy.get('#ddnm').should('have.text',hosted.model);
  });

  it('visibly distinguishes deployments of the same pair, including a shared host', () => {
    const owned = [
      {...external, user_managed:true, id:'external-aaaaaaaa0000000000000001',
        label:'Experimental Qwen 27B Q4 + 8B embeddings — chat-a.example.test (user managed)'},
      {...external, user_managed:true, id:'external-bbbbbbbb0000000000000002',
        label:'Experimental Qwen 27B Q4 + 8B embeddings — chat-b.example.test (user managed)'},
      {...external, user_managed:true, id:'external-aaaaaaaa0000000000000003',
        label:'Experimental Qwen 27B Q4 + 8B embeddings — chat-a.example.test (user managed)'},
    ];
    cy.intercept('GET', '/api/onboarding-models', {models:[hosted, ...owned], default:owned[0].id});
    cy.intercept('POST', '/api/estimate', req => {
      req.reply(estimate(owned.find(choice => choice.id === req.body.onboarding_model) || hosted));
    }).as('ownedEstimate');
    cy.visit(base + '/');
    cy.get('#srcName').type('demo'); cy.get('#srcPath').type('/workspace/demo');
    cy.get('#nextBtn').click(); cy.wait('@discover');
    cy.get('#nextBtn').click(); cy.wait('@ownedEstimate');
    cy.get('#modeldd .ddbtn').click();
    const names = [];
    owned.forEach(choice => {
      cy.get('#modelSeg [data-model="' + choice.id + '"] .mn')
        .scrollIntoView().should('be.visible').and('contain',choice.label).invoke('text').then(text => names.push(text));
      cy.get('#modelSeg [data-model="' + choice.id + '"] .deployment').scrollIntoView().should('be.visible');
    });
    cy.then(() => {expect(new Set(names).size).to.eq(3);});
    cy.get('#modelSeg [data-model="' + owned[1].id + '"]').click();
    cy.wait('@ownedEstimate').its('request.body.onboarding_model').should('eq',owned[1].id);
    cy.then(() => {cy.get('#ddnm').should('have.text',names[1]);});
    cy.get('#modeldd').scrollIntoView();
    cy.get('#ddnm .deployment').should('be.visible');
    cy.get('#modeldd .ddbtn').click();
    cy.get('#modelSeg [data-model="' + owned[2].id + '"]').click();
    cy.wait('@ownedEstimate').its('request.body.onboarding_model').should('eq',owned[2].id);
    cy.then(() => {cy.get('#ddnm').should('have.text',names[2]).and('not.have.text',names[0]);});
    cy.get('#modeldd').scrollIntoView();
    cy.get('#ddnm .deployment').should('be.visible');
    cy.get('#nextBtn').click(); cy.get('#nextBtn').click(); cy.wait('@ownedEstimate');
    cy.then(() => {cy.get('#rvModel').should('contain',names[2]);});
  });
});
