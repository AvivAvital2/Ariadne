/// <reference types="cypress" />
// Compare pairs in Preview, commit on Start build, then retain the pair for Ask.
const choices = [
  {id:'gpt-6-astra', model:'gpt-6-astra', provider:'openai', embedding_model:'text-embedding-3-large'},
  {id:'claude-opus-5-5', model:'claude-opus-5-5', provider:'anthropic', embedding_model:'text-embedding-3-large'},
  {id:'qwen-hosted', model:'Qwen/Qwen3.8-27B', provider:'deepinfra', embedding_model:'Qwen/Qwen3-Embedding-8B'},
  {id:'qwen-local', model:'Qwen/Qwen3.8-27B', provider:'local', embedding_model:'Qwen/Qwen3-Embedding-8B'},
].map(m=>({...m, label:`${m.model} + ${m.embedding_model} (${m.provider})`, available:true, preview_available:true}));

function estimate(choice) {
  const local=choice.provider==='local';
  const cost={openai:2, anthropic:1, deepinfra:0.1, local:0}[choice.provider];
  return {
    source:'demo', model:choice.model, provider:choice.provider,
    batch_supported:['openai','anthropic'].includes(choice.provider), max_concurrency:local?1:null,
    estimate_note:local?'Provider token charges: $0. Hardware/runtime cost: unestimated.':'Hosted inference charges apply.',
    input_per_million:local?0:0.2, output_per_million:local?0:2.5,
    file_count:1, input_tokens:100, output_tokens:100, embedding_tokens:100,
    total_cost_usd:cost, total_cost_batched_usd:cost, embedding_cost_usd:local?0:0.001,
    languages:[], by_doc_type:[], by_directory:[],
    available_models:[{model:choice.model, input_per_million:local?0:0.2, output_per_million:local?0:2.5}],
  };
}

function selectPair(id) {
  cy.get('#modeldd .ddbtn').click();
  cy.get(`#modelSeg [data-model="${id}"]`).click();
  cy.wait('@estimate').then(({request})=>{
    expect(request.body.onboarding_model).to.eq(id);
    expect(request.body).not.to.have.property('project_id');
    expect(request.body).not.to.have.property('model');
  });
}

describe('Model pair selected per onboarding', () => {
  let base;
  beforeEach(()=>cy.task('fixedStaticUrl').then(u=>{base=u;}));

  choices.forEach(choice=>it(`keeps ${choice.id} with its project after onboarding`, ()=>{
    const project='0123456789abcdef01234567';
    const local=choice.provider==='local';
    cy.intercept('GET', '/api/onboarding-models', {models:choices, default:'gpt-6-astra'});
    cy.intercept('GET', '/api/projects', {projects:[{...choice, id:project, source:'demo'}]});
    cy.intercept('POST', '/api/sources', {default_source:'demo', sources:[{name:'demo'}]});
    cy.intercept('POST', '/api/source_add', {source:'demo', created:true}).as('connect');
    cy.intercept('POST', '/api/discover', {source:'demo', file_count:1, languages:[], indexers:[]}).as('discover');
    cy.intercept('POST', '/api/estimate', req=>req.reply(estimate(choices.find(m=>m.id===req.body.onboarding_model)))).as('estimate');
    cy.intercept('POST', '/api/onboard', {job_id:'onboarding-demo', status:'running', project_id:project}).as('onboard');
    cy.intercept('GET', '/api/onboard/events*', {headers:{'content-type':'text/event-stream'}, body:
      'data: {"type":"done","result":{"docs_written":1,"themes_found":0,"coverage_percent":100,"files_indexed":1}}\n\n'});
    cy.intercept('POST', '/api/ask', {answer:'Retry uses bounded backoff.', event_id:'evt', documents:[]}).as('ask');
    cy.intercept('POST', '/api/search', {documents:[]}).as('search');
    cy.visit(base+'/');
    cy.get('.step[data-step="1"]').should('not.contain','Models for this project');
    cy.get('#srcName').type('demo');
    cy.get('#srcPath').type('/workspace/demo');
    cy.get('#nextBtn').click();
    cy.wait('@connect').then(({request})=>{
      expect(request.body).not.to.have.property('onboarding_model');
      expect(request.body).not.to.have.property('project_id');
    });
    cy.wait('@discover').its('request.body').should('not.have.property','project_id');
    cy.get('#nextBtn').click();
    cy.wait('@estimate').its('request.body.onboarding_model').should('eq','gpt-6-astra');
    cy.get('#costV').should('contain','$2');
    selectPair('qwen-local');
    cy.get('#costV').should('contain','$0');
    selectPair('claude-opus-5-5');
    cy.get('#costV').should('contain','$1');
    cy.get('#concV').should('have.text','4');
    cy.get('#engBatch').should('be.visible');
    selectPair(choice.id);
    cy.get('#ddnm').should('have.text',choice.model);
    if(local) cy.get('#localCostNote').scrollIntoView().should('be.visible');
    if(['local','deepinfra'].includes(choice.provider)) cy.get('#engBatch').should('not.be.visible');
    cy.get('#nextBtn').click();
    cy.get('#nextBtn').click();
    cy.wait('@estimate'); // final scope/cost review before Start build
    cy.get('#nextBtn').click();
    cy.wait('@onboard').then(({request})=>{
      expect(request.body).not.to.have.property('project_id');
      expect(request.body).not.to.have.property('model');
      expect(request.body.onboarding_model).to.eq(choice.id);
    });
    cy.get('#gen-ready a').click();
    cy.location('search').should('eq','?project='+project);
    cy.get('#projectPicker').should('have.value',project);
    cy.reload();
    cy.get('#q').type('How are retries handled?{enter}');
    cy.wait('@ask').then(({request})=>{
      expect(request.body.project_id).to.eq(project);
      expect(request.body).not.to.have.property('model');
    });
    cy.wait('@search').its('request.body.project_id').should('eq',project);
  }));

  it('ignores a late paid estimate after switching to local Qwen', ()=>{
    cy.intercept('GET', '/api/onboarding-models', {models:choices, default:'gpt-6-astra'});
    cy.intercept('POST', '/api/sources', {sources:[]});
    cy.intercept('POST', '/api/source_add', {source:'demo', created:true});
    cy.intercept('POST', '/api/discover', {source:'demo', file_count:1, languages:[], indexers:[]}).as('discover');
    cy.intercept('POST', '/api/estimate', req=>{
      const choice=choices.find(m=>m.id===req.body.onboarding_model);
      req.alias=choice.id==='qwen-hosted'?'slowEstimate':'estimate';
      req.reply({delay:choice.id==='qwen-hosted'?1000:0, body:estimate(choice)});
    });
    cy.visit(base+'/');
    cy.get('#srcName').type('demo');
    cy.get('#srcPath').type('/workspace/demo');
    cy.get('#nextBtn').click();
    cy.wait('@discover');
    cy.get('#nextBtn').click();
    cy.wait('@estimate');
    cy.get('#modeldd .ddbtn').click();
    cy.get('#modelSeg [data-model="qwen-hosted"]').click();
    cy.get('#nextBtn').should('be.disabled');
    selectPair('qwen-local');
    cy.wait('@slowEstimate');
    cy.get('#modelPair').should('contain','local');
    cy.get('#costV').should('contain','$0');
    cy.get('#engBatch').should('not.be.visible');
    cy.get('#nextBtn').should('not.be.disabled');
  });

  it('previews a paid pair without credentials but prevents starting its build', ()=>{
    const choice={...choices[2], available:false, reason:'Missing DEEPINFRA_API_KEY'};
    cy.intercept('GET', '/api/onboarding-models', {models:[choice], default:choice.id});
    cy.intercept('POST', '/api/sources', {sources:[]});
    cy.intercept('POST', '/api/source_add', {source:'demo', created:true});
    cy.intercept('POST', '/api/discover', {source:'demo', file_count:1, languages:[], indexers:[]}).as('discover');
    cy.intercept('POST', '/api/estimate', estimate(choice)).as('estimate');
    cy.intercept('POST', '/api/onboard', ()=>{throw new Error('Must not submit a build without credentials');});
    cy.visit(base+'/');
    cy.get('#srcName').type('demo');
    cy.get('#srcPath').type('/workspace/demo');
    cy.get('#nextBtn').click();
    cy.wait('@discover');
    cy.get('#nextBtn').click();
    cy.wait('@estimate');
    cy.get('#costV').should('contain','$0.10');
    cy.get('#nextBtn').click();
    cy.get('#nextBtn').click();
    cy.wait('@estimate');
    cy.get('#nextBtn').click();
    cy.contains('Could not start generation: Missing DEEPINFRA_API_KEY').should('be.visible');
    cy.get('#gen-config').should('be.visible');
  });
});
