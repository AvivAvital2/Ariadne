/// <reference types="cypress" />
// Opt-in paid smoke against an already-running Docker demo. Use a one-file
// synthetic project only; this spec refuses larger sources before generation.
const url = Cypress.env('HOSTED_ONBOARDING_URL');
const previewUrl = Cypress.env('PREVIEW_ONBOARDING_URL');
(previewUrl ? describe : describe.skip)('Docker cost comparison without inference', () => {
  it('compares GPT, Claude and hosted Qwen before saving any project', () => {
    let projects;
    cy.request(previewUrl+'/api/projects').then(({body})=>{projects=body.projects;});
    cy.intercept('POST', '/api/onboard', ()=>{throw new Error('Preview must not start a build');});
    cy.intercept('POST', '/api/source_add').as('connect');
    cy.intercept('POST', '/api/discover').as('discover');
    cy.intercept('POST', '/api/estimate').as('estimate');
    cy.visit(previewUrl+'/');
    cy.get('#srcName').type('retry-demo');
    cy.get('#srcPath').type('/workspace/retry-demo');
    cy.get('#nextBtn').click();
    cy.wait('@connect').then(({request,response})=>{
      expect(request.body).not.to.have.property('onboarding_model');
      expect(response.body).not.to.have.property('project_id');
    });
    cy.wait('@discover', {timeout:120000});
    cy.get('#nextBtn').click();
    cy.wait('@estimate', {timeout:120000});
    const costs={};
    ['gpt-6-astra','claude-opus-5-5','qwen-hosted'].forEach(id=>{
      cy.get('#modeldd .ddbtn').click();
      cy.get(`#modelSeg [data-model="${id}"]`).click();
      cy.wait('@estimate', {timeout:120000}).then(({request,response})=>{
        expect(response.statusCode).to.eq(200);
        expect(request.body.onboarding_model).to.eq(id);
        expect(response.body).not.to.have.property('project_id');
        expect(response.body.file_count).to.eq(1);
        expect(response.body.embedding_cost_usd).to.be.greaterThan(0);
        costs[id]=response.body.grand_total_cost_usd;
      });
      cy.get('#nextBtn').should('not.be.disabled');
    });
    cy.then(()=>expect(costs['gpt-6-astra']).to.be.greaterThan(costs['claude-opus-5-5']));
    cy.then(()=>expect(costs['claude-opus-5-5']).to.be.greaterThan(costs['qwen-hosted']));
    cy.request(previewUrl+'/api/projects').then(({body})=>expect(body.projects).to.deep.eq(projects));
    cy.screenshot('model-preview', {capture:'viewport'});
  });
});

(url ? describe : describe.skip)('Hosted Qwen in the Docker onboarding UI', () => {
  it('selects Qwen once, generates, and reopens its saved library for Ask', () => {
    cy.intercept('POST', '/api/source_add').as('connect');
    cy.intercept('POST', '/api/discover').as('discover');
    cy.intercept('POST', '/api/estimate').as('estimate');
    cy.intercept('POST', '/api/onboard').as('onboard');
    cy.intercept('GET', '/api/onboard/events*').as('events');
    cy.visit(url+'/');
    cy.get('#srcName').type('retry-demo');
    cy.get('#srcPath').type('/workspace/retry-demo');
    cy.get('#nextBtn').click();
    let project;
    cy.wait('@connect', {timeout:60000}).then(({request,response})=>{
      expect(response.statusCode).to.eq(200);
      expect(request.body).not.to.have.property('onboarding_model');
      expect(response.body).not.to.have.property('project_id');
    });
    cy.wait('@discover', {timeout:120000});
    cy.get('#nextBtn').click();
    cy.wait('@estimate', {timeout:120000});
    cy.get('#modeldd .ddbtn').click();
    ['gpt-6-astra', 'claude-opus-5-5', 'qwen-hosted'].forEach(id=>{
      cy.get(`#modelSeg [data-model="${id}"]`).should('not.have.attr','aria-disabled','true');
    });
    cy.get('#modelSeg [data-model="qwen-hosted"]').click();
    cy.wait('@estimate', {timeout:120000}).then(({response})=>{
      expect(response.statusCode).to.eq(200);
      expect(response.body.file_count, 'only the synthetic file may be sent').to.eq(1);
      expect(response.body.provider).to.eq('deepinfra');
      expect(response.body.total_cost_usd).to.be.greaterThan(0).and.lessThan(0.15);
    });
    cy.get('#engBatch').should('not.be.visible');
    cy.get('#nextBtn').click();
    ['architecture','qa','gotcha','diagram'].forEach(type=>{
      cy.get(`.dtrow[data-dt="${type}"] [data-act="dt"]`).click();
    });
    cy.get('#nextBtn').click();
    cy.wait('@estimate', {timeout:120000});
    cy.get('#nextBtn').click();
    cy.wait('@onboard').then(({request,response})=>{
      expect(response.statusCode).to.eq(200);
      expect(request.body.onboarding_model).to.eq('qwen-hosted');
      project=response.body.project_id;
    });
    cy.wait('@events', {timeout:600000}).then(({response})=>{
      expect(response.body).not.to.match(/"type"\s*:\s*"error"/);
    });
    cy.get('#gen-ready', {timeout:600000}).should('be.visible');
    cy.get('#gen-ready a').click();
    cy.then(()=>cy.location('search').should('eq','?project='+project));
    cy.reload();
    cy.intercept('POST', '/api/ask').as('ask');
    cy.intercept('POST', '/api/search').as('search');
    cy.get('#q').type('How does Retry limit attempts and calculate the delay?{enter}');
    cy.wait('@ask', {timeout:180000}).then(({request,response})=>{
      expect(response.statusCode).to.eq(200);
      expect(request.body.project_id).to.eq(project);
      expect(request.body).not.to.have.property('model');
    });
    cy.wait('@search').then(({response})=>{
      expect(response.statusCode).to.eq(200);
      expect(response.body.documents).not.to.be.empty;
    });
    cy.get('#answerCard').should('be.visible').invoke('text').should('match', /retry|delay|attempt/i);
  });
});
