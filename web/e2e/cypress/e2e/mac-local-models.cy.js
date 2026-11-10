const choice = {
  id:'qwen-mac-q4-v1', model:'bartowski/Qwen_Qwen3.5-9B-GGUF', provider:'local',
  embedding_model:'Qwen/Qwen3-Embedding-8B-GGUF', label:'Qwen local on Mac',
  managed_local:true, available:false, preview_available:true, reason:'Start local models before building this project.',
  managed_local_profile:'qwen-mac-q4-v1', profile_hash:'legacy-profile-hash',
  quality_note:'Experimental: generated documentation has shown factual errors.',
};
const accuracy={...choice,id:'qwen-mac-27b-q4-thinking-v1',managed_local_profile:'qwen-mac-27b-q4-thinking-v1',
  profile_hash:'accuracy-profile-hash',model:'Qwen3.8-27B-Q4',label:'27B thinking + 8B embeddings',
  quality_note:'Experimental: 27B thinking has not passed Ariadne quality qualification.',
  memory_note:'32GB qualification target; 24GB fit is unqualified. Memory use has not been measured on M5.'};
const project='0123456789abcdef01234567';
const estimate={source:'demo', model:choice.model, provider:'local', batch_supported:false,
  max_concurrency:1, input_per_million:0, output_per_million:0, file_count:1,
  input_tokens:100, output_tokens:100, embedding_tokens:100, total_cost_usd:0,
  total_cost_batched_usd:0, embedding_cost_usd:0, languages:[], by_doc_type:[], by_directory:[]};

describe('Managed Mac models from the UI',()=>{
  let base, state;
  beforeEach(()=>{
    cy.task('fixedStaticUrl').then(url=>{base=url;});
    state={phase:'stopped', busy:false, error:null, platform_error:null, qualification:'unqualified',
      profile_id:choice.id,profile_hash:choice.profile_hash};
    cy.intercept('GET','/api/local-models',req=>req.reply(state)).as('status');
    cy.intercept('GET','/api/onboarding-models',{models:[choice],default:choice.id});
    cy.intercept('GET','/api/projects',{projects:[{...choice,id:project,source:'demo'}]});
    cy.intercept('POST','/api/sources',{default_source:'demo',sources:[{name:'demo'}]});
    cy.intercept('POST','/api/source_add',{source:'demo',created:true});
    cy.intercept('POST','/api/discover',{source:'demo',file_count:1,languages:[],indexers:[]}).as('discover');
    cy.intercept('POST','/api/estimate',estimate).as('estimate');
  });

  it('downloads, starts, builds, and reopens the saved pair without terminal configuration',()=>{
    cy.intercept('POST','/api/local-models',req=>{
      expect(req.body).to.deep.equal(req.body.action==='stop'?{action:'stop'}:{action:req.body.action,profile_id:choice.id});
      if(req.body.action==='download') state={...state,phase:'downloading',busy:true,file:'generation model',downloaded_bytes:1000000,total_bytes:2000000};
      if(req.body.action==='start') state={...state,phase:'starting',busy:true};
      if(req.body.action==='stop') state={...state,phase:'stopped',busy:false};
      req.reply({statusCode:202,body:state});
    }).as('action');
    cy.intercept('POST','/api/onboard',{job_id:'mac-build',project_id:project}).as('build');
    cy.intercept('GET','/api/onboard/events*',{headers:{'content-type':'text/event-stream'},body:
      'data: {"type":"done","result":{"docs_written":1,"themes_found":0,"coverage_percent":100,"files_indexed":1}}\n\n'});
    cy.visit(base+'/');
    cy.get('#localModels').should('be.visible');
    cy.get('#localModels [data-action="download"]').click();
    cy.wait('@action').its('request.body.action').should('eq','download');
    cy.get('#localModels [role="status"]').should('contain','Downloading');
    cy.get('#localModels progress').should('have.value',1000000);
    cy.get('#localModels [data-action="start"]').should('be.disabled');
    cy.then(()=>{state={...state,phase:'stopped',busy:false};});
    cy.get('#localModels [data-action="start"]').should('be.enabled').click();
    cy.wait('@action').its('request.body.action').should('eq','start');
    cy.get('#localModels [role="status"]').should('contain','Starting');
    cy.then(()=>{state={...state,phase:'ready',busy:false};});
    cy.get('#localModels [role="status"]').should('contain','Ready');
    cy.get('#srcName').type('demo'); cy.get('#srcPath').type('/workspace/demo');
    cy.get('#nextBtn').click(); cy.wait('@discover');
    cy.get('#nextBtn').click(); cy.wait('@estimate');
    cy.get('#nextBtn').click(); cy.get('#nextBtn').click(); cy.wait('@estimate');
    cy.get('#nextBtn').click();
    cy.wait('@build').its('request.body.onboarding_model').should('eq',choice.id);
    cy.get('#gen-ready a').click();
    cy.location('search').should('eq','?project='+project);
    cy.get('#localModels [role="status"]').should('contain','Ready');
    cy.get('#localModels [data-action="stop"]').click();
    cy.wait('@action').its('request.body.action').should('eq','stop');
    cy.reload();
    cy.get('#localModels [role="status"]').should('contain','Stopped');
    cy.get('#localModels [data-action="start"]').should('be.enabled');
  });

  it('shows failures as text and allows retry or download cancellation',()=>{
    cy.intercept('POST','/api/local-models',req=>{
      state=req.body.action==='stop'?{...state,phase:'stopped',busy:false,error:null}:
        {...state,phase:'error',error:'Download failed <img src=x onerror=alert(1)>'};
      req.reply({statusCode:202,body:state});
    }).as('action');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localModels [data-action="download"]').click(); cy.wait('@action');
    cy.get('#localModels [role="alert"]').should('contain','Download failed <img');
    cy.get('#localModels img').should('not.exist');
    cy.get('#localModels [data-action="download"]').should('be.enabled');
    cy.then(()=>{state={...state,phase:'downloading',busy:true,error:null};});
    cy.get('#localModels [data-action="stop"]').should('be.enabled').click(); cy.wait('@action');
    cy.get('#localModels [role="status"]').should('contain','Stopped');
  });

  it('explains unsupported hardware without offering an unusable start',()=>{
    state.platform_error='The managed Mac runtime requires native Apple Silicon macOS.';
    cy.visit(base+'/');
    cy.get('#localModels').should('contain','requires native Apple Silicon');
    cy.get('#localModels [data-action="download"]').should('be.disabled');
    cy.get('#localModels [data-action="start"]').should('be.disabled');
  });

  it('selects the Mac pair explicitly, previews while stopped, and builds only after readiness',()=>{
    const hosted={id:'configured',model:'gpt-6-astra',provider:'openai',available:true,preview_available:true};
    const external={...choice,id:'external',managed_local:false,available:true};
    cy.intercept('GET','/api/onboarding-models',{models:[hosted,external,choice],default:hosted.id});
    cy.intercept('POST','/api/onboard',{job_id:'chosen-mac',project_id:project}).as('build');
    cy.intercept('GET','/api/onboard/events*',{headers:{'content-type':'text/event-stream'},body:
      'data: {"type":"done","result":{"docs_written":1}}\n\n'});
    cy.visit(base+'/');
    cy.get('#localModels').should('not.be.visible');
    cy.get('#srcName').type('demo'); cy.get('#srcPath').type('/workspace/demo');
    cy.get('#nextBtn').click(); cy.wait('@discover');
    cy.get('#nextBtn').click(); cy.wait('@estimate');
    cy.get('#modeldd .ddbtn').click();
    cy.get(`#modelSeg [data-model="${choice.id}"]`).click();
    cy.wait('@estimate').its('request.body.onboarding_model').should('eq',choice.id);
    cy.get('#localModels').scrollIntoView().should('be.visible').and('contain','Experimental').and('contain','factual errors');
    cy.get('#nextBtn').click(); cy.get('#nextBtn').click(); cy.wait('@estimate');
    cy.get('#nextBtn').should('be.disabled');
    cy.get('@build.all').should('have.length',0);
    cy.then(()=>{state={...state,phase:'ready'};});
    cy.get('#nextBtn').should('be.enabled').click();
    cy.wait('@build').then(({request})=>{
      expect(request.body.onboarding_model).to.eq(choice.id);
      expect(request.body).not.to.have.property('model');
    });
  });

  it('retains the managed project for Ask/Search and blocks queries whenever its runtime is unavailable',()=>{
    cy.intercept('POST','/api/ask',{answer:'A grounded answer.',documents:[]}).as('ask');
    cy.intercept('POST','/api/search',{documents:[]}).as('search');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localModels [role="status"]').should('contain','Stopped');
    cy.get('#q').type('What does the code do?{enter}');
    cy.get('#askBtn').should('be.disabled');
    cy.get('@ask.all').should('have.length',0);
    for(const phase of ['starting','stopping','error']){
      cy.then(()=>{state={...state,phase};});
      cy.get('#localModels [role="status"]').should('contain',phase[0].toUpperCase()+phase.slice(1));
      cy.get('#askBtn').should('be.disabled');
    }
    cy.then(()=>{state={...state,phase:'ready',busy:true};});
    cy.get('#localModels [data-action="start"]').should('be.disabled');
    cy.get('#askBtn').should('be.disabled');
    cy.then(()=>{state={...state,busy:false};});
    cy.get('#askBtn').should('be.enabled');
    cy.reload();
    cy.get('#askBtn').should('be.enabled');
    cy.get('#q').type('What does the code do?{enter}');
    for(const alias of ['@ask','@search']) cy.wait(alias).then(({request})=>{
      expect(request.body.project_id).to.eq(project);
      expect(request.body.source).to.eq('demo');
      expect(request.body).not.to.have.property('model');
    });
    cy.get('#answerCard').should('contain','A grounded answer.');
  });

  it('invalidates stale readiness on status failure and lets a failed start be retried',()=>{
    state={...state,phase:'ready'};
    let unavailable=false, starts=0;
    cy.intercept('GET','/api/local-models',req=>req.reply(unavailable?
      {statusCode:503,body:{error:'Status unavailable <img src=x>'}}:state));
    cy.intercept('POST','/api/local-models',req=>{
      expect(req.body.action).to.eq('start');
      starts++;
      req.reply(starts===1?{statusCode:500,body:{error:'Start failed <img src=x>'}}:
        {statusCode:202,body:{...state,phase:'starting',busy:true}});
    }).as('start');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#askBtn').should('be.enabled');
    cy.then(()=>{unavailable=true;});
    cy.get('#localModels [role="alert"]').should('contain','Status unavailable <img');
    cy.get('#localModels img').should('not.exist');
    cy.get('#askBtn').should('be.disabled');
    cy.then(()=>{unavailable=false;state={...state,phase:'stopped'};});
    cy.get('#localModels [data-action="start"]').should('be.enabled').click(); cy.wait('@start');
    cy.get('#localModels [role="alert"]').should('contain','Start failed <img');
    cy.get('#localModels img').should('not.exist');
    cy.get('#localModels [data-action="start"]').should('be.enabled').click(); cy.wait('@start');
  });

  it('does not gate an external project with the same model names on the managed runtime',()=>{
    cy.intercept('GET','/api/projects',{projects:[{...choice,id:project,source:'demo',managed_local:false}]});
    cy.intercept('GET','/api/local-models',()=>{throw new Error('External projects do not use the managed runtime');});
    cy.intercept('POST','/api/ask',{answer:'External answer.',documents:[]}).as('ask');
    cy.intercept('POST','/api/search',{documents:[]}).as('search');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localModels').should('not.be.visible');
    cy.get('#q').type('What does the code do?{enter}');
    cy.wait('@ask'); cy.wait('@search');
    cy.get('#answerCard').should('contain','External answer.');
  });

  it('selects the accuracy profile explicitly and describes its unqualified hardware scope',()=>{
    cy.intercept('GET','/api/onboarding-models',{models:[accuracy,choice],default:accuracy.id});
    cy.intercept('POST','/api/local-models',req=>{
      expect(req.body).to.deep.equal({action:'start',profile_id:accuracy.id});
      state={...state,phase:'ready',profile_id:accuracy.id,profile_hash:accuracy.profile_hash};
      req.reply({statusCode:202,body:state});
    }).as('accuracyStart');
    cy.visit(base+'/');
    cy.get('#localModels').should('contain','27B thinking').and('contain','32GB qualification target')
      .and('contain','24GB fit is unqualified').and('contain','not passed');
    cy.get('#localModels [data-action="start"]').click();
    cy.wait('@accuracyStart');
    cy.get('#localModels [role="status"]').should('contain','Ready');
  });

  it('reopens a legacy project without accepting another profiles ready servers',()=>{
    state={...state,phase:'ready',profile_id:accuracy.id,profile_hash:accuracy.profile_hash};
    cy.intercept('POST','/api/local-models',req=>{
      if(req.body.action==='stop'){
        expect(req.body).to.deep.equal({action:'stop'});
        state={...state,phase:'stopped'};
      }else{
        expect(req.body).to.deep.equal({action:'start',profile_id:choice.id});
        state={...state,phase:'ready',profile_id:choice.id,profile_hash:choice.profile_hash};
      }
      req.reply({statusCode:202,body:state});
    }).as('switch');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#askBtn').should('be.disabled');
    cy.get('#localModels').should('contain','different model profile');
    cy.get('#localModels [data-action="start"]').should('be.disabled');
    cy.get('#localModels [data-action="stop"]').click(); cy.wait('@switch');
    cy.get('#localModels [data-action="start"]').should('be.enabled').click(); cy.wait('@switch');
    cy.get('#askBtn').should('be.enabled');
    cy.reload();
    cy.get('#askBtn').should('be.enabled');
  });

  it('rejects a matching profile name with a different model identity',()=>{
    state={...state,phase:'ready',profile_hash:'different-weights'};
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#askBtn').should('be.disabled');
    cy.get('#localModels').should('contain','different model profile');
  });

  it('refreshes runtime readiness after restoring a page from the back/forward cache',()=>{
    state={...state,phase:'ready'};
    cy.intercept('POST','/api/ask',{answer:'Restored answer.',documents:[]}).as('ask');
    cy.intercept('POST','/api/search',{documents:[]}).as('search');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#askBtn').should('be.enabled');
    cy.window().then(win=>{
      win.dispatchEvent(new win.PageTransitionEvent('pagehide',{persisted:true}));
      state={...state,phase:'stopped'};
      win.dispatchEvent(new win.PageTransitionEvent('pageshow',{persisted:true}));
    });
    cy.get('#localModels [role="status"]').should('contain','Stopped');
    cy.get('#askBtn').should('be.disabled');
    cy.then(()=>{state={...state,phase:'ready'};});
    cy.get('#askBtn').should('be.enabled');
    cy.get('#q').type('Can I continue?{enter}');
    cy.wait('@ask'); cy.wait('@search');
    cy.get('#answerCard').should('contain','Restored answer.');
  });
});
