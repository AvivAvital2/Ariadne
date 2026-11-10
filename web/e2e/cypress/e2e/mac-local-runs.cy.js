const project='0123456789abcdef01234567';
const runId='local:0123456789abcdef0123456789abcdef';
const choice={id:'qwen-mac-q4-v1',model:'Qwen/Qwen3.5-9B',provider:'local',managed_local:true,
  managed_local_profile:'qwen-mac-q4-v1',profile_hash:'legacy-profile-hash',
  available:true,preview_available:true,label:'Experimental Mac models'};

describe('Saved local builds in the UI',()=>{
  let base, runtime, run;
  beforeEach(()=>{
    cy.task('fixedStaticUrl').then(url=>{base=url;});
    runtime={phase:'stopped',busy:false,platform_error:null,qualification:'unqualified',
      profile_id:choice.id,profile_hash:choice.profile_hash};
    run={run_id:runId,client_run_id:'original-job',source:'demo',operation_kind:'onboard',
      phase:'generate',state:'running',counts:{applied:1,total:2},active_here:false};
    cy.intercept('GET','/api/projects',{projects:[{...choice,id:project,source:'demo'}]});
    cy.intercept('GET','/api/onboarding-models',{models:[choice],default:choice.id});
    cy.intercept('POST','/api/sources',{default_source:'demo',sources:[{name:'demo'}]});
    cy.intercept('GET','/api/local-models',req=>req.reply(runtime));
    cy.intercept('GET','/api/local-runs*',req=>{
      expect(req.query.project_id).to.eq(project);
      req.reply({runs:[run]});
    }).as('runs');
  });

  it('finds an interrupted build after reopening and resumes it after models start',()=>{
    cy.intercept('POST','/api/local-models',req=>{
      expect(req.body).to.deep.eq({action:'start',profile_id:choice.id});
      runtime={...runtime,phase:'ready'};
      req.reply({statusCode:202,body:runtime});
    }).as('start');
    cy.intercept('POST','/api/local-runs',req=>{
      expect(req.body).to.deep.eq({project_id:project,run_id:runId,action:'resume'});
      run={...run,active_here:true};
      req.reply({statusCode:202,body:{job_id:'resume-job',run_id:runId,project_id:project}});
    }).as('resume');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localRuns').should('contain','Unfinished').and('contain','1 / 2 jobs applied');
    cy.get('#localRuns [data-action="resume"]').should('be.disabled');
    cy.get('#localRuns [data-action="cancel"]').should('be.enabled');
    cy.get('#localModels [data-action="start"]').click(); cy.wait('@start');
    cy.get('#localRuns [data-action="resume"]').should('be.enabled').click(); cy.wait('@resume');
    cy.get('#localRuns').should('contain','Running');
    cy.get('#localRuns [data-action="resume"]').should('be.disabled');
    cy.then(()=>{run={...run,state:'completed',active_here:false,counts:{applied:2,total:2}};});
    cy.get('#localRuns').should('contain','Completed').and('contain','2 / 2');
    cy.reload();
    cy.get('#localRuns').should('contain','Completed');
    cy.get('#localRuns [data-action="resume"]').should('be.disabled');
  });

  it('cancels an unfinished run while models are stopped and retains completed work',()=>{
    cy.intercept('POST','/api/local-runs',req=>{
      expect(req.body).to.deep.eq({project_id:project,run_id:runId,action:'cancel'});
      run={...run,state:'cancelled',cancel_requested:true};
      req.reply({run});
    }).as('cancel');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localRuns [data-action="cancel"]').click(); cy.wait('@cancel');
    cy.get('#localRuns').should('contain','Cancelled').and('contain','1 / 2');
    cy.get('#localRuns [data-action="resume"]').should('be.disabled');
    cy.get('#localRuns [data-action="cancel"]').should('be.disabled');
  });

  it('recovers the exact saved onboarding run when its old event stream is gone',()=>{
    runtime={...runtime,phase:'ready'};
    cy.intercept('GET','/api/onboard/events*',{statusCode:404,body:{error:'unknown job_id'}});
    cy.intercept('POST','/api/onboard',()=>{throw new Error('Recovery must not create a new build');});
    cy.visit(base+'/',{onBeforeLoad(win){
      win.localStorage.setItem('ariadneOnboardJob',JSON.stringify({job_id:'original-job',source:'demo',project_id:project}));
    }});
    cy.get('#localRuns').should('contain',runId).and('contain','Unfinished');
    cy.get('#localRuns [data-action="resume"]').should('be.enabled');
    cy.then(()=>{run={...run,state:'completed',counts:{applied:2,total:2}};});
    cy.get('#gen-ready').should('be.visible');
    cy.get('#gen-ready a').should('have.attr','href','/static/console.html?project='+project);
    cy.window().then(win=>expect(win.localStorage.getItem('ariadneOnboardJob')).to.eq(null));
  });

  it('shows action errors as text and preserves the run for retry',()=>{
    runtime={...runtime,phase:'ready'};
    cy.intercept('POST','/api/local-runs',{statusCode:409,body:{error:'Writer busy <img src=x onerror=alert(1)>'}}).as('resume');
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localRuns [data-action="resume"]').should('be.enabled').click(); cy.wait('@resume');
    cy.get('#localRuns [role="alert"]').should('contain','Writer busy <img');
    cy.get('#localRuns img').should('not.exist');
    cy.get('#localRuns').should('contain','Unfinished').and('contain','1 / 2');
    cy.get('#localRuns [data-action="resume"]').should('be.enabled');
  });

  for(const projectsFirst of [true,false]){
    it('retains the hosted lost-stream fallback when project settings arrive '+(projectsFirst?'first':'last'),()=>{
      const hosted={id:project,source:'demo',provider:'deepinfra',model:'Qwen/Qwen3.8-27B'};
      let releaseProjects;
      cy.intercept('GET','/api/projects',req=>{
        if(projectsFirst) req.reply({projects:[hosted]});
        else return new Promise(resolve=>{
          releaseProjects=()=>{req.reply({projects:[hosted]});resolve();};
        });
      }).as('projects');
      cy.intercept('POST','/api/onboard',()=>{throw new Error('Reconnecting must not create a hosted build');});
      cy.visit(base+'/',{onBeforeLoad(win){
        win.localStorage.setItem('ariadneOnboardJob',JSON.stringify({job_id:'hosted-job',source:'demo',project_id:project}));
        win.__streams=[];
        win.EventSource=class {
          static CLOSED=2;
          constructor(){this.readyState=1;win.__streams.push(this);}
          close(){this.readyState=2;}
        };
      }});
      if(projectsFirst) cy.wait('@projects');
      else cy.window().should(()=>expect(releaseProjects).to.be.a('function'));
      cy.window().then(win=>{
        const stream=win.__streams[0];
        stream.readyState=win.EventSource.CLOSED;stream.onerror();
      });
      if(!projectsFirst){cy.then(()=>releaseProjects());cy.wait('@projects');}
      cy.get('#runLine').should('contain','Lost track of this build');
      cy.get('#localRuns').should('not.be.visible');
      cy.get('@runs.all').should('have.length',0);
      cy.window().then(win=>expect(win.localStorage.getItem('ariadneOnboardJob')).to.eq(null));
    });
  }

  it('disables resume after a failed status poll until fresh status succeeds',()=>{
    runtime={...runtime,phase:'ready'};
    let unavailable=false;
    cy.intercept('GET','/api/local-runs*',req=>req.reply(unavailable?
      {statusCode:503,body:{error:'Saved-build service unavailable'}}:{runs:[run]}));
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localRuns [data-action="resume"]').should('be.enabled');
    cy.then(()=>{unavailable=true;});
    cy.get('#localRuns [role="alert"]').should('contain','Saved-build service unavailable');
    cy.get('#localRuns [data-action="resume"]').should('be.disabled');
    cy.then(()=>{unavailable=false;});
    cy.get('#localRuns [data-action="resume"]').should('be.enabled');
    cy.get('#localRuns [role="alert"]').should('not.be.visible');
  });

  it('refreshes saved runs after returning from the browser back-forward cache',()=>{
    runtime={...runtime,phase:'ready'};
    cy.visit(base+'/static/console.html?project='+project);
    cy.get('#localRuns').should('contain','Unfinished');
    cy.window().then(win=>{
      win.dispatchEvent(new win.PageTransitionEvent('pagehide',{persisted:true}));
      run={...run,state:'completed',counts:{applied:2,total:2}};
      win.dispatchEvent(new win.PageTransitionEvent('pageshow',{persisted:true}));
    });
    cy.get('#localRuns').should('contain','Completed');
    cy.get('#localRuns [data-action="resume"]').should('be.disabled');
  });
});
