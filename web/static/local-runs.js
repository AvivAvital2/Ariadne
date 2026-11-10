/* Durable local runs belong to a saved project, independently of browser jobs. */
(function(){
  window.createLocalRuns=function(root,{canResume,onChange=()=>{}}){
    let project=null, runs=[], timer=null, revision=0, pending=false, paused=false;
    let statusError=null, actionError=null;
    const cards=new Map();
    const terminal=new Set(['completed','cancelled','failed','obsolete']);
    root.innerHTML='<strong>Saved local builds</strong><p data-empty>Checking saved builds…</p>'+ 
      '<p role="alert" hidden></p><div data-runs></div>';
    const list=root.querySelector('[data-runs]');
    function render(){
      const error=statusError||actionError;
      const alert=root.querySelector('[role="alert"]');
      alert.textContent=error||''; alert.hidden=!error;
      root.querySelector('[data-empty]').hidden=!!runs.length;
      root.querySelector('[data-empty]').textContent=statusError?'Saved builds unavailable. Retrying…':'No saved local builds.';
      const ids=new Set(runs.map(run=>run.run_id));
      for(const [id,card] of cards){ if(!ids.has(id)){card.remove();cards.delete(id);} }
      for(const run of runs){
        let card=cards.get(run.run_id);
        if(!card){
          card=document.createElement('article'); card.className='local-run';
          card.innerHTML='<p><strong data-state></strong> · <span data-counts></span></p>'+ 
            '<p data-id></p><p data-detail></p><div class="local-models-actions">'+ 
            '<button type="button" data-action="resume">Resume build</button>'+ 
            '<button type="button" data-action="cancel">Cancel build</button></div>';
          card.dataset.runId=run.run_id; cards.set(run.run_id,card); list.append(card);
        }
        const labels={completed:'Completed',cancelled:'Cancelled',failed:'Failed',obsolete:'Obsolete'};
        card.querySelector('[data-state]').textContent=labels[run.state]||(run.active_here?'Running':'Unfinished');
        card.querySelector('[data-counts]').textContent=(run.counts?.applied||0)+' / '+(run.counts?.total||0)+' jobs applied';
        card.querySelector('[data-id]').textContent=run.run_id;
        card.querySelector('[data-detail]').textContent=run.execution_error||
          (terminal.has(run.state)?'Completed work is retained.':
            run.active_here?'Generation is running. You can close this page.':
            'Saved progress is available. Resume continues this build; Cancel retains completed work.');
        const disabled=pending||!!statusError||terminal.has(run.state);
        card.querySelector('[data-action="resume"]').disabled=disabled||run.active_here||!canResume(project);
        card.querySelector('[data-action="cancel"]').disabled=disabled;
      }
    }
    function schedule(){
      clearTimeout(timer);
      if(project&&!paused) timer=setTimeout(refresh,1000);
    }
    async function request(url,options){
      const response=await fetch(url,{...options,signal:AbortSignal.timeout(options?120000:10000)});
      const body=await response.json();
      if(!response.ok) throw new Error(body.error||('HTTP '+response.status));
      return body;
    }
    async function refresh(){
      if(!project||paused||pending){schedule();return;}
      clearTimeout(timer); const current=++revision;
      try{
        const body=await request('/api/local-runs?project_id='+encodeURIComponent(project.id));
        if(current!==revision||paused) return;
        if(!Array.isArray(body.runs)) throw new Error('Invalid saved-build status.');
        runs=body.runs; statusError=null; onChange(runs);
      }catch(error){ if(current===revision&&!paused) statusError=error.message; }
      finally{ if(current===revision&&!paused){render();schedule();} }
    }
    async function action(runId,name){
      if(pending||!project||paused) return;
      clearTimeout(timer); const current=++revision;
      pending=true; actionError=null; render();
      try{
        await request('/api/local-runs',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({project_id:project.id,run_id:runId,action:name})});
      }catch(error){ if(current===revision) actionError=error.message; }
      finally{
        pending=false;
        if(current===revision&&!paused){render();await refresh();}
      }
    }
    root.addEventListener('click',event=>{
      const button=event.target.closest('button[data-action]');
      if(button&&!button.disabled) action(button.closest('[data-run-id]').dataset.runId,button.dataset.action);
    });
    const control={render,refresh,setProject(value){
      project=value?.provider==='local'?value:null;
      ++revision; clearTimeout(timer); runs=[]; statusError=null; actionError=null;
      root.hidden=!project;
      if(project){render();refresh();}
    }};
    window.addEventListener('pagehide',()=>{paused=true;++revision;clearTimeout(timer);});
    window.addEventListener('pageshow',()=>{if(paused){paused=false;refresh();}});
    return control;
  };
})();
