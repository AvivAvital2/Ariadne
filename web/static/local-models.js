/* Shared managed-runtime controls. Model selection and query scope stay with each page. */
(function(){
  window.createLocalModels = function(root, onChange){
    let active=false, selected=null, state=null, pending=false, timer=null, revision=0, restoreActive=false;
    let statusError=null, actionError=null;
    root.innerHTML='<div class="local-models-heading"><strong>Local models on this Mac</strong>'+ 
      '<span role="status" aria-live="polite">Checking status…</span></div>'+ 
      '<p data-profile></p><p data-quality></p><p data-memory></p>'+ 
      '<p data-detail></p><progress hidden aria-label="Model download progress"></progress>'+ 
      '<p role="alert" hidden></p><div class="local-models-actions">'+ 
      '<button type="button" data-action="download">Download models</button>'+ 
      '<button type="button" data-action="start">Start models</button>'+ 
      '<button type="button" data-action="stop">Stop models</button>'+ 
      '<button type="button" data-action="refresh">Refresh status</button></div>';
    const find=selector=>root.querySelector(selector);
    const matches=()=>!!selected?.managed_local_profile && !!selected?.profile_hash &&
      state?.profile_id===selected.managed_local_profile && state?.profile_hash===selected.profile_hash;
    const ready=()=>active && matches() && !pending && !statusError && state?.phase==='ready' &&
      !state.busy && !state.platform_error;

    function render(){
      const phase=state?.phase;
      const names={stopped:'Stopped',downloading:'Downloading',starting:'Starting',ready:'Ready',stopping:'Stopping',error:'Error'};
      find('[role="status"]').textContent=statusError?'Status unavailable':names[phase]||'Checking status…';
      find('[data-profile]').textContent=selected?.label||selected?.model||'Saved local model profile';
      find('[data-quality]').textContent=selected?.quality_note||'Experimental: model quality has not been qualified.';
      find('[data-quality]').hidden=selected?.qualification==='qualified';
      find('[data-memory]').textContent=selected?.memory_note||'Memory use on the target Mac remains unqualified.';
      find('[data-detail]').textContent=state?.platform_error || (
        phase==='downloading'?'Downloading '+(state.file||'model files'):
        phase==='ready'?(matches()?'Both model servers are running.':
          'A different model profile is running. Stop models, then start this project\'s saved profile.'):
        phase==='stopped'?'Download the models once, then start them to generate or ask questions.':'');
      const error=statusError || actionError || state?.error;
      find('[role="alert"]').textContent=error||'';
      find('[role="alert"]').hidden=!error;
      const progress=find('progress');
      progress.hidden=phase!=='downloading';
      progress.max=Math.max(1,state?.total_bytes||1);
      progress.value=state?.downloaded_bytes||0;
      const unavailable=!state || !!statusError || !!state.platform_error ||
        !selected?.managed_local_profile || !selected?.profile_hash;
      const busy=pending || state?.busy || ['downloading','starting','stopping'].includes(phase);
      find('[data-action="download"]').disabled=unavailable || busy || phase==='ready';
      find('[data-action="start"]').disabled=unavailable || busy || phase==='ready';
      find('[data-action="stop"]').disabled=pending || !state || !(state.busy || ['downloading','starting','ready','error'].includes(phase));
      find('[data-action="stop"]').textContent=['downloading','starting'].includes(phase)?'Cancel operation':'Stop models';
      find('[data-action="refresh"]').disabled=pending;
      onChange({ready:ready(),state,error});
    }

    function schedule(){
      clearTimeout(timer);
      if(active) timer=setTimeout(refresh,1000);
    }

    async function request(options){
      const response=await fetch('/api/local-models',{...options,signal:AbortSignal.timeout(10000)});
      const body=await response.json();
      if(!response.ok) throw new Error(body.error||('HTTP '+response.status));
      if(!body || typeof body.phase!=='string') throw new Error('Invalid local model status. Refresh to retry.');
      return body;
    }

    async function refresh(){
      if(!active || pending) { schedule(); return false; }
      clearTimeout(timer);
      const current=++revision;
      try{
        const next=await request();
        if(!active || current!==revision) return false;
        state=next; statusError=null;
      }catch(error){
        if(!active || current!==revision) return false;
        state=null; statusError=error.message;
      }finally{
        if(active && current===revision){ render(); schedule(); }
      }
      return ready();
    }

    async function action(name){
      if(name==='refresh'){ await refresh(); return; }
      if(pending) return;
      clearTimeout(timer);
      const current=++revision;
      pending=true; actionError=null; render();
      try{
        const body={action:name};
        if(name!=='stop') body.profile_id=selected.managed_local_profile;
        const next=await request({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
        if(active && current===revision){ state=next; statusError=null; }
      }catch(error){
        if(active && current===revision) actionError=error.message;
      }finally{
        pending=false;
        if(active){ render(); schedule(); }
      }
    }

    root.addEventListener('click',event=>{
      const button=event.target.closest('button[data-action]');
      if(button && !button.disabled) action(button.dataset.action);
    });
    const control={
      get ready(){return ready();},
      refresh,
      setProfile(value){
        const same=selected?.managed_local_profile===value?.managed_local_profile &&
          selected?.profile_hash===value?.profile_hash && active===!!value?.managed_local;
        selected=value;
        if(same){ if(active) render(); return; }
        active=!!value?.managed_local; root.hidden=!active; ++revision; clearTimeout(timer);
        if(active){ state=null; statusError=null; actionError=null; render(); refresh(); }
      },
    };
    window.addEventListener('pagehide',()=>{
      restoreActive=active; active=false; ++revision; clearTimeout(timer);
    });
    window.addEventListener('pageshow',()=>{
      if(restoreActive){ restoreActive=false; control.setProfile(selected); }
    });
    return control;
  };
})();
