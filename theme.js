/* Apply the preference before the styles load, including on deep article links. */
(()=>{
  const key='darrenzhang-theme',root=document.documentElement;
  const media=window.matchMedia('(prefers-color-scheme: dark)');
  const labels={system:'跟随系统',light:'日间',dark:'夜间'};
  const valid=value=>Object.hasOwn(labels,value);
  let preference='system';
  try{const saved=localStorage.getItem(key);if(valid(saved))preference=saved;}catch{}
  function apply(){
    const resolved=preference==='system'?(media.matches?'dark':'light'):preference;
    root.dataset.theme=resolved;
    root.dataset.themePreference=preference;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content',resolved==='dark'?'#171d23':'#ffffff');
    const toggle=document.querySelector('#theme-toggle');
    if(toggle){
      const label='显示主题，当前：'+labels[preference];
      toggle.setAttribute('aria-label',label);toggle.title=label;
      document.querySelectorAll('[data-theme-symbol]').forEach(el=>el.toggleAttribute('hidden',el.dataset.themeSymbol!==preference));
      document.querySelectorAll('[data-theme-option]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.themeOption===preference)));
    }
  }
  apply();
  media.addEventListener('change',()=>{if(preference==='system')apply();});
  window.addEventListener('storage',event=>{if(event.key===key){preference=valid(event.newValue)?event.newValue:'system';apply();}});
  function setup(){
    const toggle=document.querySelector('#theme-toggle'),panel=document.querySelector('#theme-panel');
    if(!toggle||!panel)return;
    const close=()=>{panel.hidden=true;toggle.setAttribute('aria-expanded','false');};
    toggle.addEventListener('click',()=>{panel.hidden=!panel.hidden;toggle.setAttribute('aria-expanded',String(!panel.hidden));});
    panel.querySelectorAll('[data-theme-option]').forEach(button=>button.addEventListener('click',()=>{
      preference=button.dataset.themeOption;
      try{localStorage.setItem(key,preference);}catch{}
      apply();close();toggle.focus({preventScroll:true});
    }));
    document.addEventListener('click',event=>{if(!event.target.closest('.theme-switcher'))close();});
    document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!panel.hidden){close();toggle.focus({preventScroll:true});}});
    panel.addEventListener('focusout',event=>{if(!panel.contains(event.relatedTarget)&&event.relatedTarget!==toggle)close();});
    apply();
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',setup,{once:true});else setup();
})();
