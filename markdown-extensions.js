let library,serial=Promise.resolve(),sequence=0;
function mermaidLibrary(){return library??=import('./vendor--mermaid.js?v=6a74797e2dfd').then(module=>module.default).catch(error=>{library=null;throw error;});}
// Also used by sandboxed private previews: render in this module's document,
// then transfer only the sanitized SVG into the preview document.
export function enhanceMarkdown(root,{eager=false}={}){
 const doc=root.ownerDocument,win=doc.defaultView,graphs=[];let disposed=false;
 const dark=()=>doc.documentElement.dataset.theme==='dark'||doc.documentElement.style.colorScheme==='dark';
 for(const pre of root.querySelectorAll('pre')){
  const code=pre.querySelector(':scope > code.language-mermaid');if(!code||pre.closest('.mermaid-source'))continue;
  const figure=doc.createElement('figure');figure.className='mermaid-figure';
  const canvas=doc.createElement('div');canvas.className='mermaid-canvas';canvas.setAttribute('role','img');canvas.setAttribute('aria-label','Mermaid 图表');canvas.setAttribute('aria-busy','true');canvas.dataset.copyIgnore='';canvas.textContent='图表加载中…';
  const details=doc.createElement('details');details.className='mermaid-source';const summary=doc.createElement('summary');summary.textContent='Mermaid 源码';summary.dataset.copyIgnore='';details.append(summary);
  pre.before(figure);figure.append(canvas,details);details.append(pre);
  const graph={figure,canvas,details,source:code.textContent,state:'idle',theme:'',generation:0};graphs.push(graph);
 }
 function visible(graph){
  if(eager)return true;
  for(let parent=graph.figure.parentElement;parent&&parent!==root;parent=parent.parentElement)if(parent.tagName==='DETAILS'&&!parent.open)return false;
  const rect=graph.figure.getBoundingClientRect();return rect.width>0&&rect.bottom>=-500&&rect.top<=(win?.innerHeight||900)+500;
 }
 function render(graph,force=false){
  const theme=dark()?'dark':'default';if(disposed||!visible(graph)||graph.state==='loading'||(!force&&graph.state==='error')||(graph.state==='ready'&&graph.theme===theme))return;
  graph.state='loading';const generation=++graph.generation;graph.canvas.setAttribute('aria-busy','true');
  serial=serial.catch(()=>{}).then(async()=>{
   if(disposed)return;let scratch;
   try{
    const mermaid=await mermaidLibrary();if(disposed)return;
    mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme,fontFamily:'system-ui, "Microsoft YaHei", sans-serif',suppressErrorRendering:true,maxTextSize:100000,htmlLabels:false,flowchart:{htmlLabels:false},secure:['secure','securityLevel','startOnLoad','maxTextSize','suppressErrorRendering','htmlLabels','flowchart','theme','themeCSS','themeVariables','dompurifyConfig']});
    scratch=document.createElement('div');scratch.style.cssText='position:fixed;left:-100000px;top:0;visibility:hidden;width:'+Math.max(320,Math.min(1200,root.clientWidth||860))+'px';document.body.append(scratch);
    const result=await mermaid.render('mermaid-diagram-'+(++sequence),graph.source,scratch);
    if(disposed||generation!==graph.generation)return;
    graph.canvas.innerHTML=result.svg;graph.canvas.querySelector('svg')?.setAttribute('aria-label','Mermaid 图表');graph.canvas.removeAttribute('data-copy-ignore');graph.state='ready';graph.theme=theme;
   }catch{
    if(disposed)return;graph.state='error';graph.canvas.replaceChildren();const message=doc.createElement('span');message.textContent='图表暂未渲染，请查看下方源码。';const retry=doc.createElement('button');retry.type='button';retry.textContent='重试';retry.addEventListener('click',()=>render(graph,true));graph.canvas.append(message,retry);graph.canvas.dataset.copyIgnore='';graph.details.open=true;
   }finally{scratch?.remove();graph.canvas.setAttribute('aria-busy','false');if(graph.state==='ready'&&graph.theme!==(dark()?'dark':'default'))render(graph);}
  });
 }
 const observer=win?.IntersectionObserver?new win.IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){const graph=graphs.find(g=>g.figure===entry.target);if(graph)render(graph);}},{rootMargin:'500px'}):null;
 const check=()=>graphs.forEach(graph=>render(graph));for(const graph of graphs){observer?.observe(graph.figure);if(eager||!observer)render(graph);}
 root.addEventListener('toggle',check,true);
 const themeObserver=win?.MutationObserver?new win.MutationObserver(check):null;themeObserver?.observe(doc.documentElement,{attributes:true,attributeFilter:['data-theme','style']});
 function footnote(event){const link=event.target.closest?.('a[href^="#fn"]');if(!link||!root.contains(link)||!link.closest('.footnote-ref,.footnotes'))return;const target=doc.getElementById(decodeURIComponent(link.getAttribute('href').slice(1)));if(!target)return;event.preventDefault();target.scrollIntoView({block:'center',behavior:'instant'});target.setAttribute('tabindex','-1');target.focus({preventScroll:true});}
 root.addEventListener('click',footnote);
 return()=>{disposed=true;observer?.disconnect();themeObserver?.disconnect();root.removeEventListener('toggle',check,true);root.removeEventListener('click',footnote);};
}
