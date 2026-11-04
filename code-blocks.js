import {highlightCode} from './code-highlight.js?v=a3c99b87b884';
export const COLLAPSE_AFTER=32,PREVIEW_LINES=16;
const labels={python:'Python',py:'Python',triton:'Triton / Python',bash:'Bash',sh:'Shell',shell:'Shell',zsh:'Zsh',cpp:'C++','c++':'C++',cxx:'C++',cuda:'CUDA',cu:'CUDA',c:'C',javascript:'JavaScript',js:'JavaScript',typescript:'TypeScript',ts:'TypeScript',json:'JSON',yaml:'YAML',yml:'YAML',toml:'TOML',ini:'INI',diff:'Diff',sql:'SQL',dockerfile:'Dockerfile',makefile:'Makefile',rust:'Rust',rs:'Rust',html:'HTML',xml:'XML',mermaid:'Mermaid',text:'纯文本',plaintext:'纯文本',txt:'纯文本'};
export const codeLines=source=>source?source.replace(/\n$/,'').split('\n').length:0;
const copyIcon='<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg>';
export function enhanceCodeBlocks(root,{notify=()=>{}}={}){
 const doc=root.ownerDocument,timers=new Set();
 for(const [index,pre]of [...root.querySelectorAll('pre')].entries()){
  if(pre.closest('.code-frame')||pre.closest('.katex,.arithmatex,svg'))continue;
  const code=pre.querySelector('code')||pre,source=code.textContent||'',lines=codeLines(source),language=(code.dataset.language||[...code.classList].find(c=>c.startsWith('language-'))?.slice(9)||'text').toLowerCase(),filename=(code.dataset.filename||pre.dataset.filename||'').slice(0,240),label=labels[language]||language;
  const marked=highlightCode(source,language);if(marked!==null){code.innerHTML=marked;code.classList.add('hljs');}
  const frame=doc.createElement('div');frame.className='code-frame';const toolbar=doc.createElement('div');toolbar.className='code-toolbar';
  const meta=doc.createElement('div');meta.className='code-meta';const badge=doc.createElement('span');badge.className='code-language';badge.textContent=label;meta.append(badge);
  if(filename){const title=doc.createElement('span');title.className='code-filename';title.textContent=filename;title.title=filename;meta.append(title);}
  const count=doc.createElement('span');count.className='code-count';count.textContent=lines+' 行';meta.append(count);
  const copy=doc.createElement('button');copy.type='button';copy.className='code-copy';copy.innerHTML=copyIcon+'<span>复制</span>';copy.setAttribute('aria-label','复制代码：'+(filename||label)+'，第 '+(index+1)+' 个代码块');copy.title='复制完整代码';toolbar.append(meta,copy);
  const status=doc.createElement('span');status.className='code-copy-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');toolbar.append(status);
  pre.before(frame);frame.append(toolbar,pre);pre.classList.add('reader-code');if(!pre.hasAttribute('tabindex'))pre.tabIndex=0;pre.setAttribute('aria-label',(filename||label)+' 代码，共 '+lines+' 行');
  let resetTimer;copy.addEventListener('click',async()=>{if(resetTimer){clearTimeout(resetTimer);timers.delete(resetTimer);}try{if(!navigator.clipboard?.writeText)throw Error();await navigator.clipboard.writeText(source);copy.classList.add('copied');copy.querySelector('span').textContent='已复制';status.textContent='完整代码已复制';}catch{const range=doc.createRange();range.selectNodeContents(code);const selection=doc.defaultView?.getSelection();selection?.removeAllRanges();selection?.addRange(range);status.textContent='已选中代码，请按 Ctrl+C 或 ⌘C 复制';notify(status.textContent);}resetTimer=setTimeout(()=>{copy.classList.remove('copied');copy.querySelector('span').textContent='复制';status.textContent='';timers.delete(resetTimer);},2200);timers.add(resetTimer);});
  if(lines>COLLAPSE_AFTER){const toggle=doc.createElement('button');toggle.type='button';toggle.className='code-expand';let expanded=false;const update=()=>{frame.classList.toggle('code-collapsed',!expanded);toggle.setAttribute('aria-expanded',String(expanded));toggle.textContent=expanded?'收起长代码':'展开全部 '+lines+' 行（当前显示前 '+PREVIEW_LINES+' 行）';};update();frame.append(toggle);toggle.addEventListener('click',()=>{const above=frame.getBoundingClientRect().top<0;expanded=!expanded;update();if(!expanded&&above)toolbar.scrollIntoView({block:'start',behavior:'instant'});doc.defaultView?.dispatchEvent(new Event('scroll'));});pre.addEventListener('reader:reveal-code',()=>{if(!expanded){expanded=true;update();}});}
 }
 return()=>{for(const timer of timers)clearTimeout(timer);};
}
