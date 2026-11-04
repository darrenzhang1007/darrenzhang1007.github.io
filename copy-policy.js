// Public-reader copy controls discourage bulk reuse; they are not access control.
const ignored='button,script,style,svg,.code-gutter,.code-toolbar,.code-toggle,.heading-anchor,.katex-mathml,[data-copy-ignore]';
const blocks=new Set(['P','DIV','SECTION','ARTICLE','H1','H2','H3','H4','H5','H6','LI','UL','OL','BLOCKQUOTE','TABLE','TR']);
const escapeHtml=text=>String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function clean(fragment){for(const el of fragment.querySelectorAll(ignored))el.remove();return fragment;}
function plainText(node){
  if(node.nodeType===3)return node.textContent;
  if(node.nodeName==='BR')return '\n';
  if(node.nodeName==='PRE')return '\n'+(node.querySelector('code')||node).textContent+'\n';
  const text=[...node.childNodes].map(plainText).join(node.nodeName==='TR'?'\t':'');
  return blocks.has(node.nodeName)?'\n'+text+'\n':text;
}
const characterCount=text=>text.replace(/\s/g,'').length;
const elementOf=node=>node.nodeType===1?node:node.parentElement;

export function setupCopyPolicy(root,{author='DarrenZhang',url,notify=()=>{}}={}){
  const doc=root.ownerDocument,canonical=new URL(url||doc.location.href,doc.location.href);canonical.search='';canonical.hash='';
  const sourceUrl=canonical.href;
  function onCopy(event){
    if(event.defaultPrevented||!event.clipboardData||elementOf(doc.activeElement)?.closest('input,textarea,[contenteditable="true"]'))return;
    const selection=doc.getSelection();if(!selection||selection.isCollapsed)return;
    const ranges=Array.from({length:selection.rangeCount},(_,i)=>selection.getRangeAt(i));
    const selected=ranges.filter(range=>range.intersectsNode(root));if(!selected.length)return;
    // Native code selections and the existing copy buttons stay byte-for-byte code.
    if(selected.length===ranges.length&&selected.every(range=>{const pre=elementOf(range.commonAncestorContainer)?.closest('pre');return pre&&root.contains(pre);}))return;
    const boundary=doc.createRange();boundary.selectNodeContents(root);
    const fragment=doc.createDocumentFragment();
    for(const range of selected){
      const clipped=range.cloneRange();
      if(clipped.compareBoundaryPoints(0,boundary)<0)clipped.setStart(boundary.startContainer,boundary.startOffset);
      if(clipped.compareBoundaryPoints(2,boundary)>0)clipped.setEnd(boundary.endContainer,boundary.endOffset);
      fragment.append(clipped.cloneContents());
    }
    clean(fragment);const text=plainText(fragment).replace(/\n{3,}/g,'\n\n').trim();if(!text)return;
    const whole=clean(root.cloneNode(true)),total=characterCount(plainText(whole));
    event.preventDefault();
    if(total&&characterCount(text)>=total*.9){notify('已限制整篇复制。可摘录部分正文并保留来源，代码块可单独复制。');return;}
    const attribution='作者：'+author+'\n原文：'+sourceUrl;
    event.clipboardData.setData('text/plain',text+'\n\n——\n'+attribution);
    const container=doc.createElement('div');container.append(fragment);
    event.clipboardData.setData('text/html',container.innerHTML+'<hr><p>作者：'+escapeHtml(author)+'<br>原文：<a href="'+escapeHtml(sourceUrl)+'">'+escapeHtml(sourceUrl)+'</a></p>');
    notify('已复制摘录，并附上作者与原文链接');
  }
  doc.addEventListener('copy',onCopy);
  return()=>doc.removeEventListener('copy',onCopy);
}
