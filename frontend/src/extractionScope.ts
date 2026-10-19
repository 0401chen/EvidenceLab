/** Validate a bounded selection before starting a potentially paid extraction. */
export function extractionRequestScope(pageInput:string,resultInput:string,totalPages:number):{pages:number[]|null;max_results:number}{
 const resultText=resultInput.trim();
 const maxResults=Number(resultText);
 if(!/^\d+$/.test(resultText)||!Number.isSafeInteger(maxResults)||maxResults<1||maxResults>30)throw new Error('最多结果数必须是 1 至 30 的整数。');
 const pageText=pageInput.trim();
 if(!pageText)return {pages:null,max_results:maxResults};
 if(!Number.isSafeInteger(totalPages)||totalPages<1)throw new Error('文献页数不可用，请刷新详情后选择页码。');
 const pages=new Set<number>();
 for(const token of pageText.replaceAll('，',',').split(',')){
  const match=/^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$/.exec(token);
  if(!match)throw new Error('页码请使用 1,3-5 格式，空白表示全文。');
  const first=Number(match[1]),last=Number(match[2]??match[1]);
  if(!Number.isSafeInteger(first)||!Number.isSafeInteger(last)||first<1||last<first||last>totalPages)throw new Error(`页码须在 1 至 ${totalPages} 内，范围起点不能大于终点。`);
  if(last-first+1>25)throw new Error('一次最多选择 25 页，请缩小提取范围。');
  for(let page=first;page<=last;page++)pages.add(page);
  if(pages.size>25)throw new Error('一次最多选择 25 页，请缩小提取范围。');
 }
 return {pages:[...pages].sort((left,right)=>left-right),max_results:maxResults};
}
