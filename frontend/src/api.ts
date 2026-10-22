export type Paper = {id:string;title:string;filename:string;authors:string[];year:string|null;page_count:number;status:'ready'|'processing'|'failed';parser:string;created_at:string;evidence_count:number;experiment_count:number;is_demo:boolean;warnings:string[];file_url:string|null;error:string|null};
export type Evidence = {id:string;paper_id:string;page:number;section:string;text:string;bbox:number[]|null;source_type:'text'|'table'};
export type FieldQuote = {id:string;quote:string};
export type CitationQuote = FieldQuote & {paper_id?:string;page?:number;handle?:string;source_start?:number;source_end?:number;protocol?:'server_bound_fragment'|'legacy_quote'};
export type ClaimCheck = {claim:string;verdict:'supported'|'contradicted'|'insufficient';explanation:string;citation_quotes:CitationQuote[];safety_gaps?:string[];model_verdict?:'supported'|'contradicted';model_explanation?:string};
export type ModelRun = {requested_model:string;response_model?:string;http_status:number;usage?:Record<string,number>};
export type ModelOption = {id:string;enabled:boolean;default:boolean;availability:string;price_notice:string};
export type VerificationDetail = {model_id?:string;model_runs?:ModelRun[];verdict:'supported'|'partial'|'insufficient'|'contradicted';explanation:string;status?:'failed';citation_quotes?:CitationQuote[];claim_checks?:ClaimCheck[];citations?:Evidence[];verification_protocol?:'atomic_checks_v1'|'legacy_exact_quote';safety_gaps?:string[];model_verdict?:string;model_explanation?:string};
export type Experiment = {id:string;paper_id:string;name:string;task?:string|null;task_label_zh?:string;task_verification?:{version:string;status:string;basis:string;scope:string[]};model_id?:string;model_runs?:ModelRun[];paper_title?:string;method:string|null;dataset:string|null;split:string|null;metric:string|null;value:number|null;unit:string|null;sample_size:string|null;conditions:Record<string,string>;evidence_ids:string[];extraction_method:'rule'|'llm'|'demo'|'manual';notes:string[];field_evidence:Record<string,string[]>;evidence_quotes?:Record<string,FieldQuote[]>;source_pages?:number[];review_required?:boolean;manual_edited?:boolean;table_context?:{value_cell_text?:string;uncertainty?:{notation:string;value:string;kind:string};time_annotation?:{value:string;unit:string}};machine_verification?:{version:string;status:'verified'|'abstained';scope:string[];human_reviewed:boolean;unverified_fields:string[];checks:string[];reason?:string;raw_cell?:string;pdf_sha256?:string;source_ids?:string[];page?:number;limitations?:string[]}};
export type ExtractionCandidate = {id:string;paper_id:string;mode:'rule'|'llm'|'manual';status:'pending'|'accepted'|'rejected';created_at:string;experiments:Experiment[];accepted_experiment_ids?:string[];stale?:boolean;reparsed?:boolean;source_evidence?:Evidence[];extraction_scope?:{pages:number[]|null;total_pages:number;max_results:number;truncated:boolean;completeness:'not_assessed'}};
export type PaperDetail = Paper & {evidence:Evidence[];experiments:Experiment[];experiment_revisions?:Record<string,string>;extraction_candidates?:ExtractionCandidate[];latest_extraction_candidate_id?:string};
export type ExperimentHistoryEntry = {id:string;experiment:Experiment;created_at:string;reason:string};
export type Trace = {agent:string;status:'completed'|'skipped'|'failed';summary:string;evidence_ids:string[];duration_ms:number;kind?:string;input?:Record<string,unknown>;output?:Record<string,unknown>};
export type Comparison = {id:string;left_id:string;right_id:string;comparability:'direct'|'conditional'|'incomparable'|'insufficient';verdict:'consistent'|'different'|'conditions_differ'|'potential_conflict'|'insufficient';reasons:string[];differences:string[];evidence_ids:string[];delta:number|null;alignment?:Record<string,{left:unknown;right:unknown;status:'same'|'different'|'missing'}>;conflict_evidence?:(FieldQuote&{paper_id:string;direction:string})[];unit_conversion?:Record<string,unknown>|null};
export type CompareResult = {id:string;papers:Paper[];experiments:Experiment[];comparisons:Comparison[];traces:Trace[];created_at:string;mode:string;warnings?:string[];evidence?:Evidence[];stale?:boolean;stale_reason?:string};
export type QAResult = {model_id?:string;model_runs?:ModelRun[];answer:string;mode:string;citations:Evidence[];traces:Trace[];warnings:string[];verdict:'evidence_found'|'insufficient'|'unverified';citation_quotes?:CitationQuote[];assertion_evidence?:{text:string;citation_quotes:CitationQuote[]}[];withheld_assertions?:{text:string;reasons:string[];citation_quotes:CitationQuote[]}[];verification?:VerificationDetail;retrieval?:{matched:number;selected:number;total:number;coverage_limited:boolean;papers_covered:number;papers_selected:number}};
export type VerifyResult = VerificationDetail & {mode:string;citations:Evidence[];traces:Trace[];warnings:string[]};
export type Health = {status:string;version:string;providers:{llm_configured:boolean;langgraph_available?:boolean;llm_status?:'unconfigured'|'untested'|'available'|'failed'}};
export type ReportSummary = {id:string;created_at:string;papers:Paper[];experiment_count:number;comparison_count:number;stale:boolean;stale_reason?:string};
export type Engine = 'local'|'llm';
export type VerifyEngine = Engine;
export function engineIsAvailable(engine:Engine,providers?:Health['providers']):boolean{
 if(engine==='local')return true;
 return !!providers?.llm_configured;
}
export function defaultEngine(providers?:Health['providers']):Engine{
 return providers?.llm_configured?'llm':'local';
}
export class ApiError extends Error {
 constructor(message:string, public status:number, public code?:string){super(message);this.name='ApiError';}
}
export type RequestOptions = RequestInit & {timeoutMs?:number;responseType?:'json'|'blob'|'probe'};
export function isCancelled(error:unknown){return error instanceof Error&&error.name==='AbortError';}
export async function request<T>(path:string,options:RequestOptions={}):Promise<T>{
 const {timeoutMs=30000,responseType='json',signal,...init}=options;
 const controller=new AbortController();
 let timedOut=false;
 const cancel=()=>controller.abort(signal?.reason);
 if(signal?.aborted)cancel();else signal?.addEventListener('abort',cancel,{once:true});
 const timer=setTimeout(()=>{timedOut=true;controller.abort();},timeoutMs);
 try {
  const headers=new Headers(init.headers);
  if(!(init.body instanceof FormData)&&init.body!==undefined&&!headers.has('Content-Type'))headers.set('Content-Type','application/json');
  const response=await fetch(`/api${path}`,{...init,signal:controller.signal,headers});
  if(!response.ok){
   let message=`请求失败（HTTP ${response.status}）`,code:string|undefined;
   try {
    const data=await response.json();
    const detail=data.detail??data.error;
    if(typeof detail==='string')message=detail;
    else if(Array.isArray(detail))message=detail.map((item:{loc?:unknown[];msg?:string})=>`${item.loc?.slice(1).join(' / ')||'输入参数'}：${item.msg??'格式无效'}`).join('；');
    else if(detail&&typeof detail==='object'){message=detail.message??detail.error??message;code=detail.code;}
    code=code??data.code;
   }catch{/* Preserve HTTP status when the response is not JSON. */}
   throw new ApiError(message,response.status,code);
  }
  let data:unknown;
  try{if(responseType==='probe'){await response.body?.cancel();data=true;}else data=response.status===204?undefined:responseType==='blob'?await response.blob():await response.json();}
  catch(error){if(controller.signal.aborted)throw error;throw new ApiError('服务返回了无法读取的响应。请确认前后端版本一致，再刷新重试。',response.status,'invalid_response');}
  if(controller.signal.aborted)throw new DOMException('请求已取消','AbortError');
  return data as T;
 }catch(error){
  if(timedOut)throw new ApiError(`请求在 ${Math.round(timeoutMs/1000)} 秒后超时。请检查服务状态后重试；服务器可能仍在处理本次任务。`,408,'timeout');
  if(controller.signal.aborted)throw new DOMException('请求已取消','AbortError');
  if(error instanceof TypeError)throw new ApiError('无法连接本地服务。请确认后端正在运行，再重试。',0,'network');
  throw error;
 }finally{clearTimeout(timer);signal?.removeEventListener('abort',cancel);}
}
export const post=<T,>(path:string,body:unknown,options:RequestOptions={})=>request<T>(path,{...options,method:'POST',body:JSON.stringify(body)});
export function paperPagePreviewUrl(paperId:string,pageNumber:number):string|null{
 return paperId&&Number.isSafeInteger(pageNumber)&&pageNumber>0?`/api/papers/${encodeURIComponent(paperId)}/pages/${pageNumber}/preview`:null;
}
export function qaDisplayAnswer(result:QAResult):string{
 if(!result.withheld_assertions?.length)return result.answer;
 const accepted=result.assertion_evidence??[];
 return accepted.length?accepted.map(item=>item.text).join('\n\n'):'本次未形成可展示的事实回答。部分模型断言未通过来源检查，已暂缓，请核对原表或补充证据。';
}

export const DEFAULT_MODEL_ID='gpt-4o-mini';
export function defaultModelId(models?:ModelOption[]):string{
 // Default remains explicit even if unavailable. Never choose another enabled model.
 return models?.find(model=>model.default)?.id??DEFAULT_MODEL_ID;
}
