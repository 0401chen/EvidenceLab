import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {afterEach, test} from 'node:test';
import ts from 'typescript';

// Compile with the project's existing TypeScript dependency; no extra test runtime.
const source=await readFile(new URL('../src/api.ts',import.meta.url),'utf8');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const {ApiError,isCancelled,post,qaDisplayAnswer,request}=await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
const originalFetch=globalThis.fetch;
afterEach(()=>{globalThis.fetch=originalFetch;});

test('withheld numeric assertions cannot be displayed as facts through unchecked answer text',()=>{
 const result={answer:'An unsafe 99% claim.\n\nA retained 84% claim.',withheld_assertions:[{text:'An unsafe 99% claim.',reasons:['protocol missing'],citation_quotes:[]}],assertion_evidence:[{text:'A retained 84% claim.',citation_quotes:[]}]};
 assert.equal(qaDisplayAnswer(result),'A retained 84% claim.');
});
test('an entirely withheld answer stays incomplete even with stale or missing accepted-assertion data',()=>{
 const result={answer:'An unsafe 99% claim.',withheld_assertions:[{text:'An unsafe 99% claim.',reasons:['wrong benchmark'],citation_quotes:[]}]};
 assert.doesNotMatch(qaDisplayAnswer(result),/99/);
 assert.match(qaDisplayAnswer(result),/暂缓/);
 assert.doesNotMatch(qaDisplayAnswer({...result,assertion_evidence:[]}),/99/);
});
test('legacy and local answers without withheld assertions preserve their existing display',()=>{
 assert.equal(qaDisplayAnswer({answer:'Candidate evidence only.'}),'Candidate evidence only.');
});

test('reads are scoped to the local API and preserve Headers objects',async()=>{
 globalThis.fetch=async(url,init)=>{
  assert.equal(url,'/api/papers');
  assert.equal(init.headers.get('X-Request-Context'),'review');
  assert.equal(init.headers.has('Content-Type'),false);
  return Response.json([{id:'paper-a'}]);
 };
 assert.deepEqual(await request('/papers',{headers:new Headers({'X-Request-Context':'review'})}),[{id:'paper-a'}]);
});

test('JSON posts preserve the explicit paper scope and engine',async()=>{
 globalThis.fetch=async(_url,init)=>{
  assert.equal(init.method,'POST');
  assert.equal(init.headers.get('Content-Type'),'application/json');
  assert.deepEqual(JSON.parse(init.body),{paper_ids:['paper-a'],question:'dataset?',engine:'local'});
  return Response.json({answer:'candidate evidence'});
 };
 assert.equal((await post('/qa',{paper_ids:['paper-a'],question:'dataset?',engine:'local'})).answer,'candidate evidence');
});

test('trial QA and verification preserve the chosen engine without replacing normal model requests',async()=>{
 const calls=[];
 globalThis.fetch=async(url,init)=>{
  const body=JSON.parse(init.body);calls.push({url,body});
  return Response.json({mode:body.engine,answer:'mock response'});
 };
 for(const path of ['/qa','/verify']){
  for(const engine of ['llm','llm_tasks']){
   const body={paper_ids:['paper-a'],engine,...(path==='/qa'?{question:'conditions?'}:{claim:'A testable claim.'})};
   assert.equal((await post(path,body)).mode,engine);
   assert.deepEqual(calls.at(-1),{url:`/api${path}`,body});
  }
 }
 assert.equal(calls.length,4);
});

test('multipart uploads leave the browser free to add the boundary',async()=>{
 const body=new FormData();body.append('files',new Blob(['research']),'paper.txt');
 globalThis.fetch=async(_url,init)=>{assert.equal(init.body,body);assert.equal(init.headers.has('Content-Type'),false);return Response.json({papers:[]});};
 await request('/papers/upload',{method:'POST',body});
});

test('validation failures show field explanations with an HTTP status',async()=>{
 globalThis.fetch=async()=>Response.json({detail:[{loc:['body','paper_ids'],msg:'List should have at most 30 items'}]},{status:422});
 await assert.rejects(request('/compare'),err=>err instanceof ApiError&&err.status===422&&err.message==='paper_ids：List should have at most 30 items');
});

test('provider failure messages retain the structured error code',async()=>{
 globalThis.fetch=async()=>Response.json({detail:{message:'模型连接失败',code:'connection'}},{status:502});
 await assert.rejects(request('/qa'),err=>err instanceof ApiError&&err.status===502&&err.code==='connection'&&err.message==='模型连接失败');
});

test('invalid JSON error responses retain HTTP context',async()=>{
 globalThis.fetch=async()=>new Response('upstream unavailable',{status:503});
 await assert.rejects(request('/health'),err=>err.status===503&&err.message.includes('HTTP 503'));
});

test('network failures include a recovery instruction',async()=>{
 globalThis.fetch=async()=>{throw new TypeError('Failed to fetch');};
 await assert.rejects(request('/papers'),err=>err.code==='network'&&err.message.includes('后端正在运行'));
});

test('request deadlines abort an unresponsive fetch and release the caller',async()=>{
 let aborted=false;
 globalThis.fetch=(_url,init)=>new Promise((_resolve,reject)=>init.signal.addEventListener('abort',()=>{aborted=true;reject(new DOMException('aborted','AbortError'));},{once:true}));
 await assert.rejects(request('/qa',{timeoutMs:10}),err=>err.code==='timeout'&&err.status===408&&err.message.includes('服务器可能仍在处理'));
 assert.equal(aborted,true);
});

test('user cancellation stays distinguishable from timeout or provider failure',async()=>{
 const controller=new AbortController();
 globalThis.fetch=(_url,init)=>new Promise((_resolve,reject)=>init.signal.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError')),{once:true}));
 const pending=request('/qa',{signal:controller.signal});controller.abort();
 await assert.rejects(pending,err=>isCancelled(err)&&!(err instanceof ApiError));
});

test('cancellation also discards a response that arrives too late',async()=>{
 const controller=new AbortController();
 globalThis.fetch=async()=>{controller.abort();return Response.json({answer:'late result'});};
 await assert.rejects(request('/qa',{signal:controller.signal}),isCancelled);
});

test('exports support binary downloads and surface stale-report errors',async()=>{
 globalThis.fetch=async()=>new Response('# Report',{headers:{'Content-Type':'text/markdown'}});
 assert.equal(await (await request('/reports/r/export?format=markdown',{responseType:'blob'})).text(),'# Report');
 globalThis.fetch=async()=>Response.json({detail:'报告已过期，请重新生成'},{status:409});
 await assert.rejects(request('/reports/r/export',{responseType:'blob'}),err=>err.status===409&&err.message.includes('已过期'));
});

test('empty successful responses do not fail JSON parsing',async()=>{
 globalThis.fetch=async()=>new Response(null,{status:204});
 assert.equal(await request('/papers/a',{method:'DELETE'}),undefined);
});

test('unexpected HTML is an explicit response-format failure',async()=>{
 globalThis.fetch=async()=>new Response('<html>frontend fallback</html>',{status:200});
 await assert.rejects(request('/papers'),err=>err.code==='invalid_response'&&err.message.includes('前后端版本一致'));
});

test('original-file probes cancel the response stream without reading the PDF',async()=>{
 let cancelled=false;
 globalThis.fetch=async(_url,init)=>{
  assert.equal(init.headers.get('Range'),'bytes=0-0');
  return new Response(new ReadableStream({cancel(){cancelled=true;}}),{status:206});
 };
 assert.equal(await request('/papers/a/file',{headers:{Range:'bytes=0-0'},responseType:'probe'}),true);
 assert.equal(cancelled,true);
});

test('stale edit submissions preserve the version token and never auto-retry',async()=>{
 let calls=0;
 const revision='a'.repeat(64);
 globalThis.fetch=async(_url,init)=>{
  calls++;
  assert.equal(init.method,'PATCH');
  assert.equal(JSON.parse(init.body).expected_revision,revision);
  return Response.json({detail:'实验已修改，请保留输入并重新载入'},{status:409});
 };
 await assert.rejects(request('/experiments/e',{method:'PATCH',body:JSON.stringify({expected_revision:revision,name:'edited'})}),err=>err.status===409);
 assert.equal(calls,1);
});
