import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import ts from 'typescript';

const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
const asData=code=>`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`;
const apiCode=compile(await readFile(new URL('../src/api.ts',import.meta.url),'utf8'));
const apiURL=asData(apiCode);
const scopeURL=asData(compile(await readFile(new URL('../src/extractionScope.ts',import.meta.url),'utf8')));
const appCode=compile(await readFile(new URL('../src/App.tsx',import.meta.url),'utf8')).replace(/from (['"])([^'"]+)\1/g,(_whole,_quote,specifier)=>`from ${JSON.stringify(specifier==='./api'?apiURL:specifier==='./extractionScope'?scopeURL:import.meta.resolve(specifier))}`);
const {ResearchEngineOptions,TaskRequirements,engineNames}=await import(asData(appCode));
const {distinctTaskObservations,engineIsAvailable,hasTaskRequirements,taskCoverageLabel,taskFieldBindings}=await import(apiURL);
const quote={id:'source-a',paper_id:'paper-a',page:2,quote:'AlphaNet achieved 41 frames per second with no batching.',handle:'F1',protocol:'server_bound_fragment'};
const source={id:'source-a',paper_id:'paper-a',page:2,section:'Results',text:quote.quote,bbox:null,source_type:'text'};
const observation={fields:{method:{text:'AlphaNet',evidence:[quote]},speed:{text:'41 frames per second',evidence:[quote]}}};
const row={id:'requirement-a',request:'分别给出速度与批处理条件',subject:'AlphaNet',fields:['speed','batching'],status:'partial',observations:[observation],missing_fields:['batching'],field_comparisons:[],explanation:'批处理来源仍待确认。'};
const structured={task_protocol:'requirements_observations_v1',task_rows:[row],coverage:{planned:1,addressed:1,answered:0,missing:1,unaddressed:0,complete:true,source_complete:false}};
const render=result=>renderToStaticMarkup(React.createElement(TaskRequirements,{result,citations:[source],papers:[{id:'paper-a',title:'A real source'}],onOpen(){}}));

test('QA and verification expose the trial engine explicitly while retaining a local default and the normal model option',()=>{
 const providers={llm_configured:true};
 for(const purpose of ['qa','verify']){
  const html=renderToStaticMarkup(React.createElement('select',{defaultValue:'local'},React.createElement(ResearchEngineOptions,{purpose,providers})));
  assert.match(html,/<option value="local" selected="">/);
  assert.match(html,/<option value="llm">/);
  assert.match(html,/<option value="llm_tasks">逐项科研试验 · 待验证<\/option>/);
  assert.doesNotMatch(html,/推荐|可靠|自动认证/);
  assert.deepEqual([...html.matchAll(/<option value="([^"]+)"/g)].map(match=>match[1]),['local','llm','llm_tasks']);
 }
 assert.equal(engineNames.llm_tasks,'逐项科研试验 · 待验证');
 assert.equal(engineNames.llm,'大模型增强问答');
});

test('both model engines require configuration and the trial stays disabled when unconfigured',()=>{
 assert.equal(engineIsAvailable('local'),true);
 for(const engine of ['llm','llm_tasks']){
  assert.equal(engineIsAvailable(engine),false);
  assert.equal(engineIsAvailable(engine,{llm_configured:false}),false);
  assert.equal(engineIsAvailable(engine,{llm_configured:true}),true);
 }
 for(const purpose of ['qa','verify']){
  const html=renderToStaticMarkup(React.createElement(ResearchEngineOptions,{purpose}));
  assert.match(html,/<option value="llm_tasks" disabled="">逐项科研试验 · 待验证 · 未配置<\/option>/);
 }
});

test('all subitems addressed is visibly separate from missing source fields and scientific correctness',()=>{
 const html=render(structured);
 assert.match(html,/子项均已回答或明确不足/);
 assert.match(html,/缺失来源字段/);
 assert.match(html,/未处理子项/);
 assert.match(html,/仍缺所需字段的直接来源/);
 assert.match(html,/不代表全文完整、来源充分或科研判断正确/);
 assert.match(html,/未找到直接来源：批处理/);
 assert.match(html,/问题拆分与字段匹配仍需人工复核/);
});

test('field evidence opens by exact original binding and never invents an unloaded source',()=>{
 const field=Object.freeze({text:'41 frames per second',evidence:Object.freeze([Object.freeze(quote)])});
 const result=taskFieldBindings(field,[source]);
 assert.equal(result.quotes[0],quote);
 assert.equal(result.sources[0],source);
 assert.equal(result.quotes[0].quote,'AlphaNet achieved 41 frames per second with no batching.');
 assert.deepEqual(taskFieldBindings(field,[]).sources,[]);
 const html=renderToStaticMarkup(React.createElement(TaskRequirements,{result:structured,citations:[],papers:[],onOpen(){throw new Error('Must not guess a source');}}));
 assert.match(html,/来源片段尚未载入/);
 assert.doesNotMatch(html,/查看处理速度的字段原文/);
});

test('only exact duplicate observations collapse; equal values with different ownership or sources survive',()=>{
 const duplicate=structuredClone(observation);
 const anotherModel={fields:{...observation.fields,method:{text:'BetaNet',evidence:[quote]}}};
 const anotherSource={fields:{method:{text:'AlphaNet',evidence:[{...quote,id:'source-b'}]},speed:observation.fields.speed}};
 const unique=distinctTaskObservations([observation,duplicate,anotherModel,anotherSource]);
 assert.equal(unique.length,3);
 assert.equal(unique[0],observation);
 assert.equal(unique[1].fields.method.text,'BetaNet');
});

test('server field comparisons and verdict are rendered without frontend scientific inference',()=>{
 const result={...structured,task_rows:[{...row,status:'answered',missing_fields:[],verdict:'contradicted',field_comparisons:[{field:'speed',expected:'96 fps',observed:'41 fps',relation:'conflict'},{field:'hardware',expected:'GPU',observed:null,relation:'unknown'}]}]};
 const html=render(result);
 assert.match(html,/96 fps/);
 assert.match(html,/41 fps/);
 assert.match(html,/字段关系聚合 · 不符（待复核）/);
 assert.match(html,/不符/);
 assert.match(html,/未知/);
 assert.match(html,/查看处理速度的字段原文/);
 assert.match(html,/第 2 页/);
});

test('old responses retain the legacy view and task coverage is never guessed from arbitrary rows',()=>{
 assert.equal(hasTaskRequirements({task_rows:[row]}),false);
 assert.equal(render({task_rows:[row]}),'');
 assert.equal(render({verdict:'supported',answer:'An old response.'}),'');
 assert.equal(taskCoverageLabel(),'尚未返回子项覆盖统计');
 const html=render({...structured,coverage:undefined});
 assert.match(html,/尚未返回覆盖统计/);
});


test('semantic condition mapping discloses model judgment and its reason instead of implying literal proof',()=>{
 const result={...structured,task_rows:[{...row,field_comparisons:[{field:'batching',expected:'不使用批处理',observed:'no batch processing',relation:'match',basis:'semantic_model',reason:'中文条件与英文条件语义对应，需结合原句确认。'}]}]};
 const html=render(result);
 assert.match(html,/条件语义映射（模型判断，需复核）/);
 assert.match(html,/批处理 · 模型映射说明/);
 assert.match(html,/中文条件与英文条件语义对应，需结合原句确认。/);
 assert.match(html,/no batch processing/);
 const literal=render({...result,task_rows:[{...row,field_comparisons:[{field:'speed',expected:'41 fps',observed:'41 fps',relation:'match',basis:'literal'}]}]});
 assert.doesNotMatch(literal,/条件语义映射（模型判断，需复核）/);
 assert.doesNotMatch(literal,/task-semantic-notes/);
});


test('withheld observations are audit-only and cannot appear as factual source values or inflate accepted counts',()=>{
 const accepted={...observation,status:'accepted'};
 const withheld={fields:{method:observation.fields.method,speed:{text:'999 fps — wrong model ownership',evidence:[quote]}},status:'withheld',reasons:['模型配置归属检查未通过。']};
 const result={...structured,task_rows:[{...row,observations:[accepted,withheld]}]};
 const html=render(result);
 assert.match(html,/来源字段 · 1 条测量或观察记录/);
 assert.match(html,/归属检查未通过，未用于回答\/结论 · 1 条暂缓观察/);
 assert.match(html,/模型配置归属检查未通过。/);
 assert.match(html,/模型返回值已暂缓，不作为事实展示/);
 assert.match(html,/查看暂缓观察中处理速度的字段原文/);
 assert.doesNotMatch(html,/999 fps/);
 assert.doesNotMatch(html,/wrong model ownership/);
 const onlyWithheld=render({...result,task_rows:[{...row,observations:[withheld]}]});
 assert.doesNotMatch(onlyWithheld,/来源字段 · 1 条测量或观察记录/);
 assert.match(onlyWithheld,/没有通过归属检查的来源观察记录/);
 assert.doesNotMatch(onlyWithheld,/999 fps/);
});

test('accepted and withheld versions never deduplicate into one another',()=>{
 const accepted={...observation,status:'accepted'};
 const withheld={...observation,status:'withheld',reasons:['configuration unresolved']};
 const unique=distinctTaskObservations([accepted,withheld,structuredClone(withheld)]);
 assert.equal(unique.length,2);
 assert.equal(unique[0].status,'accepted');
 assert.equal(unique[1].status,'withheld');
});
