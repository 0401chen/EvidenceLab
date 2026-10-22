import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import React from 'react';
import ts from 'typescript';
const app=await readFile(new URL('../src/App.tsx',import.meta.url),'utf8');
const fragment=app.slice(app.indexOf('export function ModelPicker'),app.indexOf('const operationNames:'));
const compiled=ts.transpileModule(fragment,{compilerOptions:{jsx:ts.JsxEmit.React,module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const module={exports:{}};
new Function('React','exports',compiled)(React,module.exports);
const {ModelPicker}=module.exports;
const models=[{id:'gpt-4o-mini',enabled:true,default:true},{id:'deepseek-flash',enabled:false,default:false}];
test('model picker exposes actual default and disables server-unapproved IDs',()=>{
 const element=ModelPicker({label:'问答模型',value:'',onChange:()=>{},models});
 const select=element.props.children[1];
 assert.equal(select.props['aria-label'],'问答模型');
 assert.match(select.props.children[0].props.children.join(''),/gpt-4o-mini/);
 const options=select.props.children[1];
 assert.equal(options[0].props.disabled,false);
 assert.equal(options[1].props.disabled,true);
 assert.match(options[1].props.children.join(''),/未启用/);
});
test('changing selection only updates state and cannot initiate a fetch',()=>{
 let selected;const oldFetch=globalThis.fetch;
 globalThis.fetch=()=>{throw new Error('Selection must not access network');};
 try{
  const element=ModelPicker({label:'核验模型',value:'',onChange:value=>selected=value,models});
  element.props.children[1].props.onChange({target:{value:'gpt-4o-mini'}});
  assert.equal(selected,'gpt-4o-mini');
 }finally{globalThis.fetch=oldFetch;}
});
