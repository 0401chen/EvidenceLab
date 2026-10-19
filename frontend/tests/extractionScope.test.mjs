import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import ts from 'typescript';

const source=await readFile(new URL('../src/extractionScope.ts',import.meta.url),'utf8');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const {extractionRequestScope}=await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('blank page selection explicitly requests the entire document with bounded results',()=>{
 assert.deepEqual(extractionRequestScope(' ','12',50),{pages:null,max_results:12});
});
test('page selection expands and deduplicates ranges without changing source page numbers',()=>{
 assert.deepEqual(extractionRequestScope(' 5, 1，3-5, 1 ','30',10),{pages:[1,3,4,5],max_results:30});
});
test('twenty-five selected pages are allowed but twenty-six are blocked before a paid request',()=>{
 assert.equal(extractionRequestScope('1-25','1',100).pages.length,25);
 assert.throws(()=>extractionRequestScope('1-26','12',100),/25 页/);
 assert.throws(()=>extractionRequestScope('1-25,30','12',100),/25 页/);
});
test('invalid or out-of-document pages do not silently become full-document extraction',()=>{
 for(const pages of ['0','-1','2-1','11','1.5','1,','all','1-999999999999999999999'])assert.throws(()=>extractionRequestScope(pages,'12',10));
});
test('missing result limits and noninteger or excessive limits are rejected',()=>{
 for(const limit of ['','0','31','1.5','Infinity','1e1'])assert.throws(()=>extractionRequestScope('',limit,10));
});
