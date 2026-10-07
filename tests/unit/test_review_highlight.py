"""Regression coverage for source-only point findings and zoom-stable markers."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_review_issue_geometry():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for review geometry checks")
    script = Path(__file__).parents[2] / "src/diagex/review/static/app.js"
    check = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const context=vm.createContext({window:{matchMedia:()=>({matches:false})},localStorage:{getItem:()=>null}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace(/init\(\)\.catch[\s\S]*$/, ''),context);
vm.runInContext(`
app.state={graph:{nodes:[],edges:[]}};
const crossing={conflict:{type:'crossing_or_junction',page_index:3,x:1527,y:2356},candidates:{nodes:[],edges:[]}};
globalThis.point=conflictBounds(crossing,3);
globalThis.otherPage=conflictBounds(crossing,4);
globalThis.small=paddedIssueBounds(point,.1,{width:6000,height:4238});
globalThis.large=paddedIssueBounds(point,1,{width:6000,height:4238});
const missing={conflict:{type:'page_graph_uncertainty',page_index:3},candidates:{nodes:[],edges:[]}};
globalThis.missing=conflictBounds(missing,3);
const explicit={conflict:{page_index:3,x:10,y:20},candidates:{nodes:[{id:'far',page_index:3,bbox_global:{x:500,y:500,w:200,h:200}}],edges:[]}};
globalThis.precise=conflictBounds(explicit,3);
`,context);
assert.equal(context.point.x,1527);assert.equal(context.point.y,2356);
assert.equal(context.otherPage,null);assert.equal(context.missing,null);
assert.equal(context.precise.x,10);assert.equal(context.precise.w,1);
assert.equal(Math.round(context.small.w*.1),72);assert.equal(context.large.w,72);
'''
    subprocess.run([node, "-e", check, str(script)], check=True, capture_output=True, text=True)
