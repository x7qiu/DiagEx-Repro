"""Paid, ledger-bounded image + structured-output capability checks."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw
from diagex.config import LLMConfig, RuntimeBudgets
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.client import LLMClient
from diagex.llm.model_policy import FAST_MODEL, ESCALATION_MODEL, OPEN_WEIGHT_MODELS
from diagex.vision.encode import encode_image_block


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--models',nargs='+',default=[FAST_MODEL,ESCALATION_MODEL])
    args=parser.parse_args()
    base=LLMConfig.from_env()
    image=Image.new('RGB',(160,160),'white');ImageDraw.Draw(image).ellipse((35,35,125,125),outline='black',width=3)
    tool={'name':'describe_shape','description':'Describe the single black outline.','input_schema':{'type':'object','properties':{'shape':{'enum':['circle','square','triangle']}},'required':['shape']}}
    results=[]
    for model in args.models:
        cfg=replace(base,transport='openrouter',model=model,vision_model=None,reasoning_model=None,escalation_model=None,
                    production_open_weight=model in OPEN_WEIGHT_MODELS,reasoning_mode='disabled',
                    spending_ledger=str(args.preflight/'spending.json'),verified_prices=str(args.preflight/'verified-prices.json'),spending_category='extraction')
        client=LLMClient(cfg,budgets=RuntimeBudgets(retry_attempts=1))
        try:
            response=client.messages_create(system='Identify the black outline in the image and call describe_shape.',messages=[{'role':'user','content':[encode_image_block(image)]}],tools=[tool],tool_choice={'type':'tool','name':tool['name']},max_tokens=512,thinking={'type':'disabled'},reasoning_mode_override='disabled',time_budget_s=60,max_attempts=1)
            blocks=[b.model_dump() for b in response.content]
            success=any(b.get('type')=='tool_use' and b.get('name')==tool['name'] and b.get('input',{}).get('shape')=='circle' for b in blocks)
            results.append({'model':model,'passed':success,'response':response.model_dump(mode='json')})
        except Exception as exc:
            results.append({'model':model,'passed':False,'error':str(exc)[:1000]})
        atomic_write_json(args.preflight/'smoke-results.json',results)
        print(json.dumps({k:v for k,v in results[-1].items() if k!='response'}),flush=True)
    path=args.preflight/'verified-prices.json';prices=json.loads(path.read_text())
    for result in results:
        prices['models'][result['model']]['capabilities']['live_smoke_verified']=result['passed']
    atomic_write_json(path,prices)


if __name__=='__main__':
    main()
