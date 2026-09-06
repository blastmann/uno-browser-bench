#!/usr/bin/env python3
"""Run structured Browser Observation quality probes with AR or Uno."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

SYSTEM = "You are a browser semantic parser. Return only one compact JSON object. Do not use Markdown, explanations, or extra keys."

def render_user_chat(tokenizer, content: str) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=False
    )
    return text + "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>\n"

def prompt_for(row: dict, task: str) -> str:
    if task == "entity":
        return SYSTEM + "\nSchema: {page_type:string, entities:[{type:string,name:string,attributes:object,actions:[string]}]}\nBrowser Observation:\n" + json.dumps(row["observation"], ensure_ascii=False, separators=(",", ":"))
    return SYSTEM + "\nSchema: {intent:string, entities:[string], steps:[string]}\nBrowser events:\n" + json.dumps(row["events"], ensure_ascii=False, separators=(",", ":"))

def load_rows(path: Path, limit: int | None, task: str):
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if limit: rows=rows[:limit]
    out=[]
    for row in rows:
        tasks=("entity","trajectory") if task=="all" else (task,)
        for t in tasks:
            gold=row["entity_gold"] if t=="entity" else row["trajectory_gold"]
            out.append({"id":row["id"],"domain":row["domain"],"task":t,"trajectory_length":row["trajectory_length"],"gold":gold,"prompt":prompt_for(row,t)})
    return out

def run_ar(items, model_name, max_new_tokens, batch_size, attn_implementation):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
    tok=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation=attn_implementation)
    model.eval(); device=next(model.parameters()).device
    class FirstToken(LogitsProcessor):
        def __init__(self): self.t=None
        def __call__(self,input_ids,scores):
            if self.t is None: self.t=time.perf_counter()
            return scores
    results=[]
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size]
        rendered=[render_user_chat(tok, x["prompt"]) for x in batch]
        enc=tok(rendered, return_tensors="pt", padding=True, truncation=True).to(device)
        for i,x in enumerate(batch):
            # Keep one request per call for clean TTFT/latency accounting.
            ids=enc["input_ids"][i:i+1]; mask=enc["attention_mask"][i:i+1]; inp=int(mask.sum().item())
            proc=FirstToken(); torch.cuda.synchronize() if torch.cuda.is_available() else None; torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0=time.perf_counter()
            with torch.inference_mode():
                out=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=max_new_tokens,do_sample=False,eos_token_id=None,pad_token_id=tok.eos_token_id,logits_processor=LogitsProcessorList([proc]))
            torch.cuda.synchronize() if torch.cuda.is_available() else None; t1=time.perf_counter()
            text=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
            results.append({"id":x["id"],"domain":x["domain"],"task":x["task"],"trajectory_length":x["trajectory_length"],"gold":x["gold"],"output":text,"input_tokens":inp,"output_tokens":int(out.shape[1]-ids.shape[1]),"latency_s":t1-t0,"ttft_s":(proc.t-t0 if proc.t else None),"tps":(out.shape[1]-ids.shape[1])/(t1-t0) if t1>t0 else 0,"peak_vram_mb":torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,"runner":"transformers-ar"})
    return results

def run_uno(items, uno_root: Path, model_name, adapter_name, max_new_tokens, batch_size, hf_cache, gpu_memory_utilization):
    sys.path.insert(0,str(uno_root))
    from generation import resolve_model_sources
    from nano_vllm_uno import LLM, SamplingParams
    from nano_vllm_uno.utils.hf_compat import load_tokenizer
    model, adapter=resolve_model_sources(model_name,adapter_name,hf_cache_dir=hf_cache)
    tok=load_tokenizer(model,use_fast=True,trust_remote_code=True,cache_dir=hf_cache)
    sp=SamplingParams(temperature=0.0,top_k=1,top_p=1.0,max_tokens=max_new_tokens,ignore_eos=True,stop_token_ids=[],mask_token_id=64256,noise_mode="deterministic_uniform",diffusion_block_size=8)
    max_len=8192
    llm=LLM(model=model,tokenizer_path=model,max_num_seqs=batch_size,max_model_len=max_len,max_num_batched_tokens=max_len,gpu_memory_utilization=gpu_memory_utilization,attention_backend="fa2",max_diffusion_block_size=8,gated_lora_path=adapter,hf_cache_dir=hf_cache,cuda_graph_block_sizes=[1,8])
    results=[]
    try:
        for start in range(0,len(items),batch_size):
            batch=items[start:start+batch_size]
            prompts=[tok(render_user_chat(tok, x["prompt"]), add_special_tokens=False).input_ids for x in batch]
            import torch
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0=time.perf_counter(); outputs=llm.generate(prompts,sp,request_max_tokens=[max_new_tokens]*len(batch),use_tqdm=False); elapsed=time.perf_counter()-t0
            peak=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None; stats=llm.last_generate_stats; tpf=stats.get("accepts",0)/stats.get("forwards",1) if stats.get("forwards",0) else 0
            for x,o in zip(batch,outputs):
                text=o.get("text",""); n=len(o.get("token_ids",[]));
                results.append({"id":x["id"],"domain":x["domain"],"task":x["task"],"trajectory_length":x["trajectory_length"],"gold":x["gold"],"output":text,"input_tokens":len(prompts[0]) if prompts else None,"output_tokens":n,"latency_s":elapsed/len(batch),"ttft_s":None,"tps":n/(elapsed/len(batch)) if elapsed else 0,"tpf":tpf,"peak_vram_mb":peak,"runner":"uno-linear"})
    finally:
        llm.exit()
    return results

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("ar","uno"),required=True); ap.add_argument("--dataset",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--task",choices=("entity","trajectory","all"),default="all"); ap.add_argument("--limit",type=int); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--max-new-tokens",type=int,default=128); ap.add_argument("--model",default="IFM/K2-Horizon-0.9B"); ap.add_argument("--adapter",default="IFM/K2-Horizon-0.9B-Uno"); ap.add_argument("--uno-root",type=Path,default=Path(__file__).resolve().parents[1]/"vendor/uno"); ap.add_argument("--hf-cache"); ap.add_argument("--gpu-memory-utilization",type=float,default=.78); ap.add_argument("--attn-implementation",default="flash_attention_2")
    a=ap.parse_args(); items=load_rows(a.dataset,a.limit,a.task)
    res=run_ar(items,a.model,a.max_new_tokens,a.batch_size,a.attn_implementation) if a.mode=="ar" else run_uno(items,a.uno_root,a.model,a.adapter,a.max_new_tokens,a.batch_size,a.hf_cache,a.gpu_memory_utilization)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in res)+"\n",encoding="utf-8"); print(json.dumps({"mode":a.mode,"items":len(res),"output":str(a.output)},ensure_ascii=False))
if __name__=="__main__": main()
