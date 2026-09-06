#!/usr/bin/env python3
"""Measured AR vs Uno speed benchmark. AR uses Transformers; Uno uses official Linear sampler."""
from __future__ import annotations
import argparse, csv, json, os, statistics, subprocess, sys, time
from pathlib import Path

def render_user_chat(tokenizer, content: str) -> str:
    text=tokenizer.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=False)
    return text+"<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>\n"

def nvidia():
    try:
        s=subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",shell=True,text=True).strip().split(",")
        return {"gpu_util_pct":float(s[0]),"vram_used_mb":float(s[1])}
    except Exception: return {}
def read_prompt(length):
    base=("You are benchmarking a browser semantic layer. Read the following structured observation and return a concise JSON object with page_type, entities, attributes, and actions. ")
    filler="The browser observation contains stable text tokens for context measurement. "
    # The target is measured after chat-template rendering and tokenization.
    # Use a conservative character budget so every requested bucket is at
    # least the target token count; the actual count is recorded in output.
    return base+filler*(((length*8)//len(filler))+20)
def uno_group(args, prompts, warmup, runs, length):
    sys.path.insert(0,str(args.uno_root)); from generation import resolve_model_sources; from nano_vllm_uno import LLM,SamplingParams; from nano_vllm_uno.utils.hf_compat import load_tokenizer; import torch
    model,adapter=resolve_model_sources(args.model,args.adapter,hf_cache_dir=args.hf_cache); tok=load_tokenizer(model,use_fast=True,trust_remote_code=True,cache_dir=args.hf_cache)
    enc=[tok(render_user_chat(tok,p),add_special_tokens=False).input_ids for p in prompts]
    prompt_tokens=len(enc[0])
    sp=SamplingParams(temperature=0.0,top_k=1,top_p=1.0,max_tokens=args.output_tokens,ignore_eos=True,stop_token_ids=[],mask_token_id=64256,noise_mode="deterministic_uniform",diffusion_block_size=8)
    max_len=max(2048, prompt_tokens+args.output_tokens+512)
    llm=None
    llm=LLM(model=model,tokenizer_path=model,max_num_seqs=1,max_model_len=max_len,max_num_batched_tokens=max_len,gpu_memory_utilization=args.gpu_memory_utilization,attention_backend="fa2",max_diffusion_block_size=8,gated_lora_path=adapter,hf_cache_dir=args.hf_cache,cuda_graph_block_sizes=[1,8])
    def go(count):
        events=[]
        def cb(stats): events.extend([time.perf_counter()]*len(stats))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter(); out=llm.generate(enc*count,sp,request_max_tokens=[args.output_tokens]*count,use_tqdm=False,progress_callback=cb); torch.cuda.synchronize(); end=time.perf_counter()
        return end-t,events,torch.cuda.max_memory_allocated()/2**20
    try:
        go(warmup)
        elapsed,events,v=go(runs)
        vals=[(events[i]-events[i-1]) if i else (events[i]-time.perf_counter()+elapsed) for i in range(len(events))] if events else [elapsed/runs]*runs
        # Callback timestamps are completion times; use inter-completion intervals only
        # when they are positive, otherwise keep a conservative batch-mean fallback.
        vals=[x for x in vals if x>0] or [elapsed/runs]
        stats=llm.last_generate_stats
        return vals,[{"decoder_tokens_per_sequence_forward":stats.get("accepts",0)/stats.get("forwards",1)}],v,prompt_tokens
    finally:
        if llm is not None:
            llm.exit()
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

def run_isolated_uno_lengths(args, lengths):
    """Run each Uno length in a fresh interpreter to isolate CUDA/dist state."""
    rows=[]
    for length in lengths:
        part=args.output.with_name(f"{args.output.stem}.part-{length}{args.output.suffix}")
        cmd=[sys.executable,str(Path(__file__).resolve()),"--mode","uno","--output",str(part),"--uno-root",str(args.uno_root),"--model",args.model,"--adapter",args.adapter,"--input-lengths",str(length),"--output-tokens",str(args.output_tokens),"--warmup",str(args.warmup),"--runs",str(args.runs),"--gpu-memory-utilization",str(args.gpu_memory_utilization),"--attn-implementation",args.attn_implementation]
        if args.hf_cache:
            cmd.extend(["--hf-cache",args.hf_cache])
        child_env=os.environ.copy()
        child_env["UNO_SPEED_BENCH_CHILD"]="1"
        completed=subprocess.run(cmd,env=child_env,text=True)
        if part.exists():
            try:
                child_rows=[json.loads(line) for line in part.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("mode") == "uno"]
                if child_rows:
                    rows.append(child_rows[-1])
                    continue
            except Exception as exc:
                rows.append({"mode":"uno","input_target_tokens":length,"warmup":args.warmup,"runs":args.runs,"status":"error","error":repr(exc)})
                continue
        rows.append({"mode":"uno","input_target_tokens":length,"warmup":args.warmup,"runs":args.runs,"status":"error","error":f"child benchmark exited with code {completed.returncode} without a result"})
    return rows
def ar_group(args,prompt,warmup,runs,length):
    import torch; from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,use_fast=True); model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map="auto",attn_implementation=args.attn_implementation); model.eval(); device=next(model.parameters()).device; rendered=render_user_chat(tok,prompt); ids=tok(rendered,return_tensors="pt",add_special_tokens=False).to(device); 
    def go():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter(); out=model.generate(**ids,max_new_tokens=args.output_tokens,do_sample=False,eos_token_id=None,pad_token_id=tok.eos_token_id); torch.cuda.synchronize(); return time.perf_counter()-t,out.shape[1]-ids["input_ids"].shape[1],torch.cuda.max_memory_allocated()/2**20
    for _ in range(warmup): go()
    vals=[]; toks=[]; vm=[]
    for _ in range(runs): e,n,v=go(); vals.append(e); toks.append(n); vm.append(v)
    return vals,toks,max(vm),int(ids["input_ids"].shape[1])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("ar","uno"),required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--uno-root",type=Path,default=Path(__file__).resolve().parents[1]/"vendor/uno"); ap.add_argument("--model",default="IFM/K2-Horizon-0.9B"); ap.add_argument("--adapter",default="IFM/K2-Horizon-0.9B-Uno"); ap.add_argument("--hf-cache"); ap.add_argument("--input-lengths",default="256,1024,4096,8192"); ap.add_argument("--output-tokens",type=int,default=128); ap.add_argument("--warmup",type=int,default=5); ap.add_argument("--runs",type=int,default=30); ap.add_argument("--gpu-memory-utilization",type=float,default=.78); ap.add_argument("--attn-implementation",default="flash_attention_2"); a=ap.parse_args(); rows=[]
    lengths=[int(x) for x in a.input_lengths.split(",")]
    if a.mode == "uno" and len(lengths) > 1 and os.environ.get("UNO_SPEED_BENCH_CHILD") != "1":
        rows=run_isolated_uno_lengths(a, lengths)
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
        ok=[r for r in rows if r.get("status")=="ok"]; print(json.dumps({"mode":a.mode,"ok":len(ok),"total":len(rows),"output":str(a.output)},ensure_ascii=False)); return
    for length in lengths:
        p=read_prompt(length); observed={"prompt_chars":len(p)}; 
        try:
            if a.mode=="uno":
                vals,ms,v,prompt_tokens=uno_group(a,[p],a.warmup,a.runs,length); tpf=statistics.mean([m.get("decoder_tokens_per_sequence_forward",0) for m in ms]); toks=a.output_tokens; ttft=None
            else:
                vals,toks_list,v,prompt_tokens=ar_group(a,p,a.warmup,a.runs,length); toks=statistics.mean(toks_list); tpf=1.0; ttft=None
            observed["prompt_tokens"]=prompt_tokens
            row={"mode":a.mode,"input_target_tokens":length,"output_tokens":toks,"warmup":a.warmup,"runs":a.runs,"latency_p50_s":statistics.median(vals),"latency_p95_s":sorted(vals)[max(0,int(.95*len(vals))-1)],"tps_mean":toks/statistics.mean(vals),"tpf":tpf,"peak_vram_mb":v,"gpu_sample":nvidia(),"observed":observed,"status":"ok"}
        except Exception as e: row={"mode":a.mode,"input_target_tokens":length,"warmup":a.warmup,"runs":a.runs,"status":"error","error":repr(e),"gpu_sample":nvidia(),"observed":observed}
        rows.append(row); print(json.dumps(row,ensure_ascii=False))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
    ok=[r for r in rows if r["status"]=="ok"]; print(json.dumps({"mode":a.mode,"ok":len(ok),"total":len(rows),"output":str(a.output)},ensure_ascii=False))
if __name__=="__main__": main()
