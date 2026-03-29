import os
import sys
import argparse
import json
import jsonlines
import traceback
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from auto_scorer import get_citation_score, EVAL_MODEL

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate citation metrics (Recall, Precision, F1).")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction files.")
    parser.add_argument("--output_dir", type=str, default="scores_cite", help="Directory to save evaluation results.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for multiprocessing.")
    return parser.parse_args()

def process(item):
    js, fout_path = item
    try:
        js = get_citation_score(js, max_statement_num=40)
        # Clean up output
        if 'answer' in js: del js['answer']
        if 'few_shot_scores' in js: del js['few_shot_scores']
        
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(js, ensure_ascii=False) + '\n')
            fout.flush()
        return js
    except Exception as e:
        print(f"Error processing query: {js.get('query', 'Unknown')}")
        traceback.print_exc()
        print('-'*100)
        return None

def main():
    args = parse_args()
    
    # We evaluate following LongBench datasets
    datasets = [
        "longbench-chat", 
        "multifieldqa_en", 
        "hotpotqa", 
        "gov_report"
    ]
    
    os.makedirs(f"{args.output_dir}/tmp", exist_ok=True)
    
    if not os.path.exists(args.pred_dir):
        print(f"Error: Directory {args.pred_dir} not found.")
        return

    print(f"Processing prediction directory: {args.pred_dir}")
    print(f"Evaluation Config - Model: {EVAL_MODEL}")
    
    for filename in os.listdir(args.pred_dir):
        if not filename.endswith(".json"):
            continue
            
        path = os.path.join(args.pred_dir, filename)
        save_name = filename
        
        fout_path = os.path.join(args.output_dir, "tmp", f"{save_name}l")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        ipts = [x for x in data if x.get('dataset') in datasets]

        if os.path.exists(fout_path):
            with jsonlines.open(fout_path, 'r') as f:
                opts = [x for x in f]
        else:
            opts = []
            
        s = set(x['idx'] for x in opts)
        need_list = [(x, fout_path) for x in ipts if x['idx'] not in s]

        if need_list:
            print(f"File: {filename} - {len(need_list)} remaining to evaluate.")
            with Pool(args.num_workers) as p:
                rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
            opts = opts + [x for x in rst if x is not None]
            
        opts = sorted(opts, key=lambda x: x['idx'])
        
        result = {'scores': {}}
        for dataset in datasets:
            parts = [x for x in opts if dataset in x['dataset']]
            if len(parts) > 0:
                recall = np.mean([x['citation_recall'] for x in parts])
                precision = np.mean([x['citation_precision'] for x in parts])
                f1 = np.mean([x['citation_f1'] for x in parts])
                finish = len(parts) == sum([1 for x in ipts if dataset in x['dataset']])
            else:
                recall = precision = f1 = 0.0
                finish = False
                
            result['scores'][dataset] = {
                "citation_recall": float(recall),
                "citation_precision": float(precision),
                "citation_f1": float(f1),
                'finish': finish,
            }
        
        finish = len(opts) == len(ipts)
        gpt_usage = {
            'prompt_tokens': sum(x.get('gpt_usage', {}).get('prompt_tokens', 0) for x in opts),
            'completion_tokens': sum(x.get('gpt_usage', {}).get('completion_tokens', 0) for x in opts),
            'eval_model': EVAL_MODEL
        }
        
        # Averages ignoring multiple lang variants (Not needed since only English now)
        valid_datasets = list(result['scores'].keys())
        result.update({
            "avg_citation_recall": float(np.mean([result['scores'][k]['citation_recall'] for k in valid_datasets])),
            "avg_citation_precision": float(np.mean([result['scores'][k]['citation_precision'] for k in valid_datasets])),
            "avg_citation_f1": float(np.mean([result['scores'][k]['citation_f1'] for k in valid_datasets])),
            "finish": finish,
            "gpt_usage": gpt_usage,
        })
        
        opts.append(result)
        os.makedirs(args.output_dir, exist_ok=True)
        final_file_path = os.path.join(args.output_dir, save_name)
        with open(final_file_path, "w", encoding="utf-8") as f:
            json.dump(opts, f, indent=2, ensure_ascii=False)
            
        print(f"\nFinal Result for {filename}:")
        print(json.dumps(result, indent=2))
        print(f"Saved evaluation mapping to {final_file_path}\n")

if __name__ == "__main__":
    main()
