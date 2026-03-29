import os
import sys
import argparse
import json
import jsonlines
import traceback
import re
from tqdm import tqdm
from multiprocessing import Pool

from utils.llm_api import query_llm
from utils.retrieve import text_split_by_n_sentences

def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot prediction with citation granularities.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The name of the LLM to use.")
    parser.add_argument("--dataset", type=str, default="data/LongBench-Cite_en.json", help="Path to the dataset JSON file.")
    parser.add_argument("--n_sentences", type=int, default=5, help="Number of sentences per chunk for text splitting.")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save the output predictions.")
    parser.add_argument("--prompt_file", type=str, default="prompts/zero_shot_prompt.txt", help="Path to zero-shot prompt template.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for multiprocessing.")
    return parser.parse_args()

def get_citations(statement, sents):
    c_texts = re.findall(r'<cite>(.*?)</cite>', statement, re.DOTALL)
    
    individual_chunks = []
    for c_text in c_texts:
        found_ranges = re.findall(r"\[C?([0-9]+)\-C?([0-9]+)\]", c_text)
        for st, ed in found_ranges:
            st, ed = int(st), int(ed)
            for i in range(st, ed + 1):
                individual_chunks.append(i)
        
        c_text_no_ranges = re.sub(r"\[C?([0-9]+)\-C?([0-9]+)\]", "", c_text)
        found_singletons = re.findall(r"\[C?([0-9]+)\]", c_text_no_ranges)
        for n in found_singletons:
            individual_chunks.append(int(n))
    
    individual_chunks = sorted(list(set(individual_chunks)))
    statement = re.sub(r'<cite>(.*?)</cite>', '', statement, flags=re.DOTALL)
    
    citations_list = []
    for chunk_idx in individual_chunks:
        try:
            if chunk_idx > len(sents) - 1:
                continue
            
            citation_data = {
                "st_sent": chunk_idx,
                "ed_sent": chunk_idx,
                'cite': sents[chunk_idx]['content'],
                "start_char": sents[chunk_idx]['start_idx'],
                "end_char": sents[chunk_idx]['end_idx']
            }
                
            citations_list.append(citation_data)
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
            traceback.print_exc()
            
    return statement, citations_list[:10]

def postprocess(answer, context, sents):
    res = []
    pos = 0
    while True:
        st = answer.find("<statement>", pos)
        if st == -1:
            st = len(answer)
        ed = answer.find("</statement>", st)
        
        statement = answer[pos:st]
        if len(statement.strip()) > 5:
            res.append({
                "statement": statement.strip(),
                "citation": []
            })
        
        if ed == -1:
            break

        statement_with_cite = answer[st+len("<statement>"):ed]
        if len(statement_with_cite.strip()) > 0:
            statement_text, citations = get_citations(statement_with_cite, sents)
            res.append({
                "statement": statement_text.strip(),
                "citation": citations
            })
        pos = ed + len("</statement>")
    return res

def worker_init(args_dict, prompt_text):
    global ARGS, PROMPT_FORMAT
    ARGS = args_dict
    PROMPT_FORMAT = prompt_text

def process(js):
    try:
        context, query = js['context'], js['query']
        
        # We exclusively use sentence-level splitting
        sents = text_split_by_n_sentences(
            context, 
            sentences_per_chunk=ARGS['n_sentences'], 
            return_dict=True
        )

        passage = ""
        for i, c in enumerate(sents):
            passage += f"<C{i}>{c['content']}"
        
        prompt = PROMPT_FORMAT.replace('<<context>>', passage).replace('<<question>>', query)
        
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=ARGS['model'], temperature=ARGS['temperature'], max_new_tokens=1024)
        if output is None:
            raise NotImplementedError("Error: No Output")
        if output.startswith("statement>"):
            output = "<" + output
        
        statements_with_citations = postprocess(output, context, sents)
        
        res = {
            'idx': js['idx'],
            'dataset': js['dataset'],
            'query': js['query'],
            'answer': js['answer'],
            'few_shot_scores': js['few_shot_scores'],
            'prediction': output,
            'statements': statements_with_citations,
        }
        with open(ARGS['fout_path'], "a") as fout:
            fout.write(json.dumps(res, ensure_ascii=False)+'\n')
            fout.flush()
        return res
    except Exception as e:
        print(f"Warning: Failed to process index {js.get('idx', 'Unknown')} (Query: {js.get('query', 'Unknown')[:50]}...) | Error: {e}")
        return None

def main():
    args = parse_args()
    print(f"Loading dataset from: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        ipts = json.load(f)

    print(f"Loading prompt from: {args.prompt_file}")
    with open(args.prompt_file, "r", encoding="utf-8") as fp:
        prompt_format = fp.read()

    save_name = args.model.split('/')[-1]
    
    # We use '_s_' to represent sentence splitting mode and suffix the n_sentences
    output_suffix = f"s_{args.n_sentences}"
    
    os.makedirs(f'{args.output_dir}/tmp', exist_ok=True)
    fout_path = f'{args.output_dir}/tmp/{save_name}_{output_suffix}.jsonl'

    if os.path.exists(fout_path):
        with jsonlines.open(fout_path, 'r') as f:
            opts = [x for x in f]
    else:
        opts = []
        
    s = set(x['idx'] for x in opts)
    need_list = [x for x in ipts if x['idx'] not in s]
    
    print(f'Model: {args.model}')
    print(f'Sentence chunk size: {args.n_sentences}')
    print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
    
    if len(need_list) == 0:
        print("Done predicting all samples.")
        return

    args_dict = vars(args)
    args_dict['fout_path'] = fout_path

    # Distribute work via Pool
    with Pool(args.num_workers, initializer=worker_init, initargs=(args_dict, prompt_format)) as p:
        rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))

    rst = opts + [x for x in rst if x is not None]
    rst = sorted(rst, key=lambda x: x['idx'])

    os.makedirs(f'{args.output_dir}/{save_name}', exist_ok=True)
    final_output_path = f'{args.output_dir}/{save_name}/{save_name}_{output_suffix}.json'
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(rst, f, indent=2, ensure_ascii=False)
        
    print(f"Saved complete predictions to {final_output_path}")

if __name__ == "__main__":
    main()
