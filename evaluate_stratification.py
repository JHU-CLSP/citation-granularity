import os
import json
import re
import argparse
import pandas as pd
import numpy as np

def calculate_f1(precision, recall):
    if (precision + recall) == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def create_buckets(df, bins=[0, 1, 2, 4, 8, 16, 32, 64]):
    bucket_labels = []
    for i in range(len(bins)-1):
        if i == 0: 
            bucket_labels.append('0')
        elif bins[i+1] - bins[i] == 1: 
            bucket_labels.append(f'{bins[i]}')
        else: 
            bucket_labels.append(f'{bins[i]}-{bins[i+1]-1}')
    bucket_labels.append(f'{bins[-1]}+')
    
    bins_with_inf = bins + [float('inf')]
    df['bucket'] = pd.cut(df['total_sentences'], bins=bins_with_inf, labels=bucket_labels, right=False)
    return df, bucket_labels

def main():
    parser = argparse.ArgumentParser(description="Generate volume-controlled stratification data based on citation output.")
    parser.add_argument("--eval_dir", type=str, default="scores_cite", help="Directory containing evaluated citation JSONs.")
    parser.add_argument("--output_dir", type=str, default="stratification_results", help="Directory to save CSV buckets.")
    args = parser.parse_args()

    if not os.path.exists(args.eval_dir):
        print(f"Error: Directory {args.eval_dir} not found.")
        return

    all_data = []

    # Parse flat JSON result files (e.g. Llama-3.1-8B-Instruct_s_4.json)
    for filename in os.listdir(args.eval_dir):
        if not filename.endswith(".json"): continue
        
        match = re.search(r'^(.*)_s_(\d+)\.json$', filename)
        if not match: 
            # Fallback if no specific k is identified (could manually be supplied, but skipping unformatted for now)
            continue
            
        model = match.group(1)
        granularity = int(match.group(2))
        
        filepath = os.path.join(args.eval_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            for result in results:
                if 'statements' not in result: continue

                dataset = result.get('dataset', 'unknown')
                query_idx = result.get('idx')

                for stmt_idx, statement in enumerate(result['statements']):
                    citations = statement.get('citation', [])
                    num_citations = len(citations)
                    total_sentences = num_citations * granularity
                    
                    rel_scores = [c.get('relevant_score', 0) for c in citations]
                    precision = sum(rel_scores) / num_citations if num_citations > 0 else 0.0
                    recall = statement.get('support_score', 0.0)
                    f1 = calculate_f1(precision, recall)
                    
                    all_data.append({
                        'model': model,
                        'granularity': granularity,
                        'dataset': dataset,
                        'query_idx': query_idx,
                        'statement_idx': stmt_idx,
                        'num_citations': num_citations,
                        'total_sentences': total_sentences,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
            
    if not all_data:
        print("No valid citation evaluation files found to stratify.")
        return
        
    df = pd.DataFrame(all_data)
    df, bucket_labels = create_buckets(df)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Simple Aggregate (Bucket + Granularity grouped)
    print("\n--- Volume-Controlled Stratification Results ---")
    
    summary_df = df.groupby(['model', 'bucket', 'granularity'], observed=True)[['precision', 'recall', 'f1']].mean().reset_index()
    summary_df['num_statements'] = df.groupby(['model', 'bucket', 'granularity'], observed=True).size().values
    
    # Save the aggregated summary
    summary_path = os.path.join(args.output_dir, "bucket_performance.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Print the bucket output snippet to stdout
    print(summary_df.to_string())
    print("\n")
    print(f"✅ Success! Stratification analysis saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
