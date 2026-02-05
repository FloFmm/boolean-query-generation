import os

def count_keywords_in_file(file_path, keywords):
    counts = {key: 0 for key in keywords}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for key in keywords:
                counts[key] += line.count(key)
    counts['SUM'] = sum(counts.values())
    return counts

def main(folder_path):
    keywords = ['OR', 'AND', 'NOT', '$W']
    all_counts = []
    
    # Go through all .txt files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                counts = count_keywords_in_file(file_path, keywords)
                all_counts.append(counts)
                
                # Print per-file counts
                print(f"File: {file}")
                for key in keywords:
                    print(f"  {key}: {counts[key]}")
                print(f"  SUM: {counts['SUM']}\n")
    
    # Compute averages across all files
    if all_counts:
        avg_counts = {key: sum(d[key] for d in all_counts) / len(all_counts) for key in keywords}
        avg_counts['SUM'] = sum(d['SUM'] for d in all_counts) / len(all_counts)
        
        print("AVERAGES ACROSS ALL FILES:")
        for key in keywords:
            print(f"  {key}: {avg_counts[key]:.2f}")
        print(f"  SUM: {avg_counts['SUM']:.2f}")
    else:
        print("No .txt files found in the specified folder.")

if __name__ == "__main__":
    main("../master-thesis-writing/data/pid/boolean-query-examples/")
