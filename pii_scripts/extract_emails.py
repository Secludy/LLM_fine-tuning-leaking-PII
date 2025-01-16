import json
import re
from collections import Counter
from pathlib import Path

def extract_email(text):
    """Extract email address from a string using regex."""
    email_pattern = r'<([\w\.-]+@[\w\.-]+\.\w+)>'  # Updated pattern to extract from <email>
    match = re.search(email_pattern, text)
    return match.group(1) if match else None

def process_json_file(input_path):
    """Process JSON file and extract emails with their counts."""
    email_counter = Counter()
    
    print(f"Reading file: {input_path}")
    
    # Read the entire JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            
            # Process each record in the array
            for record in data:
                # Extract email from 'from' field
                if 'from' in record:
                    from_email = extract_email(record['from'])
                    if from_email:
                        email_counter[from_email] += 1
                
                # Extract email from 'to' field
                if 'to' in record:
                    to_email = extract_email(record['to'])
                    if to_email:
                        email_counter[to_email] += 1
                        
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file - {str(e)}")
            return dict()
        except Exception as e:
            print(f"Error: Unexpected error while processing file - {str(e)}")
            return dict()
    
    return dict(email_counter)

def main():
    # Define paths
    base_dir = Path.cwd()
    input_file = base_dir / 'pii_scripts' / 'data' / 'finetuning_spam_dataset_input_modified_no_dp_1_pii.jsonl'
    output_dir = base_dir / 'pii_scripts' / 'outputs'
    output_file = output_dir / 'email_stats.json'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the file and get email counts
    email_counts = process_json_file(input_file)
    
    if not email_counts:
        print("Error: No emails were found or processed!")
        return
    
    # Sort by count in descending order
    sorted_counts = dict(sorted(email_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Save results to JSON file with statistics
    output_data = {
        "email_counts": sorted_counts,
        "statistics": {
            "total_unique_emails": len(sorted_counts),
            "total_email_occurrences": sum(sorted_counts.values())
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total unique emails found: {len(sorted_counts)}")
    print(f"Total email occurrences: {sum(sorted_counts.values())}")
    print("\nTop 5 most frequent emails:")
    for email, count in list(sorted_counts.items())[:5]:
        print(f"{email}: {count} occurrences")

if __name__ == '__main__':
    main()