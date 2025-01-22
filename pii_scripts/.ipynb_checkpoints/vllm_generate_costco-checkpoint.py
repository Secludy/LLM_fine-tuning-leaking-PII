# LLM_fine-tuning-leaking-PII/fine-tuning/vllm_generate_costco.py

from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
import re
from collections import defaultdict
import argparse

def load_model_for_inference(model_path):
    """Helper function to load and validate model for vLLM"""
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Initializing vLLM engine...")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=512,
        dtype="float16",
        gpu_memory_utilization=0.8,
        quantization=None,  # Disable quantization for merged model
        enforce_eager=True
    )
    
    return tokenizer, llm

def validate_email_content(email_content, category):
    """Validate that the email content is complete and matches the category"""
    # Check for required fields
    required_fields = ['From:', 'To:', 'Subject:', 'Content:']
    if not all(field in email_content for field in required_fields):
        return False
        
    # Check minimum length (to avoid stub responses)
    if len(email_content) < 100:  # Adjust threshold as needed
        return False
    
    # Ensure it's not just the category name repeated
    if email_content.strip() == category:
        return False
    
    return True

def extract_email_content(text, category):
    """Extract only the first complete email content from the response"""
    # Remove the Response prefix if present
    if "### Response:" in text:
        text = text.split("### Response:")[1].strip()
    
    # Find all occurrences of email starts
    email_starts = [m.start() for m in re.finditer(r'From:', text)]
    
    if not email_starts:
        return None
    
    # Try each email start position until we find a valid one
    for start in email_starts:
        # Extract until next email or end
        match = re.search(r'From:.*?(?=\n\nFrom:|\n\n### |\Z)', text[start:], re.DOTALL)
        if not match:
            continue
            
        email_content = match.group(0).strip()
        if validate_email_content(email_content, category):
            return email_content
    
    return None

def generate_emails(llm, categories, num_per_category, sampling_params):
    """Generate emails with retry logic to ensure we get enough valid examples per category"""
    valid_examples = []
    counts_per_category = defaultdict(int)
    max_attempts_per_category = num_per_category * 3  # Allow for some retries
    
    # Keep track of attempts per category
    attempts_per_category = defaultdict(int)
    
    while True:
        # Check if we have enough examples for all categories
        if all(counts_per_category[cat] >= num_per_category for cat in categories):
            break
            
        # Check if we've exceeded max attempts for all remaining categories
        remaining_categories = [cat for cat in categories 
                              if counts_per_category[cat] < num_per_category 
                              and attempts_per_category[cat] < max_attempts_per_category]
        if not remaining_categories:
            print("\nWarning: Max attempts reached for all remaining categories")
            break
            
        # Prepare prompts for categories that need more examples
        prompts = []
        categories_map = []
        
        for category in remaining_categories:
            needed = num_per_category - counts_per_category[category]
            # Generate a few extra to account for potential failures
            batch_size = min(needed * 2, max_attempts_per_category - attempts_per_category[category])
            
            for _ in range(batch_size):
                prompt = alpaca_prompt.format(
                    f"Write exactly one corporate email example in the category of {category}. "
                    f"The response should contain exactly one email with From, To, Subject, and Content fields. "
                    f"Do not include multiple emails or explanations.",
                    "",
                    ""
                )
                prompts.append(prompt)
                categories_map.append(category)
                attempts_per_category[category] += 1
        
        if not prompts:
            break
            
        # Generate responses
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs
        for output, category in zip(outputs, categories_map):
            if counts_per_category[category] >= num_per_category:
                continue
                
            generated_text = output.outputs[0].text
            email_content = extract_email_content(generated_text, category)
            
            if email_content is not None:
                valid_examples.append({
                    "category": category,
                    "example": email_content
                })
                counts_per_category[category] += 1
                print(f"Valid example #{counts_per_category[category]} for {category}")
            else:
                print(f"Skipping invalid example for {category}")
        
        # Print progress
        print("\nProgress:")
        for category in categories:
            print(f"{category}: {counts_per_category[category]}/{num_per_category} "
                  f"(attempts: {attempts_per_category[category]}/{max_attempts_per_category})")
    
    return valid_examples, dict(counts_per_category)

# Initialize model and tokenizer
model_path = "trained_model_no_dp_4_PII"
tokenizer, llm = load_model_for_inference(model_path)

# Define the Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

# List of categories (COSTCO)
categories = [
    "Client Communications",
    "Company Announcements",
    "Department Updates",
    "Employee Engagement",
    "Financial Reports",
    "HR Communications",
    "IT Support and Systems",
    "Meeting Coordination",
    "Operations",
    "Performance Reviews",
    "Policy Updates",
    "Process Documentation",
    "Project Updates",
    "Research and Development",
    "Resource Requests",
    "Sales and Marketing",
    "Team Collaboration",
    "Technical Documentation",
    "Training and Development"
]

# Number of examples to generate per category
num_replicas_per_category = 245

# Configure sampling parameters for generation
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=500
)

# Generate examples with retry logic
generated_examples, final_counts = generate_emails(
    llm, 
    categories, 
    num_replicas_per_category, 
    sampling_params
)

# Print final statistics
print("\nFinal counts per category:")
for category in categories:
    print(f"{category}: {final_counts[category]}")

print(f"\nTotal examples generated: {len(generated_examples)}")

# Save the generated examples to a JSON file
output_path = "generated_email_examples_no_dp_4_PII.json"
with open(output_path, "w") as f:
    json.dump(generated_examples, f, indent=4)

print(f"\nGenerated examples saved to {output_path}")

def main(model_path, output_file):
    """Main function to generate emails using the trained model
    
    Args:
        model_path (str): Path to the trained model directory
        output_file (str): Path where generated examples will be saved
    """
    # Initialize model and tokenizer
    print(f"\nLoading model from: {model_path}")
    tokenizer, llm = load_model_for_inference(model_path)

    # Generate examples with retry logic
    print("\nStarting email generation...")
    generated_examples, final_counts = generate_emails(
        llm, 
        categories, 
        num_replicas_per_category, 
        sampling_params
    )

    # Print final statistics
    print("\nFinal counts per category:")
    for category in categories:
        print(f"{category}: {final_counts[category]}")

    print(f"\nTotal examples generated: {len(generated_examples)}")

    # Save the generated examples
    print(f"\nSaving generated examples to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(generated_examples, f, indent=4)

    print(f"Generation complete! Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate emails using trained model')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model directory')
    parser.add_argument('--output-file', type=str, required=True,
                      help='Path where generated examples will be saved')
    
    args = parser.parse_args()
    
    # Remove hardcoded values
    main(
        model_path=args.model_path,
        output_file=args.output_file
    )