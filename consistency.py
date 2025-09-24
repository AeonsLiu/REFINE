import argparse
import json
import os
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description='Apply self-consistency methodology to combine multiple rule sets')
    parser.add_argument('--input_file', type=str, default='./outputs/rule_set_samples.jsonl',
                        help='Path to the input file containing rule sets')
    parser.add_argument('--output_file', type=str, default='./outputs/consistency_result.jsonl',
                        help='Path to save the consistency analysis output')
    return parser.parse_args()

def load_rules_from_file(file_path):
    """Load rules from jsonl file and return as string"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Handle JSONL format with escaped content
        if content.startswith('"') and content.endswith('"'):
            # Remove outer quotes first
            content = content[1:-1]
            # Manually handle common escape sequences
            content = content.replace('\\n', '\n')  # Convert \n to actual newlines
            content = content.replace('\\t', '\t')  # Convert \t to actual tabs
            content = content.replace('\\"', '"')   # Convert \" to actual quotes
            content = content.replace('\\\\', '\\') # Convert \\ to actual backslashes
            return content
        else:
            # If it's not quoted, return as is
            return content

def main():
    args = parse_args()
    
    # Load rules from input file
    rules_content = load_rules_from_file(args.input_file)

    # Create prompt with loaded content
    prompt = f"""
You are an intelligent assistant tasked with applying a self-consistency methodology to combine multiple sets of regression-derived interval-based rules. 

You have received multiple independently generated rule-sets from multiple executions of previous analysis steps. Each rule-set consists of interval-based rules predicting continuous-values (e.g., MMSE) with specific conditions. Your goal is to integrate these sets of rules into one consistent, summarized, and robust rule set.

### Input: (Insert your independently generated rule-sets below)

{rules_content}

### Step 1: Identify Consistent and Robust Rule Patterns

For each predicted feature (e.g., Diagnosis), examine all rule-sets and perform the following:

- Identify stable, frequently occurring conditions and intervals across rule-sets.
- For intervals appearing consistently (at least 3 out of 5 times), retain them as robust and representative intervals.
- If intervals differ slightly across sets, choose interval bounds that reflect the majority consensus, using [mean(min_values), mean(max_values)].

### Step 2: Generate a Single Consistent Interval-based Rule Set (Self-consistency)

Produce a single final integrated rule-set incorporating the majority consensus intervals clearly. Format rules as follows:

- If [Predicted Feature] = [label], then [CONDITION1] and [CONDITION2]...

Example Output:
- If Diagnosis = 1, then MMSE ≤ 18.77 and ADL ≤ 3.18

Ensure the final output:
- Clearly represents stable intervals identified across multiple rule-sets.
- Uses inclusive/exclusive operators properly (>, ≥, <, ≤).
- Focuses clearly on key features:
  1. MMSE (Importance: 0.26)
  2. FunctionalAssessment (Importance: 0.20)
  3. ADL (Importance: 0.15)
- Avoids overly broad intervals, prioritizing clarity and consistency.

Example Summary:
When Diagnosis = 1:
- MMSE is typically low (≤ 18.77).
- ADL is usually impaired (≤ 3.18).
- PhysicalActivity often varies and isn't consistently influential.

When Diagnosis = 0:
- MMSE is typically normal or high (> 18.77).
- ADL is usually unimpaired (> 3.18).
- Other conditions are rarely consistent or influential.

"""
    


    system_prompt = """
You are a rule reasoning and generalization assistant designed to work with structured if-else logic extracted from decision tree-based machine learning models. 
Your purpose is to help transform raw nested if-else logic into usable, generalized knowledge that can guide downstream data generation, explanation, or modeling tasks.

You support both **classification** and **regression** trees. The input rules may predict:
- A **discrete target label** (e.g., Diagnosis = 0 or 1), or
- A **continuous value** (e.g., MMSE ≈ 21.04)

Your responsibilities include:

1. **Parsing and Flattening Logic**:
   - Convert nested if-else logic into a flat, human-readable rule format.
   - For regression rules, format each as:
     - "If [Target Variable] ≈ [Predicted Value], then [Conditions...]"
   - For classification rules, format each as:
     - "If [Label] = [Value], then [Conditions...]"

2. **Rule Aggregation and Merging**:
   - When provided with multiple rules for the same target, identify and merge overlapping or consistent conditions.
   - For continuous values, convert individual point predictions into interval rules using:
     - "[Target Variable] ∈ [min_value, max_value]"

Assume that the input may refer to **any dataset and any set of feature names**. 
Adapt dynamically and do not assume domain-specific semantics unless provided.

Always format your output with clarity, structure, and in a way that is useful for further prompt construction or dataset simulation.
"""
    # Initialize OpenAI client
    client = OpenAI(
        api_key="sk-zk2283a5f2c353dc6462303ff2d6aadd8706819afeac1f3c",
        base_url="https://api.zhizengzeng.com/v1/"
    )

    # Make API call
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        top_p=0.95
    )

    generated_data = response.choices[0].message.content
    print(generated_data)
    
    # Save generated data to output file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()
