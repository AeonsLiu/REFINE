from openai import OpenAI

prompt = """
You are an intelligent assistant tasked with applying a self-consistency methodology to combine multiple sets of regression-derived interval-based rules. 

You have received five independently generated rule-sets from multiple executions of previous analysis steps. Each rule-set consists of interval-based rules predicting continuous-values (e.g., MMSE) with specific conditions. Your goal is to integrate these five sets of rules into one consistent, summarized, and robust rule set.

### Input: (Insert your five independently generated rule-sets below)

Rule Set 1:
- If Diagnosis = 0, then: 
  - MMSE > 17.89 and Age <= 73.50 and (SystolicBP <= 111.00 or CholesterolTriglycerides <= 128.98)
  - or FunctionalAssessment <= 6.57 and BMI <= 17.36
  - or Age > 73.50 and MMSE > 16.72 and AlcoholConsumption <= 11.63
- If Diagnosis = 1, then: 
  - FunctionalAssessment <= 6.57 and BMI > 17.36 and SystolicBP <= 138.50 
  - or (FunctionalAssessment > 6.57 and CholesterolLDL <= 156.65 and MMSE <= 15.62) 
  - or Age > 73.50 and MMSE <= 16.72 
  - or Age <= 73.50 and Smoking <= 0.50 and BMI <= 27.95

Rule Set 2:
1. **If Diagnosis = 0, then MMSE > 17.89 and Age > 73.50 and SystolicBP <= 168.50 and CholesterolTriglycerides <= 192.86 and AlcoholConsumption > 9.55.  
2. **If Diagnosis = 1, then MMSE <= 17.89 and Age <= 73.50 and BehavioralProblems > 0.50 and ADL <= 5.06 and Confusion <= 0.50. 

Rule Set 3:
- **If Diagnosis = 0**, then MMSE > 17.89 and SystolicBP ≤ 168.50 and FunctionalAssessment > 6.57.
- **If Diagnosis = 1**, then MMSE ≤ 16.72 and FunctionalAssessment ≤ 6.57 and ADL ≤ 5.06 and AlcoholConsumption ≤ 9.55.

Rule Set 4:
- **If Diagnosis = 0**, then:
  - **MMSE > 17.89**  
  - and **FunctionalAssessment > 6.57**  
  - and **SystolicBP ≤ 168.50**  
  - and **DietQuality ≤ 8.07**  
  - and **CholesterolTriglycerides ≤ 128.98**
- **If Diagnosis = 1**, then:
  - **MMSE ≤ 17.89**  
  - and **ADL ≤ 5.23**  
  - and **BMI ≤ 27.95**  
  - and **CholesterolHDL ≤ 68.54**  
  - and **AlcoholConsumption ≤ 11.63**

Rule Set 5:
**Diagnosis = 0:**  
- MMSE > 16.72 and CholesterolLDL > 130.67 and SystolicBP ≤ 168.50 and ADL > 5.06.  
**Diagnosis = 1:**  
- MMSE ≤ 16.72 and ADL ≤ 5.06 and Smoking ≤ 0.50 and SystolicBP > 111.00.

### Step 1: Identify Consistent and Robust Rule Patterns

For each predicted feature (e.g., Diagnosis), examine all five rule-sets and perform the following:

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
- Avoids overly broad intervals, prioritizing clarity and consistency.
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


client = OpenAI(
    api_key = "your_api_key",
    base_url = "your_base_url"
)

response = client.chat.completions.create(
    model="gpt-4o-2024-0806",
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": prompt},
    ],
    temperature=0.9,
    top_p  = 0.95
)

generated_data = response.choices[0].message.content
print(generated_data)
