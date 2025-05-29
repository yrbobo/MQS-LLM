
from openai import OpenAI
import os
import json
import time
from tqdm import tqdm
import argparse

client = OpenAI(api_key="", base_url="")

# Evaluation prompt template based on G-Eval (Adapted for Medical Healthcare Question Summarization)
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summarized question generated from a medical healthcare question. 
Your task is to rate the summarized question on one metric, and the score will be output in JSON format.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Source Medical Question:

{document}

Summarized Question:

{summary}

EXAMPLE JSON OUTPUT:
{{
    "{metric_name}": "3"
}}

OUTPUT: 

"""

# Metric 1: Relevance
RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source medical question. \
The summarized question should include only the most critical information from the source question. \
Annotators were instructed to penalize summaries that contained redundancies, excess information, or omitted key medical details.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summarized question and the source medical question carefully.
2. Compare the summarized question to the source question and identify the main medical issue or concern.
3. Assess how well the summarized question captures the main medical issue or concern, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

# Metric 2: Coherence
COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the logical flow and clarity of the summarized question. \
The summarized question should be well-structured and easy to understand, clearly conveying the medical issue or concern. \
It should not be a disjointed collection of related information but should build a coherent question about the medical topic.
"""

COHERENCE_SCORE_STEPS = """
1. Read the source medical question carefully and identify the main medical issue or concern.
2. Read the summarized question and compare it to the source question. Check if the summarized question clearly and logically presents the main medical issue or concern.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency
CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summarized question and the source medical question. \
A factually consistent summarized question contains only information that is directly supported by the source question. \
Annotators were also asked to penalize summaries that contained hallucinated or unsupported medical details.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the source medical question carefully and identify the key medical facts and details.
2. Read the summarized question and compare it to the source question. Check if the summarized question contains any factual errors or unsupported medical details.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

# Metric 4: Fluency
FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the linguistic quality of the summarized question in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summarized question has many errors that make it hard to understand or sound unnatural.
2: Fair. The summarized question has some errors that affect the clarity or smoothness of the text, but the main medical issue is still comprehensible.
3: Good. The summarized question has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summarized question and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""


def get_geval_score(
    criteria: str, steps: str, document: str, summary: str, metric_name: str
):
    max_retries = 10000
    retry_count = 0

    while retry_count < max_retries:
        try:
            prompt = EVALUATION_PROMPT_TEMPLATE.format(
                criteria=criteria,
                steps=steps,
                metric_name=metric_name,
                document=document,
                summary=summary,
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=32,
                top_p=1,
                stream=False,
                response_format={
                    'type': 'json_object'
                }
            )
            content = response.choices[0].message.content
            content = content.lower()
            json_obj = json.loads(content)
            return json_obj[metric_name.lower()]

        except Exception as e:
            print(f"Request failed with error: {e}. Retrying in 5 seconds...")
            retry_count += 1
            time.sleep(5)

    raise Exception(f"Failed to get geval score after {max_retries} retries.")


evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}

def save_eval(path, obj):
    if not os.path.exists(path):
        new_list = [obj]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(new_list, f, ensure_ascii=False, indent=4)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            new_list = json.load(f)
            new_list.append(obj)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(new_list, f, ensure_ascii=False, indent=4)

def eval(chq, pred, faq):
    save_obj = {
        "chq": chq,
        "pred": pred,
        "faq": faq,
        "eval": {}
    }
    for eval_type, (criteria, steps) in tqdm(evaluation_metrics.items()):
        result = get_geval_score(criteria, steps, chq, pred, eval_type)
        
        score_num = int(result.strip())
        save_obj["eval"][eval_type] = score_num
    
    return save_obj

def run(test_file, wait_time):
    if wait_time > 0:
        time.sleep(wait_time)
    save_file_path = test_file.replace('results.jsonl', 'eval.json')

    print(f"Processing {test_file}...")
    with open(test_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            obj = json.loads(line)
            if "chq" not in obj:
                chq = obj["data"]["chq"]
            else:
                chq = obj["chq"]
            if "faq" not in obj:
                faq = obj["data"]["faq"]
            else:
                faq = obj["faq"]
            pred = obj["pred"]
            save_obj = eval(chq, pred, faq)
            save_eval(save_file_path, save_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EVAL')
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--wait_time', type=int)
    args = parser.parse_args()
    run(args.test_file, args.wait_time)
        

                
