import argparse
from tqdm import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import LogTool
import os
import logging

def load_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_prompt(data):
    prompt = f'''Please summarize the following medical healthcare question into a concise and clear one-sentence question. Focus on the key issue or main intent:\n
            Medical Healthcare Question: {data["chq"]}\n
            Summarized Question: '''
    return prompt

def build_history_messages(args):
    example_datas = load_dataset(args.dataset_path.replace('test', 'val'))[:args.example_num]
    history_messages = []
    for example_data in example_datas:
        prompt = build_prompt(example_data)
        ans = example_data['faq']
        history_messages.append({"role": "user", "content": prompt})
        history_messages.append({"role": "assistant", "content": ans})

    return history_messages

def run(args, logger):
    if 'Llama' in args.model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    datas = load_dataset(args.dataset_path)
    if args.example_num > 0:
        history_messages = build_history_messages(args)
    else:
        history_messages = None


    results = []

    for data in tqdm(datas):
        prompt = build_prompt(data)
        if history_messages != None:
            messages = [_ for _ in history_messages]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        ref = data['faq']
        logger.info(f'Pred: {pred} || Ref: {ref}')

        filtered_data = {key: value for key, value in data.items() if key != "candidates"}
        
        results.append({
            "prompt": prompt,
            "data": filtered_data,
            "pred": pred
        })


    output_jsonl_path = os.path.join(args.log_path, args.dataset, f"{args.model}_{args.dataset}_results.jsonl")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info(f"Results saved to {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MQS with LLM Summarizer")
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--dataset_path", type=str, default='')
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--example_num", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_path", type=str, default='')
    args = parser.parse_args()

    # os.makedirs(os.path.join(args.log_path, args.dataset), exist_ok=True)
    log_name = f"{args.model}_{args.dataset}.log"
    log_tool = LogTool(log_file=os.path.join(args.log_path, args.dataset, log_name), log_level=logging.DEBUG)
    logger = log_tool.get_logger()

    run(args, logger)