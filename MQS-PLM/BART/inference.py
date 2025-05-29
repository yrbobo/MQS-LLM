
import argparse
import os.path
import torch
from tqdm import tqdm
import data_loader
from transformers import BartTokenizer
from FtModel import FtModel
import json
from log_utils import LogTool
import logging
def model_test(args, model, tokenizer, test_set, logger):
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    results = []
    with torch.no_grad():
        for src_ids, tgt_ids, src_mask, tgt_mask, labels in tqdm(test_set):
            generate_ids = model.model.generate(input_ids=src_ids, attention_mask=src_mask,
                                                max_length=args.max_tgt_length, num_beams=args.num_beams,
                                                early_stopping=True)
            pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                    generate_ids]
            tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                   tgt_ids]
            src = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                   src_ids]
            for s, t, p in zip(src, tgt, pred):
                results.append({
                    "chq": s,
                    "faq": t,
                    "pred": p
                })
                logger.info(f"CHQ: {s} || PRED: {p} || FAQ: {t}")
    
    output_jsonl_path = os.path.join(args.log_path, args.dataset, f"{args.model}_{args.dataset}_results.jsonl")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MQS with Seq2Seq PLM")
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--dataset_path", type=str, default='')
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_src_length", type=float, default=128)
    parser.add_argument("--max_tgt_length", type=float, default=20)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--log_path", type=str, default='log')

    args = parser.parse_args()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    test_set_raw = data_loader.load_dataset(args.dataset_path)
    test_set = data_loader.DatasetIterator(tokenizer, test_set_raw, args.batch_size, args.device, args.max_src_length)

    model = FtModel()
    model = model.to(args.device)
    log_name = f"{args.model}_{args.dataset}.log"
    log_tool = LogTool(log_file=os.path.join(args.log_path, args.dataset, log_name), log_level=logging.DEBUG)
    logger = log_tool.get_logger()

    model_test(args, model, tokenizer, test_set, logger)
    


