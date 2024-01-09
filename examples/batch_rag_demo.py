# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import json
import os
import time

from tqdm import tqdm

from chatpdf import ChatPDF

pwd_path = os.path.abspath(os.path.dirname(__file__))


def get_truth_dict(jsonl_file_path):
    truth_dict = dict()

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            input_text = entry.get("question", "")
            output_text = entry.get("answer", "")
            if input_text and output_text:
                truth_dict[input_text] = output_text

    return truth_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_model", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="medical_corpus.jsonl")
    parser.add_argument('--query_file', default="medical_query.txt", type=str, help="query file, one query per line")
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--test_size", type=int, default=-1)
    args = parser.parse_args()
    print(args)

    model = ChatPDF(
        sim_model_name_or_path=args.sim_model,
        gen_model_type=args.gen_model_type,
        gen_model_name_or_path=args.gen_model,
        lora_model_name_or_path=args.lora_model,
        corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
    )
    print(f"chatpdf model: {model}")

    truth_dict = dict()
    for i in args.corpus_files.split(','):
        if i.endswith('.jsonl'):
            tmp_truth_dict = get_truth_dict(i)
            truth_dict.update(tmp_truth_dict)
    # test data
    if args.query_file is None:
        examples = ["肛门病变可能是什么疾病的症状?", "膺窗穴的定位是什么?"]
    else:
        with open(args.query_file, 'r', encoding='utf-8') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    if args.test_size > 0:
        examples = examples[:args.test_size]
    print("Start inference.")
    t1 = time.time()
    counts = 0
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    eval_batch_size = args.eval_batch_size
    for batch in tqdm(
            [
                examples[i: i + eval_batch_size]
                for i in range(0, len(examples), eval_batch_size)
            ],
            desc="Generating outputs",
    ):
        responses = []
        for i in batch:
            response, reference_results = model.predict(i)
            responses.append(response)
        results = []
        for example, response in zip(batch, responses):
            truth = truth_dict.get(example, '')
            print(f"===")
            print(f"Input: {example}")
            print(f"Output: {response}")
            print(f"Truth: {truth}\n")
            results.append({"Input": example, "Output": response, "Truth": truth})
            counts += 1
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    t2 = time.time()
    print(f"Saved to {args.output_file}, Time cost: {t2 - t1:.2f}s, size: {counts}, "
          f"speed: {counts / (t2 - t1):.2f} examples/s")
