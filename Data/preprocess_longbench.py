import json
import jsonlines
import json_repair
import pandas as pd
import numpy as np
import torch
import shutil
import glob
import re
from pathlib import Path
from typing import Union, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset

here = Path(__file__)
BASE_DIR = here.parent

prompt_lbv2_sum ="""
Please read the following text and write a one-page summary.
{context}
"""

prompt_lbv2_nocot="""
Please read the following text and answer the question below.

{context}

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)".
"""
prompt_lbv2_cot="""
You are given a long document such as a story, meeting script, a news article, etc, and a question. Your task is to answer the question based on the information provided in the document. You should follow the instructions below to provide an accurate reasoning path, as well as a answer chosen from ABCD options to the question:

**Instructions:**
Step 1. **Reasoning:** First retrieve all relevant information, then deduce the correct answer. Begin by carefully reading the provided context. Identify and extract all relevant information that is directly related to the question. Be succinct and only extract the most important excerpts that will help you answer the question. Finally, deduce the correct answer based on the retrieved information.
Step 2. **Answer:** Using the information you have retrieved, and your deduction, answer the question as concisely as you can, using a single phrase or sentence if possible. Ensure that your answer should be brief and to the point.
Step 3. **Format Your Response:** Present your response in JSON format, comprising two components: "reasoning" and "answer". The "reasoning" section should detail your thought process, including the breakdown of the question, the relevant excerpts (indicated by [Excerpt xxx] at the start), and the derived conclusion. Ensure that each excerpt is an exact match to the original document. Limit the number of excerpts to a maximum of 10. The "answer" part should contain your final answer to the question, which is a choice selected from the ABCD options.

Illustrative Examples:

Example #1:

**Context:** [... Saltram is living with the Mulvilles at Wimbledon ... He is not working or producing anything ... He is idle and dependent on others ...]
**Question:** What is Saltram's living situation?
**Choices:**
(A) He is a guest in the home of the Mulvilles.
(B) He is in a hotel.
(C) He is homeless now.
(D) Unkonwn

**Response:**
{{
    "reasoning": "Let me first retrieve relevant excerpts from the document, then answer the question. The question asks about Saltram's living situation. In the document, I can first locate that [Excerpt 1] `Saltram is living with the Mulvilles at Wimbledon`. Additionally, it is mentioned that [Excerpt 2] `He is not working or producing anything` and [Excerpt 3] `He is idle and dependent on others`. From these excerpts, I can deduce that Saltram is a guest in the home of the Mulvilles.",
    "answer": "A"
}}

Now, based on the context provided below, answer the question with a choice selected from ABCD.

**Context:** {context}
**Question:** {question}
**Choices:**
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

**Response:**
"""

def write_jsonl_file_longbenchv2(filtered_data,tag):
    new_data_list = []
    for item in tqdm(filtered_data):
        context = item["context"]
        question=item['question']
        _id=item['_id']
        difficulty=item['difficulty']
        instruction=prompt_lbv2_sum.format(context=context)
        # instruction=prompt_lbv2.format(context=context, question=question,choice_A=item['choice_A'],choice_B=item['choice_B'],choice_C=item['choice_C'],choice_D=item['choice_D'])
        new_data_list.append({"id": id, "instruction": instruction, "output": item['answer'], "id":_id,"difficulty":difficulty,"question":item['question'],"system": "You are a helpful assistant."})

    print(f"size of new_data_list: {len(new_data_list)}")
    with jsonlines.open(f"{BASE_DIR}/longbenchv2/{tag}_over_64K_sum.jsonl", 'w') as writer:
        writer.write_all(new_data_list)

def load_jsonl_file(path_to_file: Union[str, Path]):
    data_list = []
    error_cnt = 0
    with open(path_to_file) as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                data_list.append(data)
            except Exception as e:
                error_cnt += 1
                print(f"Failed loading line {idx}, error: {e}")
                print(line)
    print(f"Failed loading {error_cnt} lines, total {len(data_list)} lines loaded")
    return data_list

def preprocess_longbenchv2(split, tag):
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    filter_data=[]
    print(f"filtering {tag} data\n")
    for data in tqdm(dataset):
        if data['domain']==split and len(data['context'].split())>64*1024: #store long context(>64k) only
            filter_data.append(data)
    write_jsonl_file_longbenchv2(filter_data,tag=tag)

dataset2prompt = {
    "gov_report": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
    ),
    "qmsum": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nQuery: {input}</s>\nTranscript:\n{context}\n\n"
    ),
    "multi_news": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.</s>\n"
        "<s>assistant\nSummary:"
    ),
    "lcc": ("Please complete the code given below. \n{context}Next line of code:\n"),
    "repobench-p": ("Please complete the code given below. \n{context}{input}Next line of code:\n"),
}

dataset2prompt_old = {
    "gov_report": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.</s>\n"
        "<s>assistant\nSummary:"
    ),
    "qmsum": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}</s>\n"
        "<s>assistant\nAnswer:"
    ),
    "multi_news": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nYou are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.</s>\n"
        "<s>assistant\nSummary:"
    ),
    "lcc": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nPlease complete the code given below. \n{context}Now, complete the code given.</s>\n"
        "<s>assistant\n"
    ),
    "repobench-p": (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nPlease complete the code given below. \n{context}Now, complete the code given.</s>\n"
        "<s>assistant\n"
    ),
}

def write_jsonl_file_longbenchv1(filtered_data, tag, is_under_32k):
    new_data_list = []

    prompt_format = dataset2prompt[tag]
    for item in tqdm(filtered_data):
        context = item["context"]
        input=item['input']
        instruction=prompt_format.format(context=context, input=input)
        new_data_list.append({"_id": item["_id"], "instruction": instruction, "output": item['answers'], "system": "You are a helpful assistant."})

    print(f"size of new_data_list: {len(new_data_list)}")

    if is_under_32k:
        with jsonlines.open(f"{BASE_DIR}/longbenchv1/{tag}_under_32K.jsonl", 'w') as writer:
            writer.write_all(new_data_list)
    else:
        with jsonlines.open(f"{BASE_DIR}/longbenchv1/{tag}.jsonl", 'w') as writer:
            writer.write_all(new_data_list)

def preprocess_longbenchv1(split, tag, is_under_32k=False):
    dataset = load_dataset('THUDM/LongBench',split, split='test')
    filter_data=[]
    print(f"filtering {tag} data\n")
    for data in tqdm(dataset):
        if is_under_32k:
            if data['dataset']==split and len(data['context'].split())<32*1024: #store short context(<32k) only
                filter_data.append(data)
        else:
            if data['dataset']==split:
                filter_data.append(data)
    write_jsonl_file_longbenchv1(filter_data,tag,is_under_32k)

if __name__ == "__main__":
    # longbenchv1
    split_list = ["gov_report", "qmsum", "multi_news", "lcc", "repobench-p"]
    split_tag = ["gov_report", "qmsum", "multi_news", "lcc", "repobench-p"]

    for (split,tag) in zip(split_list,split_tag):
        preprocess_longbenchv1(split, tag,False)

    # # longbenchv2
    # split_list=["Single-Document QA","Multi-Document QA","Long In-context Learning"]
    # split_tag=["SQA","MQA","LongICL"]
    # for (split,tag) in zip(split_list,split_tag):
    #     preprocess_longbenchv2(split, tag)
