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

def construct_split(filtered_data,tag="MQA"):
    new_cot_data_list = []
    for item in tqdm(filtered_data):
        print(f"construct json file\n")
        context = item["context"]
        question=item['question']
        _id=item['_id']
        difficulty=item['difficulty']
        instruction_cot=prompt_lbv2_sum.format(context=context)
        # instruction_cot=prompt_lbv2_cot.format(context=context, question=question,choice_A=item['choice_A'],choice_B=item['choice_B'],choice_C=item['choice_C'],choice_D=item['choice_D'])
        new_cot_data_list.append({"id": id, "instruction": instruction_cot, "output": item['answer'], "id":_id,"difficulty":difficulty,"question":item['question'],"system": "You are a helpful assistant."})

    print(f"size of new_cot_data_list: {len(new_cot_data_list)}")
    with jsonlines.open(f"longbenchv2/{tag}_over_64K_sum.jsonl", 'w') as writer:
        writer.write_all(new_cot_data_list)

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
    construct_split(filter_data,tag=tag)

if __name__ == "__main__":
    split_list=["Single-Document QA","Multi-Document QA","Long In-context Learning"]
    split_tag=["SQA","MQA","LongICL"]
    for (split,tag) in zip(split_list,split_tag):
        preprocess_longbenchv2(split, tag)
