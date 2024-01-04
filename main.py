import json
import openai
openai.api_key = "81a03e872d294b499f6ce530185d811c"
openai.api_base = "https://fudancanadaeast.openai.azure.com/"  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = "azure"
openai.api_version = "2023-05-15"
import time
from tqdm import tqdm
import os
import sys
import logging
import argparse
import fitlog
logger = logging.getLogger(__name__)

import requests
from requests.auth import HTTPBasicAuth
from rank_bm25 import BM25Okapi
from utils import *   
import random

############# beta for ChatGPT ###################

args = get_args()
# fitlog.set_log_dir("test/")         # 设定日志存储的目录
# fitlog.add_hyper(args)              # 保存参数信息
# fitlog.add_hyper_in_file(__file__)  # 保存代码信息


set_logger(args.log_path)
data = []  # Define the 'data' variable
if args.seed > 0:
    random.seed(args.seed)
    random.shuffle(data)

if 'bbq' in args.task:
    #测试
    from dataloader import load_data_bbq as load_data
    from settings import task_descrip_prompt_bbq as task_descrip_prompt
    from settings import summary_prompt_bbq as summary_prompt
    from settings import line_data_to_key_bbq as line_data_to_key
    from settings import check_true_or_false_bbq as check_true_or_false
    from settings import convert_prompt_bbq as convert_prompt
    from settings import construct_summary_prompt_bbq as construct_summary_prompt
    from utils import reasoning_rules_bbq as reasoning_rules
elif 'tweet' in args.task:
    from dataloader import load_data_tweet as load_data
    from settings import task_descrip_prompt_tweet, summary_prompt_tweet
    task_descrip_prompt, summary_prompt = task_descrip_prompt_tweet[args.task], summary_prompt_tweet[args.task]
    from settings import line_data_to_key_tweet as line_data_to_key
    from settings import check_true_or_false_tweet as check_true_or_false
    from settings import convert_prompt_tweet as convert_prompt
    from settings import construct_summary_prompt_tweet as construct_summary_prompt
    from utils import reasoning_rules_tweet as reasoning_rules
elif 'bbh' in args.task:
    from dataloader import load_data_bbh as load_data
    from settings import task_descrip_prompt_bbh as task_descrip_prompt
    from settings import summary_prompt_bbh
    summary_prompt = summary_prompt_tweet[args.task]
    from settings import line_data_to_key_bbh as line_data_to_key
    from settings import check_true_or_false_bbh as check_true_or_false
    from settings import convert_prompt_bbh as convert_prompt
    from settings import construct_summary_prompt_bbh as construct_summary_prompt
    from utils import reasoning_rules_bbh as reasoning_rules
else:
    print('Not Implemented Yet')
    raise AttributeError

from settings import formulate_rule_prompt, check_rules_example

if __name__ == '__main__': 
    try:
        args = get_args()
        # print("args===========",args)
        fitlog.set_log_dir("fitlogs/")         # 设定日志存储的目录
        fitlog.add_hyper(args)              # 保存参数信息
        fitlog.add_hyper_in_file(__file__)  # 保存代码信息

        rule_book = RuleBook(args.task, logger)
        rule_book.load_check_rule_example(check_rules_example, convert_prompt, task_descrip_prompt, check_true_or_false)
        rule_book.load_construct_summary_prompt(construct_summary_prompt, summary_prompt)
        data = load_data(args.data_dir, args.task,args.test_data_ratio)

        faults = []

        count, tokens = 0, 0
        for line_data in tqdm(data[:]):

            query_prompt = convert_prompt(line_data, task_prompt=task_descrip_prompt)

            if len(rule_book.valid_samples) >= 1:
                
                query = line_data_to_key(line_data)
                top_rules = rule_book.retrieval_rules_bm25(query, line_data, n_sample=5, n_rule=args.num_rule_per_sample)
                query_prompt = formulate_rule_prompt(top_rules) + query_prompt

            messages = [{'role': 'user', 'content': query_prompt}]
            messages, response, tokens = post_message(messages, tokens, logger)

            answer = messages[1]['content'].replace('assistant', '').replace(':', '').strip()
            correct = check_true_or_false(answer, line_data, args.task)

            if not correct: faults.append(count)

            count += 1
            logger.info(messages[0]['role'] + ' : ' + messages[0]['content'])
            logger.info(messages[1]['role'] + ' : ' + messages[1]['content'])
            logger.info(f'Faults: {faults}.\t' + str(len(faults)) + f' faults from {count} samples')
            logger.info('==='*20)

            if not correct:

                messages, response, tokens = reasoning_rules(messages, tokens, logger, line_data, args.task)

                raw_rules = response['choices'][0]['message']['content'].split('\n')
                valid_rules = get_valid_rules(raw_rules)
                for vr in valid_rules: logger.info('Valid Rule: ' + vr)
                
                # Check rules
                logger.info('***'*10)
                logger.info('Check rules...')
                logger.info('***'*10)
                success_rules, tokens = check_rules_example(valid_rules, line_data, tokens, logger, convert_prompt, task_descrip_prompt, check_true_or_false, args.task)
                
                logger.info(str(len(success_rules)) + ' rules correct the answer')
                for rule in success_rules: logger.info('Successful Rule: ' + rule)
                
                line_data['index'] = count
                sample = line_data_to_key(line_data)

                if len(success_rules) > 0:
                    
                    already_exist_rules = rule_book.update_rules(sample, line_data, success_rules, count)
                    
                    new_rules = [sr for sr in success_rules if sr not in already_exist_rules]
                    
                    if len(rule_book.valid_samples) >= 20 and len(new_rules) > 0:
                        new_tokens = rule_book.check_contradictory_identical(new_rules)
                        tokens += new_tokens

                else:
                    logger.info('Fail to correct the current example')
                                    
                    if len(rule_book.fail_samples) > 10:    
                        new_tokens = rule_book.summary_and_update(sample, line_data, n_sample=2)
                        tokens += new_tokens
                    else:
                        rule_book.fail_samples.append(sample)
                        rule_book.fail_sample_line_data[sample] = line_data
                
                rule_book.compress_lru(threshold=args.num_rule_limit)

            if (count) % 20 == 0:
                    
                rule_book.save(path=args.rule_path)
            
            # for line_data in tqdm(data[:]):
            fitlog.add_metric(len(faults), name='Faults', step=count)
            fitlog.add_metric(round(( len(faults)/count) * 100, 2), name='Err', step=count)
            fitlog.add_metric(round((1 - len(faults)/count) * 100, 2), name='Acc', step=count)
            # fitlog.add_metric(tokens, name='Tokens', step=count)


        rule_book.log_rules()
        logger.info('Faults: ' + str(faults))
        logger.info('Err: ' + str(len(faults)) + '/' + str(count))
        logger.info('Acc: ' + str(1 - len(faults)/(count)))
        logger.info('All Tokens: ' + str(tokens))
        logger.info('==='*20)
        fitlog.add_best_metric(len(faults), name='Faults')
        fitlog.add_best_metric(round(( len(faults)/count) * 100, 2), name='Err')
        fitlog.add_best_metric(round((1 - len(faults)/count) * 100, 2), name='Acc')
        fitlog.add_best_metric(tokens, name='Tokens')

    except Exception as e:
        logger.error("An error occurred: {}".format(e))
    finally:
        fitlog.finish()                     # finish the logging
