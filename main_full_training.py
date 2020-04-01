import os
import sys
import os
import logging
from easydict import EasyDict
import numpy as np
import pandas as pd
import random
import time
import datetime
from deap import tools
from collections import OrderedDict
from pprint import pprint
import json
import torch

from utils.create_toolbox import evaluate_full_train_cutmix

import argparse


from utils.pareto_front import get_info_from_train_log, get_all_frontier

from utils.dataset import get_train_valid_loader_CIFAR10_cutmix, get_test_loader_CIFAR10


class Full_train:
    def __init__(self, json_file):
        self.root = os.path.abspath(os.getcwd())
        self.param_dir = os.path.join(self.root + '/parameters/', json_file)
        f = open(self.param_dir)
        params = json.load(f)
        pprint(params)
        self.name = params['NAME']

        ## toolbox params
        self.args_train = EasyDict(params['ARGS_TRAIN'])
        self.data_path = params['DATA_PATH']
        self.run_code = params['RUN_CODE']
        self.stage_pool_path = './graph_pool' + '/' + self.run_code + '_' + self.name + '/'   
        self.stage_pool_path_list = []
        for i in range(1, 4):
            stage_pool_path_i = self.stage_pool_path + str(i) + '/'  # eg. [graph_pool/run_code_name/1/, ... ]
            self.stage_pool_path_list.append(stage_pool_path_i)
        
        self.log_path = './logs/' + self.run_code + '_' + self.name + '/'
        self.log_file_name = self.log_path + 'logging.log'
        self.train_log_file_name = self.log_path + 'train_logging.log'
        
        if not os.path.exists(self.stage_pool_path):
            os.makedirs(self.stage_pool_path)
            for i in range(3):
                os.makedirs(self.stage_pool_path_list[i])
                
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
            
        logging.basicConfig(filename=self.log_file_name, level=logging.INFO)
        logging.info('[Start] Full_train class is initialized.')
        logging.info('Start to write log.')
            
        self.num_graph = params['NUM_GRAPH']
        
        
        # logs
        self.log = OrderedDict()
        self.log['hp'] = self.args_train
        self.train_log = OrderedDict()
        
        # training log 불러올 디렉토리
        self.GA_data_path = params['GA_DATA_PATH']
        self.RS_data_path = params['RS_DATA_PATH']


    def train(self):
        ###################################
        # 1. Initialize
        ###################################
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("Initialion starts ...")
        logging.info("Initialion starts at " + now_str)

        
        ###################################
        # 2. Full training - GA pareto frontier
        ###################################
        # 2.1. GA training.log 읽어오기
        with open(os.path.join(self.GA_data_path, "train_logging.log")) as json_file:
            data = json.load(json_file)

        train_log = data['train_log']
        
        # 2.2. pareto frontier 찾기
        search_history_df, objs_chromo = get_info_from_train_log(train_log)
        pareto_fitness_list, pareto_chromo_list = get_all_frontier(search_history_df, objs_chromo)        
        
        pareto_chromos = pareto_chromo_list[0]
        # check
        for ind_i in pareto_chromos:
            print(ind_i)
            
        # 2.3. Dataset & Dataloader
        data_path = './data'
        if self.data_path is not None :
            data_path = self.data_path

        if self.args_train.data == "CIFAR10" :
            # cutmix
            train_loader, val_loader = get_train_valid_loader_CIFAR10_cutmix(data_path, self.args_train.batch_size,
                                                                       valid_size=0.2,
                                                                       shuffle=True,
                                                                       num_workers=self.args_train.workers,
                                                                       pin_memory=True)            
            
            test_loader = get_test_loader_CIFAR10(data_path, self.args_train.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.args_train.workers,
                                                   pin_memory=True)
        
        # 2.4. 파레토 프론티어에 있는 크로모좀들 풀트레이닝
        print("Number of Chromosomes on Pareto Frontier :", len(pareto_chromos)) 
        logging.info("Number of Chromosomes on Pareto Frontier : " + str(len(pareto_chromos))) 

        for idx, ind in enumerate(pareto_chromos):
            num = idx + 1
            print('\t', num, 'th Chromosome - evaluation...')
            logging.info(str(num) + 'th Chromosome - evaluation...')
            train_init_time = time.time()
            
            model_dict = {}

            # Cutmix
            best_prec1, test_prec1, params = evaluate_full_train_cutmix(ind, args_train=self.args_train,
                                                    train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    test_loader=test_loader,
                                                    stage_pool_path_list=self.stage_pool_path_list,
                                                    data_path=self.data_path,
                                                    channels=self.args_train.channels,
                                                    log_file_name=self.log_file_name
                                                   )

            trained_end_time = time.time() - train_init_time
            
            model_dict['model_id'] = ind
            model_dict['val_acc'] = best_prec1
            model_dict['test_acc'] = test_prec1
            model_dict['params'] = params
            model_dict['time'] = trained_end_time
            model_dict['epoch'] = self.args_train.epochs

            # log 기록 - initialize (= 0th generation)
            self.train_log[str(idx)] = model_dict
            self.save_log()
            print("\t trained_end_time: ", trained_end_time)
            logging.info('\t trained_end_time: %.3fs' % (trained_end_time))
                 
        
    # Save Log
    def save_log(self):
        self.log['train_log'] = self.train_log

        with open(self.train_log_file_name, 'w', encoding='utf-8') as make_file:
            json.dump(self.log, make_file, ensure_ascii=False, indent='\t')            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Parameter Json file')
    
    args = parser.parse_args()
    
    trainer = Full_train(json_file=args.params)

    trainer.train()
    
    print("Finished.")
