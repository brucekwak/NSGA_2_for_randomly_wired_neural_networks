import os
import sys
import os
import logging
from easydict import EasyDict
import numpy as np
import random
import time
import datetime
from deap import tools
from collections import OrderedDict
from pprint import pprint
import json
import torch


from utils.graph import make_random_graph
from utils.create_toolbox import create_toolbox_for_NSGA_RWNN, evaluate_one_chromo

from utils.dataset import get_train_valid_loader_CIFAR10

import argparse


class Main_train:
    def __init__(self, json_file):
        self.root = os.path.abspath(os.getcwd())
        self.param_dir = os.path.join(self.root + '/parameters/', json_file)
        f = open(self.param_dir)
        params = json.load(f)
        pprint(params)
        self.name = params['NAME']
        
        ## 디렉토리 생성
        graph_pool_dir = './graph_pool/'
        if not os.path.exists(graph_pool_dir):
            os.makedirs(graph_pool_dir)
        
        log_dir = './logs/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        data_dir = './data/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
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
        logging.info('[Start] Main_train class is initialized.')        
        logging.info('Start to write log.')
            
        self.num_graph = params['NUM_GRAPH']
        
        self.toolbox = self.create_toolbox()

        
        # GA params
        self.pop_size = params['POP_SIZE']
        self.ngen = params['NGEN']
        self.cxpb = params['CXPB']
        self.mutpb = params['MUTPB']

        # logs
        self.log = OrderedDict()
        self.log['hp'] = self.args_train  # hp := hyperparameter
        self.train_log = OrderedDict()
        
        # 기존의 training_log 읽어와서, 해당 generation 부터 GA search 진행하기 위한 파라미터
        self.TRAIN_FROM_LOGS = params['TRAIN_FROM_LOGS']
        self.REAL_TRAIN_LOG_PATH = params['REAL_TRAIN_LOG_PATH']
                        

    def create_toolbox(self):
        make_random_graph(self.num_graph, self.stage_pool_path_list)
        return create_toolbox_for_NSGA_RWNN(self.num_graph, self.args_train, self.data_path, self.log_file_name)


    def train(self):
        ###################################
        # 1. Initialize the population
        # toolbox.population은 creator.Individual n개를 담은 list를 반환. (=> population)
        ###################################
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("[GA] Initialion starts ...")
        logging.info("[GA] Initialion starts at " + now_str)
        init_start_time = time.time()

        GA_history_list = []
        
        start_gen = 1
        
        train_loader, val_loader = get_train_valid_loader_CIFAR10(self.data_path, self.args_train.batch_size,
                                                                  valid_size=0.2, 
                                                                  shuffle=True,
                                                                  num_workers=self.args_train.workers,
                                                                  pin_memory=True)        

        # (train_log 읽어와서 특정 generation부터 이어서 train 하지 않고) 처음부터 train 하는 경우
        # population initializatino 부터 GA search 시작
        if self.TRAIN_FROM_LOGS == False:
            print("Training start!")
            logging.info("Training start!")
            pop = self.toolbox.population(n=self.pop_size)
            ###################################
            # 2. Evaluate the population
            ###################################
            invalid_ind = [ind for ind in pop]

            for idx, ind in enumerate(invalid_ind):
                eval_time_for_1_chromo = time.time()
                fitness, ind_model = evaluate_one_chromo(ind, args_train=self.args_train,
                                                 train_loader=train_loader, val_loader=val_loader,
                                                 stage_pool_path_list=self.stage_pool_path_list,
                                                 data_path=self.data_path,
                                                 log_file_name=self.log_file_name)
                
                eval_time_for_1_chromo = time.time() - eval_time_for_1_chromo
                print('\t\t [eval_time_for_1_chromo: %.3fs]' % eval_time_for_1_chromo, idx, 'th chromo is evaluated.')
                logging.info('\t\t [eval_time_for_1_chromo: %.3fs] %03d th chromo is evaluated.' % (eval_time_for_1_chromo, idx))       
                ind.fitness.values = fitness
                GA_history_list.append([ind, fitness])

            # log 기록 - initialize (= 0th generation)
            self.train_log[0] = GA_history_list

            self.save_log()

            # This is just to assign the crowding distance to the individuals
            # no actual selection is done
            pop = self.toolbox.select(pop, len(pop))

            
        # train_log 읽어와서 중간부터 이어서 train 하는 경우
        elif self.TRAIN_FROM_LOGS == True:
            print("Training start!")
            print("Read train_log...")
            logging.info("Training start!")
            logging.info("Read train_log...")
            
            # train_log 읽어오기
            with open(self.REAL_TRAIN_LOG_PATH) as train_log_json_file:
                data = json.load(train_log_json_file)  # hp(=hyperparameter), train_log 있음

            train_log_past = data['train_log']
            niter = len(train_log_past)  # 기록 상 총 init 횟수
            npop = len(train_log_past['0'])
            
            start_gen = niter  # niter = 11 이면, log 상에 0 ~ 10번까지 기록되어있는 것.
            
            # self.train_log 에 읽어온 로그 넣어놓기 (OrderedDict())
            for i in range(niter):
                self.train_log[str(i)] = train_log_past[str(i)]

            self.save_log()
            
            # population 읽어오기
            # train_log 에서 last population 읽어오기
            last_population = train_log_past[str(int(niter)-1)]

            # last population으로 population 만들기
            pop = self.toolbox.population_load(last_population)

            # fitness values 도 읽어오기
            for i in range(len(last_population)):    
                pop[i].fitness.values = last_population[i][1]
            
            pop = self.toolbox.select(pop, len(pop))
            
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("Initialization is finished at", now_str)
        logging.info("Initialion is finished at " + now_str)

        init_time = time.time() - init_start_time
        logging.info("Initialization time = " + str(init_time) + "s")
        print()
                
        ###################################
        # 3. Begin GA
        ###################################
        # Begin the generational process
        for gen in range(start_gen, self.ngen+1):  # self.ngen 남은 횟수를 이어서 돌리기
            # 3.1. log 기록
            now = datetime.datetime.now()
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            print(str(gen) + "th generation starts at" + now_str)
            logging.info(str(gen) + "th generation starts at" + now_str)

            start_gen_time = time.time()
            
            # 3.2. Offspring pool 생성 후, crossover(=mate) & mutation
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # ::2, 1::2 즉, 짝수번째 크로모좀과 홀수번쨰 크로모좀들 차례로 선택하면서 cx, mut 적용
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(ind1, ind2, self.cxpb)

                self.toolbox.mutate(ind1, mutpb=self.mutpb)
                self.toolbox.mutate(ind2, mutpb=self.mutpb)
                del ind1.fitness.values, ind2.fitness.values

            # 3.3. Evaluation
            # Evaluate the individuals with an invalid fitness
            print("\t Evaluation...")
            start_time = time.time()

            # fitness values (= accuracy, flops) 모음
            GA_history_list = []
            
            invalid_ind = [ind for ind in offspring]
            
            for idx, ind in enumerate(invalid_ind):
                eval_time_for_1_chromo = time.time()
                fitness, ind_model = evaluate_one_chromo(ind, args_train=self.args_train,
                                                 train_loader=train_loader, val_loader=val_loader,
                                                 stage_pool_path_list=self.stage_pool_path_list,
                                                 data_path=self.data_path,
                                                 log_file_name=self.log_file_name)
                
                # <= evaluate() returns  (-prec, flops), NN_model
                eval_time_for_1_chromo = time.time() - eval_time_for_1_chromo
                print('\t\t [eval_time_for_1_chromo: %.3fs]' % eval_time_for_1_chromo, idx, 'th chromo is evaluated.')
                logging.info('\t\t [eval_time_for_1_chromo: %.3fs] %03d th chromo is evaluated.' % (eval_time_for_1_chromo, idx))       

                ind.fitness.values = fitness
                GA_history_list.append([ind, fitness])

            # log 기록
            self.train_log[gen] = GA_history_list
            self.save_log()

            eval_time_for_one_generation = time.time() - start_time
            print("\t Evaluation ends (Time : %.3f)" % eval_time_for_one_generation)

            # Select the next generation population
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            gen_time = time.time() - start_gen_time
            print('\t [gen_time: %.3fs]' % gen_time, gen, 'th generation is finished.')

            logging.info('\t Gen [%03d/%03d] -- evals: %03d, evals_time: %.4fs, gen_time: %.4fs' % (
                gen, self.ngen, len(invalid_ind), eval_time_for_one_generation, gen_time))


    # Save Log
    def save_log(self):
        self.log['train_log'] = self.train_log

        with open(self.train_log_file_name, 'w', encoding='utf-8') as make_file:
            json.dump(self.log, make_file, ensure_ascii=False, indent='\t')
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Parameter Json file')
    
    args = parser.parse_args()
    
    trainer = Main_train(json_file=args.params)

    trainer.train()
    trainer.save_log()
