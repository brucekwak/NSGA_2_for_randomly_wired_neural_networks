import random
import glob
import logging
import time
from easydict import EasyDict

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn    # for hardware tunning (cudnn.benchmark = True)
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# genetic algorithm
from deap import base, creator
from deap import tools

# custom module
from utils.graycode import *
from utils.graph import load_graph
from utils.models import RWNN
from utils.train_validate import train, validate, test

# additional package
from warmup_scheduler import GradualWarmupScheduler
from cutmix.utils import CutMixCrossEntropyLoss

#####################################
# Create the toolbox
#####################################
def create_toolbox_for_NSGA_RWNN(num_graph, args_train, stage_pool_path, data_path=None ,log_file_name=None):
    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0 ))  # name, base (class), attribute
    creator.create('Individual', list, fitness=creator.FitnessMin)  # creator.FitnessMaxMin를 attribute로 가짐    
    
    #####################################
    # Initialize the toolbox
    #####################################
    toolbox = base.Toolbox()
    if args_train.graycode :
        gray_len = len(str(grayCode(num_graph-1)))
        IND_SIZE = gray_len * 3
        BOUND_LOW = 0
        BOUND_UP = 1
        toolbox.register('attr_int', random.randint, BOUND_LOW, BOUND_UP)

    else:
        IND_SIZE = 3    # 하나의 individual(=chromosome)은 3개의 graph(또는 stage)를 가짐

        # toolbox.attribute(0, (num_graph-1)) 이렇게 사용함.
        # 즉, 0 ~ (num_grpah - 1) 중 임의의 정수 선택 => 이걸 3번하면 하나의 small graph가 생김
        BOUND_LOW = 0
        BOUND_UP = num_graph-1
        toolbox.register('attr_int', random.randint, BOUND_LOW, BOUND_UP)
        
    toolbox.register('individual', tools.initRepeat,
                     creator.Individual, toolbox.attr_int, n=IND_SIZE)

    toolbox.register('population', tools.initRepeat,
                     list, toolbox.individual)    # n은 생략함. toolbox.population 함수를 뒤에서 실행할 때 넣어줌.    
    
    # crossover
    if args_train.graycode :
        toolbox.register('mate', cxgray, num_graph=num_graph)
    else:
        toolbox.register('mate', tools.cxTwoPoint)

    # mutation
    if args_train.graycode :
        toolbox.register('mutate', mutgray, num_graph=num_graph)
    else:
        toolbox.register('mutate', mutUniformInt_custom, low=BOUND_LOW, up=BOUND_UP)

    # selection
    toolbox.register('select', tools.selNSGA2, nd='standard')
    
    #####################################
    # Seeding a population
    # - 기존의 training_log 읽어와서, 해당 generation 부터 GA search 진행할 때 활용
    #####################################
    # [Reference] https://deap.readthedocs.io/en/master/tutorials/basic/part1.html
    def LoadIndividual(icls, content):
        return icls(content)

    def LoadPopulation(pcls, ind_init, last_population):  # list of [chromosome, [-val_accuracy, flops]]
        return pcls(ind_init(last_population[i][0]) for i in range(len(last_population)))

    toolbox.register("individual_load", LoadIndividual, creator.Individual)

    toolbox.register("population_load", LoadPopulation, list, toolbox.individual_load)
    
    return toolbox


#####################################
# Evaluate - search
#####################################
"""
# fitness function
    input: [0, 5, 10]   하나의 크로모좀.

    1) input인 [0, 5, 10]을 받아서 (0번째, 5번째, 10번째)에 해당하는 그래프 파일 각각 읽어와서 신경망 구축
    2) training (임시로 1 epoch. 실제 실험 시, RWNN과 같은 epoch 학습시키기)
    3) return flops, val_accuracy
"""
def evaluate_one_chromo(individual, args_train, train_loader, val_loader, stage_pool_path_list, data_path=None, channels=109, log_file_name=None):
    # 1) Load graph
    total_graph_path_list = []
    for i in range(3):
        temp = glob.glob(stage_pool_path_list[i] + '*.yaml') # sorting 해줘야함
        temp.sort()
        total_graph_path_list.append( temp )

    graph_name = []

    if args_train.graycode:
        gray_len = len(individual)//3
        for i in range(3):
            # list to string
            tmp = ''
            for j in individual[gray_len*i:gray_len*(i+1)]:
                tmp += str(j)

            # sting to binary to num
            graph_name.append(graydecode(int(tmp)))

    else :
        graph_name = individual

    stage_1_graph = load_graph( total_graph_path_list[0][graph_name[0]] )
    stage_2_graph = load_graph( total_graph_path_list[1][graph_name[1]] )
    stage_3_graph = load_graph( total_graph_path_list[2][graph_name[2]] )
    
    graphs = EasyDict({'stage_1': stage_1_graph,
                       'stage_2': stage_2_graph,
                       'stage_3': stage_3_graph
                      })

    # 2) Build RWNN
    channels = channels
    NN_model = RWNN(net_type='small', graphs=graphs, channels=channels, num_classes=args_train.num_classes, input_channel=args_train.input_dim)
    NN_model.cuda()
    
    params = sum(p.numel() for p in NN_model.parameters())    
    
    # 3) Prepare for train
    NN_model = nn.DataParallel(NN_model)  # for multi-GPU
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(NN_model.parameters(), args_train.base_lr,
                                momentum=args_train.momentum,
                                weight_decay=args_train.weight_decay)
    
    start_epoch  = 0
    best_prec1 = 0    
    
    cudnn.benchmark = True    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.  
    
    
    # 4) Train
    # [Reference] https://github.com/ildoonet/pytorch-gradual-warmup-lr
    cosine_epoch = int(args_train.epochs) - int(args_train.warmup_epochs)  # arg_train = 전체 train epoch, warmup_epoch = warum up epoch
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epoch)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=int(args_train.warmup_epochs), after_scheduler=scheduler_cosine)
    
    for epoch in range(start_epoch, args_train.epochs):
        # train for one epoch        
        train(train_loader, NN_model, criterion, optimizer, scheduler_warmup, epoch, args_train.print_freq, log_file_name)

        # evaluate on validation set
        prec1 = validate(val_loader, NN_model, criterion, epoch, log_file_name)

        best_prec1 = max(prec1, best_prec1)

    return (-best_prec1, params), NN_model  # Min (-val_accuracy, flops) 이므로 val_accuracy(top1)에 - 붙여서 return


#####################################
# Evaluate - full training
#####################################
def evaluate_full_train_cutmix(individual, args_train, train_loader, val_loader, test_loader, stage_pool_path_list, data_path=None, channels=109, log_file_name=None):  # individual
    
    # 1) load graph
    total_graph_path_list = []
    for i in range(3):
        temp = glob.glob(stage_pool_path_list[i] + '*.yaml') # sorting 해줘야함
        temp.sort()
        total_graph_path_list.append( temp )

    graph_name = []

    # args_train 셋팅에서 graycode 변환이 true 인지 확인
    if args_train.graycode:
        ## Decode 해줘야 !
        gray_len = len(individual)//3
        for i in range(3):
            # list to string
            tmp = ''
            for j in individual[gray_len*i:gray_len*(i+1)]:
                tmp += str(j)

            # sting to binary to num
            graph_name.append(graydecode(int(tmp)))

    else :
        graph_name = individual

    stage_1_graph = load_graph( total_graph_path_list[0][graph_name[0]] )
    stage_2_graph = load_graph( total_graph_path_list[1][graph_name[1]] )
    stage_3_graph = load_graph( total_graph_path_list[2][graph_name[2]] )
    
    graphs = EasyDict({'stage_1': stage_1_graph,
                       'stage_2': stage_2_graph,
                       'stage_3': stage_3_graph
                      })

    # 2) build RWNN
    channels = channels
    NN_model = RWNN(net_type='small', graphs=graphs, channels=channels, num_classes=args_train.num_classes, input_channel=args_train.input_dim)
    NN_model.cuda()


    # params 계산
    params = sum(p.numel() for p in NN_model.parameters())

    # 3) Prepare for train### 일단 꺼보자!
    NN_model = nn.DataParallel(NN_model)  # for multi-GPU
#     NN_model = nn.DataParallel(NN_model, device_ids=[0,1,2,3])

    # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda()
    ######################
    # CutMix
    ######################
    criterion = CutMixCrossEntropyLoss(True).cuda()

    optimizer = torch.optim.SGD(NN_model.parameters(), args_train.base_lr,
                                momentum=args_train.momentum,
                                weight_decay=args_train.weight_decay)
    
    start_epoch  = 0
    best_prec1 = 0    
    
    cudnn.benchmark = True    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.  
    
    
    ############################################
    # GradualWarmupScheduler - from ildoonet github
    ############################################
    cosine_epoch = int(args_train.epochs) - int(args_train.warmup_epochs)  # arg_train = 전체 train epoch, warmup_epoch = warum up epoch
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epoch)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=int(args_train.warmup_epochs), after_scheduler=scheduler_cosine)    
    for epoch in range(start_epoch, args_train.epochs):
        # train for one epoch
        scheduler_warmup.step()    # => 100 epoch warmup, after that schedule as after_scheduler
        train(train_loader, NN_model, criterion, optimizer, scheduler_warmup, epoch, args_train.print_freq, log_file_name)

        # evaluate on validation set
        prec1 = validate(val_loader, NN_model, criterion, epoch, log_file_name)
        if prec1 > best_prec1:
            best_prec1 = prec1
            ######################################################################
            # best_validation 나올 때마다 test_prec1 업데이트해서
            #   best_val_accuracy 인 모델의 test accuracy 를 구하자
            ######################################################################
            test_prec1 = test(test_loader, NN_model, criterion, epoch, log_file_name)      

    return (best_prec1, test_prec1, params)  # best_prec1 = 가장 높았던 validation accuracy


############################
# Mutate
############################
# 기존 mutUniformInt 의 xrange() 함수를 range로 수정함.
# indpb: toolbox.mutate() 함수로 사용할 때, MUTPB로 넣어줌 (MUTPB = individual의 각 원소에 mutation 적용될 확률)
def mutUniformInt_custom(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)

    # random 으로 한 stage 정해서 randint 적용하기
    if random.random() < indpb:
        mut_stage = random.randint(0, 2)  # 0, 1, 2 중 랜덤하게 하나 선택
        individual[mut_stage] = random.randint(low, up)

    return individual,
