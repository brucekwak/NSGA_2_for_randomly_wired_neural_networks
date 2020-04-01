import numpy as np
import pandas as pd


# [Reference] https://pythonhealthcare.org/tag/pareto-front/
# Scores (np array)를 input으로 받아서, 해당 scores 에서의 파레토 프론티어에 해당하는 점들의 인덱스를 반환해주는 함수
# e.g. scores = (300, 2) np array
#       => 점이 300개 & 최대화해야하는 objective 가 2개
#    output = [1, 3, 5, 150, 199]
#       => scores[1], scores[3], ..., scores[199] 가 파레토 프론티어임
def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


# train_logging.log 에서 search history 읽어오기 위한 함수
def get_info_from_train_log(train_log):
    niter = len(train_log)

    npop = len(train_log['0'])

    objs_chromo = []
    objs_fitness = []
    gen_num = []
    for i in range(niter):
        gen_num.extend([i for _ in range(npop)])
        chromo_i = [train_log[str(i)][j][0] for j in range(npop)]
        fitness_i = [train_log[str(i)][j][1] for j in range(npop)]  # [-val_acc, params]    
        objs_fitness.append(fitness_i)
        objs_chromo.append(chromo_i)

    objs_fitness = np.array(objs_fitness)
    epoch = list(range(niter))    
    
    objs_fitness[:,:,0]= -1*objs_fitness[:,:,0]  # -val_acc => +val_acc

    fit_1_val_acc_list = objs_fitness[:,:,0].reshape(-1).tolist()
    # val_accuracy 는 - 붙어있는채로 사용
    # => minimize 하는 pareto frontier 찾는 함수 그대로 사용
    
    fit_2_params_list = objs_fitness[:,:,1].reshape(-1).tolist()
    idxs = [i for i in range(len(fit_1_val_acc_list))]
    
    search_history_df = pd.DataFrame({'gen':gen_num,'idx': idxs, 'val_acc':fit_1_val_acc_list, 'params': fit_2_params_list})
    
    objs_chromo = sum(objs_chromo, [])  # sum(iterable, start)
    
    return search_history_df, objs_chromo


# fitness_np (np.array) => chromosome들의 fitness 가 순서대로 담겨있는 np.array
#    fitness_np[:, 0] => val_acc
#    fitness_np[:, 1] => params
def find_pareto_idx(fitness_np):

    # 1) params 에 - 붙이기 => (값이 클수록 좋은) score 로 만들기
    fitness_np[:, 1] = -fitness_np[:, 1]

    # 2) 파레토 프론티어 찾기
    pareto_idx = list(identify_pareto(fitness_np))
    pareto_idx.sort(reverse=True)  # 내림차순 정렬
    
    # 3) fitness_np 는 원래대로
    fitness_np[:, 1] = -fitness_np[:, 1]    
    
    return pareto_idx


def get_all_frontier(search_history_df, objs_chromo):
    # fitness 부분만 빼오기
    fitness_np = search_history_df[['val_acc', 'params']].values  # np.array

    pareto_chromo_list = []
    pareto_fitness_list = []

    while fitness_np.shape[0] != 0:
        # 파레토 프론티어의 idx 구하기
        pareto_idx = find_pareto_idx(fitness_np)

        # chromo & fitness 구하기
        pareto_chromo = [objs_chromo[i] for i in pareto_idx]    
        pareto_fitness = fitness_np[pareto_idx, :]

        pareto_chromo_list.append(pareto_chromo)
        pareto_fitness_list.append(pareto_fitness)

        # 다음 턴으로 가기 전에 objs_chromo & fitness_np 에서 이번 프론티어 지우기
        for i in pareto_idx:
            objs_chromo.pop(i)
            fitness_np = np.delete(fitness_np, i, axis=0)
    
    return pareto_fitness_list, pareto_chromo_list


# fitness_np (np.array) => 해당 train_log 에 담겨있는 모든 chromosome 의 fitness 가 순서대로 담겨있는 np.array
#    fitness_np[:, 0] => val_acc
#    fitness_np[:, 1] => params
# pareto_front_np (np.array)
# pareto_chromos (list)
def find_pareto_frontier_from_train_log(train_log):
    niter = len(train_log)

    npop = len(train_log['0'])

    objs_chromo = []
    objs_fitness = []
    gen_num = []
    for i in range(niter):
        gen_num.extend([i for _ in range(npop)])
        chromo_i = [train_log[str(i)][j][0] for j in range(npop)]
        fitness_i = [train_log[str(i)][j][1] for j in range(npop)]  # [-val_acc, params]    
        objs_fitness.append(fitness_i)
        objs_chromo.append(chromo_i)

    objs_fitness = np.array(objs_fitness)
    epoch = list(range(niter))

    objs_fitness[:,:,0]= -1*objs_fitness[:,:,0]  # -val_acc => +val_acc

    fit_1_val_acc_list = objs_fitness[:,:,0].reshape(-1).tolist()
    # val_accuracy 는 - 붙어있는채로 사용
    # => minimize 하는 pareto frontier 찾는 함수 그대로 사용
    
    fit_2_params_list = objs_fitness[:,:,1].reshape(-1).tolist()
    idxs = [i for i in range(len(fit_1_val_acc_list))]
    
    search_history_df = pd.DataFrame({'gen':gen_num,'idx': idxs, 'val_acc':fit_1_val_acc_list, 'params': fit_2_params_list})
    
    # 0) fitness 부분만 빼오기
    fitness_np = search_history_df[['val_acc', 'params']].values  # np.array

    # 1) params 에 - 붙이기 => (값이 클수록 좋은) score 로 만들기
    fitness_np[:, 1] = -fitness_np[:, 1]

    # 2) 파레토 프론티어 찾기
    pareto_idx = identify_pareto(fitness_np)
    pareto_front_np = fitness_np[pareto_idx, :]  # deep copy

    # 파레토 프론티어 찾은 뒤, params 에 - 붙였던거 다시 부호 바꾸기
    fitness_np[:, 1] = -fitness_np[:, 1]
    pareto_front_np[:, 1] = -pareto_front_np[:, 1]

    # 3) 파레토 프론티어에 있는 크로모좀 리스트 만들기
    pareto_chromos = []
    for idx in list(pareto_idx):
        i = int(idx / npop)   # e.g. 33 => 1 * 20 + 13 => 1 gen 의 14번째 => objs_chromo[1][13]  ## 각각 0번째 ~ 19번째 있음
        j = idx - i*npop
        temp_chromo = objs_chromo[i][j]
        pareto_chromos.append( temp_chromo )   
    
    return fitness_np, pareto_front_np, pareto_chromos