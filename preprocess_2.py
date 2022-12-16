import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import random

num_env = 5
num_task = 1

names_vehicle = ['t_stamp', 'id', 'pos_x', 'pos_y', 'trf_light']
names_walker = ['t_stamp', 'id', 'pos_x', 'pos_y']

vehicle_trf_0_train = []
vehicle_trf_1_train = []
vehicle_trajectory_list = []
count = 0

#for env_idx in range(num_env):
for env_idx in range(0, 5): # only considering towns 1-4
    for task_idx in range(num_task):
        df_vehicle = pd.read_csv(f"Env{env_idx}/vehicle/data_env{env_idx}_task{task_idx}_vehicle.csv", names=names_vehicle)
        df_walker = pd.read_csv(f"Env{env_idx}/walker/data_env{env_idx}_task{task_idx}_walker.csv", names=names_walker)
        
        df_vehicle = df_vehicle.sort_values(by=["trf_light", "id", "t_stamp"])
        df_walker = df_walker.sort_values(by=["id", "t_stamp"])
        df_vehicle = df_vehicle.reset_index(drop=True)
        df_walker = df_walker.reset_index(drop=True)

        split_idx = df_vehicle["trf_light"].idxmax(axis=0)
        df_vehicle_trf_0 = df_vehicle.iloc[:split_idx, :]
        df_vehicle_trf_1 = df_vehicle.iloc[split_idx:, :]
        
        np_vehicle_trf_0 = np.array(df_vehicle_trf_0[["id", "pos_x", "pos_y"]], dtype="float32")
        np_vehicle_trf_1 = np.array(df_vehicle_trf_1[["pos_x", "pos_y"]], dtype="float32")
        
        #np_vehicle_trf_0 = np.diff(np_vehicle_trf_0, axis=0)
        #np_vehicle_trf_0 = np_vehicle_trf_0[np.logical_and(abs(np_vehicle_trf_0[:, 0])< 1, abs(np_vehicle_trf_0[:, 1]) < 1), :]
        #np_vehicle_trf_1 = np.diff(np_vehicle_trf_1, axis=0)
        #np_vehicle_trf_1 = np_vehicle_trf_1[np.logical_and(abs(np_vehicle_trf_1[:, 0])< 1, abs(np_vehicle_trf_1[:, 1]) < 1), :]

        id = -1.0
        #vehicle_trajectory_list = []
        split_ptr = 0
        #count = 0

        for ctr in range(np_vehicle_trf_0.shape[0]):
            pos_change_x = abs(np_vehicle_trf_0[ctr, 1] - np_vehicle_trf_0[ctr - 1, 1]) > 1
            pos_change_y = abs(np_vehicle_trf_0[ctr, 2] - np_vehicle_trf_0[ctr - 1, 2]) > 1
            pos_change = pos_change_x or pos_change_y
            
            if np_vehicle_trf_0[ctr, 0] != id:
                count += 1
                id = np_vehicle_trf_0[ctr, 0]
                diff = np.diff(np_vehicle_trf_0[split_ptr:ctr, 1:], axis=0)
                diff = diff[np.logical_and(abs(diff[:, 0])< 1, abs(diff[:, 1]) < 1), :]
                vehicle_trajectory_list.append(diff)
                split_ptr = ctr
        
        #vehicle_trf_0_train.append(np_vehicle_trf_0)
        #vehicle_trf_1_train.append(np_vehicle_trf_1)
        
        #print(np_vehicle_trf_0.shape, np_vehicle_trf_1.shape)
print('count:',count)
print(len(vehicle_trajectory_list))
np.random.shuffle(vehicle_trajectory_list)
vehicle_trajectory_list = random.choices(vehicle_trajectory_list, k=250)
print(len(vehicle_trajectory_list))
        
        
print("Saving data...")


#vehicle_trf_0_train = np.concatenate(vehicle_trf_0_train, axis=0)
vehicle_trf_0_train = np.concatenate(vehicle_trajectory_list, axis=0)

#n, bins, patches = plt.hist(np.abs(vehicle_trf_0_train[:,0]))

#plt.savefig('dist.png')
print(vehicle_trf_0_train.shape)
with open("processed/vehicle_trf_0_train_v3.npy", "wb") as f:
    np.save(f, vehicle_trf_0_train)
    
#vehicle_trf_1_train = np.concatenate(vehicle_trf_1_train, axis=0)
#with open("processed/vehicle_trf_1_train_v3.npy", "wb") as f:
#    np.save(f, vehicle_trf_1_train)
        
#print(f'Mean of trf 0 in env{env_idx} of task {task_idx}\t',np.mean(np_vehicle_trf_0[:,0]),'\t', np.mean(np_vehicle_trf_0[:,1]))
#print(f'Std of trf 0 in env {env_idx} of task {task_idx} \t', np.std(np_vehicle_trf_0[:,0]),'\t', np.std(np_vehicle_trf_0[:,1]))
        
#print(f'Mean of trf 1 in env {env_idx} of task {task_idx} \t',np.mean(np_vehicle_trf_1[:,0]),'\t' ,np.mean(np_vehicle_trf_1[:,1]))
#print(f'Std of trf 1 in env {env_idx} of task {task_idx} \t', np.std(np_vehicle_trf_1[:,0]),'\t' ,np.std(np_vehicle_trf_1[:,1]))
        
