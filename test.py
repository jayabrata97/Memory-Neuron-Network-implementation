from copy import deepcopy
import numpy as np
import pandas as pd
import pickle

# preprocess test data
names_vehicle = ['t_stamp', 'id', 'pos_x', 'pos_y', 'trf_light']

for env_idx in range(2, 5):
    for task_idx in range(2, 3):
        df_vehicle = pd.read_csv(f"Env{env_idx}/vehicle/data_env{env_idx}_task{task_idx}_vehicle.csv", names=names_vehicle)
        
        df_vehicle = df_vehicle.sort_values(by=["trf_light", "id", "t_stamp"])
        df_vehicle = df_vehicle.reset_index(drop=True)
        
        split_idx = df_vehicle["trf_light"].idxmax(axis=0)
        df_vehicle_trf_0 = df_vehicle.iloc[:split_idx, :]
        df_vehicle_trf_1 = df_vehicle.iloc[split_idx:, :]
        
        np_vehicle_trf_0 = np.array(df_vehicle_trf_0[["id","pos_x", "pos_y"]], dtype="float32")
        #print(np_vehicle_trf_0.shape)
    
# trajectory data spliting
id = -1.0
vehicle_trajectory_list = []
split_ptr = 0
count = 0

for ctr in range(np_vehicle_trf_0.shape[0]):
    pos_change_x = abs(np_vehicle_trf_0[ctr, 1] - np_vehicle_trf_0[ctr - 1, 1]) > 1
    pos_change_y = abs(np_vehicle_trf_0[ctr, 2] - np_vehicle_trf_0[ctr - 1, 2]) > 1
    pos_change = pos_change_x or pos_change_y
    
    if np_vehicle_trf_0[ctr, 0] != id:
        count += 1
        id = np_vehicle_trf_0[ctr, 0]
        vehicle_trajectory_list.append(deepcopy(np_vehicle_trf_0[split_ptr:ctr, 1:]))
        split_ptr = ctr
        
# test every 1s, 3s and 5s

with open("trained_models/trained_mnn_vehicle_trf_0_v2.obj", "rb") as f:
    mnn = pickle.load(f)

n_test_samples = 0
error = []

#vehicle_trajectory_list = [vehicle_trajectory_list[2][:40, :]]
#print(vehicle_trajectory_list)

for t_idx in vehicle_trajectory_list:
    ctr = 1
    steps = 30 # for 3 seconds
    
    while ctr < len(t_idx):
        del_pos = t_idx[ctr] - t_idx[ctr - 1]
        pred_trajectory = np.zeros((steps + 1, 2))
        init_pos = t_idx[ctr]
        pred_trajectory[0] += init_pos
        for step in range(1, steps+1):
            #print(del_pos)
            mnn.feedforward(del_pos)
            del_pos = mnn.output_nn
            #print(del_pos, pred_trajectory)
            pred_trajectory[step] = pred_trajectory[step - 1] + del_pos
        #print(ctr,len(t_idx))
        #print(pred_trajectory)
        if ctr + steps >= len(t_idx):
            break
        #print(t_idx[ctr:ctr+steps+1])
        error.append(np.sqrt(np.sum(np.abs(pred_trajectory - t_idx[ctr:ctr+steps+1])**2)/steps))
        ctr += 5
        
print('RMSE value: ', np.mean(error), len(error))
        
            