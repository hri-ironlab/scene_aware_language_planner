import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
# from evaluate import load
import pickle
import json
import re
import copy
from tqdm import tqdm
import os
import random
from pathlib import Path
import sys
import datetime
import pprint
import time

sys.path.append('../datasets/')

import add_preconds
import check_programs
from comm_unity import UnityCommunication
import utils_viz

GPU = 0
if torch.cuda.is_available():
  torch.cuda.set_device(GPU)
OPENAI_KEY = None  # replace this with your OpenAI API key, if you choose to use OpenAI API

# os.system('cd ../datasets/')
# os.system('curl http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip -o programs_processed_precond_nograb_morepreconds.zip')
# os.system('unzip programs_processed_precond_nograb_morepreconds.zip -d ../datasets/')

# Make tasks file
print("Make tasks file")
task_paths = list(Path('../datasets/programs_processed_precond_nograb_morepreconds/executable_programs/').rglob("*.txt"))
print(len(task_paths))
tasks = set()
for path in tqdm(task_paths):
    tasks.add(open(str(path)).read().split('\n')[0])

tasks = list(tasks)
tasks_file = open("../datasets/tasks.txt", "w")
for task in tasks:
    tasks_file.write(task + "\n")
tasks_file.close()


# Make init graphs
print("Make init graphs")
incorrect_samples = []
comm = UnityCommunication()
for init_path in tqdm(task_paths[6000:]):
    graph_path = str(init_path).replace('executable_programs', 'init_and_final_graphs')
    graph_path = str(graph_path).replace('txt', 'json')

    path = str(init_path).replace('executable_programs', 'withoutconds')
    path = path.split('/')
    env_id = path[4][16]
    env_id_unity = int(env_id) - 1
    path = '/'.join(path[:4] + path[5:])

    data = open(str(path)).read().split('\n')
    task = data[0]
    robot_ap = data[4:-1]

    scene_graph = json.load(open(graph_path))
    init_graph = scene_graph['init_graph']

    preconds = add_preconds.get_preconds_script(robot_ap).printCondsJSON()
    info = check_programs.check_script(robot_ap, preconds, graph_path=None, inp_graph_dict=init_graph)
    message, final_state, graph_state_list, graph_dict, id_mapping, info, helper, modif_script = info
    success = (message == 'Script is executable')
    if not success:
        print(init_path)
        incorrect_samples.append(init_path)
        continue

    starting_graph = graph_state_list[0]
    starting_ids_init = np.sort([obj['id'] for obj in starting_graph['nodes']])
    starting_ids = {}
    for node_count in range(len(starting_ids_init)):
        starting_ids[starting_ids_init[node_count]] = 3000 - node_count

    for node_count in range(len(starting_graph['nodes'])):
        starting_graph['nodes'][node_count]['id'] = starting_ids[starting_graph['nodes'][node_count]['id']]

    for edge_count in range(len(starting_graph['edges'])):
        starting_graph['edges'][edge_count]['from_id'] = starting_ids[starting_graph['edges'][edge_count]['from_id']]
        starting_graph['edges'][edge_count]['to_id'] = starting_ids[starting_graph['edges'][edge_count]['to_id']]

    task_dir = '../datasets/init_graphs/' + task + '/'
    graph_dir = task_dir + env_id + '/'
    if os.path.exists(graph_dir):
        init_graph_file = open(graph_dir + str(len(os.listdir(graph_dir)) + 1) + '.json', 'w')
    else:
        os.makedirs(graph_dir)
        init_graph_file = open(graph_dir + '1.json', 'w')

    try:
      comm.reset(env_id_unity)
      comm.expand_scene(starting_graph)
      success, unity_graph = comm.environment_graph()
    except:
      print('resetting executable\n\n\n')
      os.system('pkill exec_mac_07_01')
      os.system('/Users/maitreygramopadhye/Documents/Vision\ and\ LLM\ planners/virtualhome/simulation/exec_mac.app/Contents/MacOS/exec_mac_07_01 &')
      print('resetting done\n\n\n')
      time.sleep(5)
      comm.reset(env_id_unity)
      comm.expand_scene(starting_graph)
      success, unity_graph = comm.environment_graph()
    
    if not success:
        raise Exception('virtualhome scene expand error')

    print(env_id_unity)
    import ipdb; ipdb.set_trace()
    unity_objects = unity_graph['nodes']
    unity_object_ids = {}
    for i, node in enumerate(unity_objects):
        unity_object_ids[node['id']] = i

    for i, node in enumerate(starting_graph['nodes']):
        if node['id'] in unity_object_ids:
            starting_graph['nodes'][i]['bounding_box'] = unity_objects[unity_object_ids[node['id']]]['bounding_box']

    init_graph_file.write(json.dumps(starting_graph, indent=4))
    init_graph_file.close()


# Make robot action plans
print("Make robot action plans")
for init_path in tqdm(task_paths):
    if init_path in incorrect_samples:
        print(init_path)
        continue
    path = str(init_path).replace('executable_programs', 'withoutconds')
    path = path.split('/')
    env_id = path[4][16]
    path = '/'.join(path[:4] + path[5:])
    
    data = open(str(path)).read().split('\n')
    task = data[0]
    robot_ap = '\n'.join(data[4:-1])
    
    task_dir = '../datasets/action_plans_robot/' + task + '/'
    ap_dir = task_dir + env_id + '/'
    if os.path.exists(ap_dir):
        robot_ap_file = open(ap_dir + str(len(os.listdir(ap_dir)) + 1) + '.txt', 'w')
    else:
        os.makedirs(ap_dir)
        robot_ap_file = open(ap_dir + '1.txt', 'w')
    
    robot_ap_file.write(robot_ap)
    robot_ap_file.close()


# Make NL action plans
print("Make NL action plans")
robot_action0 = ["[SLEEP]", "[STANDUP]", "[WAKEUP]"]
robot_action1 = ["[CLOSE]", "[CUT]", "[DRINK]", "[DROP]", "[EAT]", "[FIND]", "[GRAB]", "[GREET]", "[LIE]", "[LOOKAT]", 
                "[MOVE]", "[OPEN]", "[PLUGIN]", "[PLUGOUT]", "[POINTAT]", "[PULL]", "[PUSH]", "[PUTOBJBACK]", 
                "[PUTOFF]", "[PUTON]", "[READ]", "[RINSE]", "[RUN]", "[SCRUB]", "[SIT]", "[SQUEEZE]", 
                "[SWITCHOFF]", "[SWITCHON]", "[TOUCH]", "[TURNTO]", "[TYPE]", "[WALK]", "[WASH]", "[WATCH]", "[WIPE]", "[RELEASE]"]
robot_action2 = ["[POUR]", "[PUTBACK]", "[PUTIN]"]

NL_action0 = ["Sleep", "Stand up", "Wake up"]
NL_action1 = ["Close", "Cut", "Drink", "Drop", "Eat", "Find", "Grab", "Greet", "Lie on", "Look at", 
              "Move", "Open", "Plug in", "Plug out", "Point at", "Pull", "Push", "Put back", 
              "Take off", "Put on", "Read", "Rinse", "Run to", "Scrub", "Sit on", "Squeeze", 
              "Switch off", "Switch on", "Touch", "Turn to", "Type on", "Walk to", "Wash", "Watch", "Wipe", "Release"]
NL_action2 = [["Pour", "into"], ["Put", "on"], ["Put", "in"]]

# temp_action_set = set(copy.deepcopy(robot_action2))

robot_ap_paths = list(Path('../datasets/action_plans_robot').rglob("*.txt"))
for path in tqdm(robot_ap_paths):
    robot_ap = open(path).read().split('\n')
    NL_ap = []
    for robot_step in robot_ap:
        robot_action = robot_step.split(' ')[0].upper()
        if robot_action in robot_action0:
            NL_action = NL_action0[robot_action0.index(robot_action)]
            NL_step = NL_action + " - ()"
        
        elif robot_action in robot_action1:
            NL_action = NL_action1[robot_action1.index(robot_action)]

            robot_obj = robot_step.split('<')[1].split('>')[0]
            robot_obj_ID = robot_step.split(' ')[-1]

            NL_obj = robot_obj.replace('_', ' ')
            NL_step = NL_action + " " + NL_obj + " - " + robot_obj + " " + robot_obj_ID

        elif robot_action in robot_action2:
            NL_action = NL_action2[robot_action2.index(robot_action)]

            robot_obj1 = robot_step.split('<')[1].split('>')[0]
            robot_obj_ID1 = robot_step.split(' ')[2][1:-1]

            robot_obj2 = robot_step.split('<')[-1].split('>')[0]
            robot_obj_ID2 = robot_step.split(' ')[-1][1:-1]

            NL_obj1 = robot_obj1.replace('_', ' ')
            NL_obj2 = robot_obj2.replace('_', ' ')
            NL_step = NL_action[0] + " " + NL_obj1 + " " + NL_action[1] + " " + NL_obj2 + " - " + robot_obj1 + ", " + robot_obj2 + " (" + robot_obj_ID1 + ", " + robot_obj_ID2 + ")"

        else:
            print(robot_action, path)
    
        # if robot_action in temp_action_set:
        #   print('\n', robot_step, NL_step)
        #   temp_action_set.remove(robot_action)
    
        NL_ap.append(NL_step)

    NL_ap = '\n'.join(NL_ap)
    NL_path = str(path).split('/')
    assert len(NL_path) == 6
    task = NL_path[3]
    env_id = NL_path[4]
    ap_num = NL_path[5]

    ap_dir = '../datasets/action_plans_NL/' + task + '/' + env_id + '/'
    ap_path = ap_dir + ap_num
    os.makedirs(ap_dir, exist_ok=True)
    NL_ap_file = open(ap_path, 'w')
  
    NL_ap_file.write(NL_ap)
    NL_ap_file.close()


NL_action0 = ["Sleep", "Stand up", "Wake up"]
NL_action1 = ["Close ", "Cut ", "Drink ", "Drop ", "Eat ", "Find ", "Grab ", "Greet ", "Lie on ", "Look at ", 
            "Move ", "Open ", "Plug in ", "Plug out ", "Point at ", "Pull ", "Push ", "Put back ", 
            "Take off ", "Put on ", "Read ", "Rinse ", "Run to ", "Scrub ", "Sit on ", "Squeeze ", 
            "Switch off ", "Switch on ", "Touch ", "Turn to ", "Type on ", "Walk to ", "Wash ", "Watch ", "Wipe ", "Release "]
NL_action2 = [["Pour ", " into "], ["Put ", " on "], ["Put ", " in "]]

def findObjs(NL_sample):
    if NL_sample in NL_action0:
        return []
  
    if any([NL_sample.startswith(NL_action) for NL_action in NL_action1]):
        NL_action = [NL_action for NL_action in NL_action1 if NL_sample.startswith(NL_action)]
        if len(NL_action) > 1:
            raise Exception('multiple actions 1')
            return None
        else:
            sample_objects_NL = NL_sample.replace(NL_action[0], '')
            return [sample_objects_NL]

    if any([(NL_sample.startswith(NL_action[0]) and NL_action[1] in NL_sample) for NL_action in NL_action2]):
        NL_action = [NL_action for NL_action in NL_action2 if (NL_sample.startswith(NL_action[0]) and NL_action[1] in NL_sample)]
        if len(NL_action) > 1:
            raise Exception('multiple actions 2')
            return None
        else:
            sample_objects_NL = NL_sample.split(NL_action[0][0])[-1]
            sample_objects_NL = sample_objects_NL.split(NL_action[0][1])
            return sample_objects_NL

    print("NL_sample for no object match", NL_sample)
    raise Exception('no object match')


NL_ap_paths = list(Path('../datasets/action_plans_NL').rglob("*.txt"))
NL_objects = set()
for path in tqdm(NL_ap_paths):
    NL_ap = open(path).read().split('\n')
    for step in NL_ap:
        step = step.split(' - ')[0]
        obj_names = [obj for obj in findObjs(step)]
        [NL_objects.add(obj) for obj in obj_names]

available_steps = []
for step in NL_action0:
    available_steps.append(step)

for step in NL_action1:
    for obj in NL_objects:
        available_steps.append(step + obj)

for step in NL_action2:
    for obj1 in NL_objects:
        for obj2 in NL_objects:
            available_steps.append(step[0] + obj1 + step[1] + obj2)

# print(available_steps)
available_actions_file = open('../datasets/available_actions.json', 'w')
available_actions_file.write(json.dumps(available_steps, indent=4))
available_actions_file.close()

# # Make init objects
# print("Make init objects")
# graph_paths = list(Path('../datasets/init_graphs/').rglob("*.json"))

# for path in tqdm(graph_paths):
#     init_graph = json.load(open(path))
#     path = str(path).split('/')
#     assert len(path) == 6
#     task = path[3]
#     env_id = path[4]
#     env_id_unity = int(env_id) - 1
#     graph_num = path[5]
    
#     try:
#       comm.reset(env_id_unity)
#       comm.expand_scene(init_graph)
#       success, graph = comm.environment_graph()
#     except:
#       print('resetting executable\n\n\n')
#       os.system('pkill exec_mac_07_01')
#       os.system('/Users/maitreygramopadhye/Documents/Vision\ and\ LLM\ planners/virtualhome/simulation/exec_mac.app/Contents/MacOS/exec_mac_07_01 &')
#       print('resetting done\n\n\n')
#       time.sleep(5)
#       comm.reset(env_id_unity)
#       comm.expand_scene(init_graph)
#       success, graph = comm.environment_graph()
    
#     if not success:
#         raise Exception('virtualhome scene expand error')

#     # import ipdb; ipdb.set_trace()
#     objects = graph['nodes']
    
#     obj_dir = '../datasets/objects/' + task + '/' + env_id + '/'
#     obj_path = obj_dir + graph_num

#     os.makedirs(obj_dir, exist_ok=True)
#     obj_file = open(obj_path, 'w')
#     obj_file.write(json.dumps(objects, indent=4))
#     obj_file.close()