import collections
import numpy as np
from representations import StateAction
import random

from scienceworld import ScienceWorldEnv

import time


GraphInfo = collections.namedtuple('GraphInfo', 
                                    'objs, ob_rep, act_rep, graph_state, graph_state_rep, admissible_actions, admissible_actions_rep, referents_to_type_id_lut,type_to_obj_str_lut')

def load_vocab(env):
    vocab_kge = env.getObjectTypes()
    vocab_kge = {v: k for k, v in vocab_kge.items()}

    actions = env.getPossibleActions()
    vocab_act = vocab_kge.copy()
    vocab_act_rev = {v: k for k, v in vocab_act.items()}
    
    idx = len(vocab_kge)    

    for action in actions:
        for word in action.split():
            if word not in vocab_act and word != 'OBJ':
                vocab_act[idx] = word
                vocab_act_rev[word] = idx
                idx += 1

    vocab_act[idx] = ' '
    vocab_act[idx+1] = '<s>'
    vocab_act_rev[' '] = idx
    vocab_act_rev['<s>'] = idx + 1    

    return vocab_act, vocab_act_rev, vocab_kge

def clean_obs(s):
    garbage_chars = ['*', '-', '!', '[', ']']
    for c in garbage_chars:
        s = s.replace(c, ' ')
    return s.strip()


def extract_templates(template_id_list):
    # Temperary work around
    NUM_OPTION = 0
    OPTION_BASE = 100
    sorted_templates = sorted(template_id_list, key=lambda x : x['template_id'])
    ret = [cdict["action_example"] for cdict in sorted_templates]
    extra_actions = [str(i) for i in range(NUM_OPTION)]
    ret = ret + extra_actions
    template_lut = {sorted_templates[i]["template_id"] : i for i in range(len(sorted_templates))}
    for i in range(NUM_OPTION):
        template_lut[OPTION_BASE + i] = len(template_id_list) + i
    return ret, template_lut

class KGA2CEnv:
    '''

    KGA2C environment performs additional graph-based processing.

    '''
    def __init__(self, rom_path, seed, spm_model, max_word_len, step_limit=None, stuck_steps=10, gat=True, simplification_str=""):
        random.seed(seed)
        np.random.seed(seed)
        self.rom_path        = rom_path
        self.seed            = seed
        self.episode_steps   = 0
        self.stuck_steps     = 0
        self.valid_steps     = 0
        self.spm_model       = spm_model
        self.step_limit      = step_limit
        self.max_stuck_steps = stuck_steps
        self.gat             = gat
        self.env             = None
        self.state_rep       = None
        self.taskName        = None
        self.max_word_len = max_word_len
        self.simplification_str = simplification_str
        
        
    def create(self, thread_id, task_num, var_no):
        ''' Create the ScienceWorld environment '''

        print("Creating environment (threadNum = " + str(thread_id) + ")")
        self.env = ScienceWorldEnv("", self.rom_path, envStepLimit=self.step_limit, threadNum=100+thread_id)
        time.sleep(2)

        taskNames = self.env.getTaskNames()  # Just get first task
        task_num %= len(taskNames)
        self.taskName = taskNames[task_num]
        self.env.load(self.taskName, var_no, self.simplification_str)

        self.vocab_act, self.vocab_act_rev, self.vocab_kge = load_vocab(self.env)

        _, _ = self.env.reset()
        self.templates, self.template_lut = extract_templates(self.env.getPossibleActionsWithIDs())

        self.all_act_combos = self.env.getPossibleActionObjectCombinations()

        self.all_act_combos_lut = {}

        for act in self.all_act_combos[0]:
            self.all_act_combos_lut[act['action']] = {'template_id': act['template_id'], 'obj_ids': act['obj_ids']}
        self.obj_lut = self.all_act_combos[1]



    def _get_admissible_actions(self):
        ''' Queries ScienceWorld API for a list of admissible actions from the current state. '''
        self.all_act_combos = self.env.getPossibleActionObjectCombinations()

        self.all_act_combos_lut = {}

        for act in self.all_act_combos[0]:
            self.all_act_combos_lut[act['action']] = {'template_id': act['template_id'],
                                                      'obj_ids': act['obj_ids']}

        self.obj_lut = self.all_act_combos[1]

        admissible = []
        types = set()
        possible_acts = self.env.getValidActionObjectCombinationsWithTemplates()
        type_lut = self.env.getAllObjectIdsTypesReferentsLUTJSON()
        for act in possible_acts:
            curr_act = {'action': act['action'], 'template_id': act['template_id']}
            curr_type_ids = []
            for obj_id in act['obj_ids']:
                if str(obj_id) in type_lut:
                    type_id = type_lut[str(obj_id)]["type_id"]
                    type_str = self.vocab_act[type_id]
                    curr_type_ids.append(type_id)
                    types.add(type_str)
            if curr_type_ids == []:
                curr_type_ids = [self.vocab_act_rev[' ']]
            curr_act['type_ids'] = curr_type_ids
            admissible.append(curr_act)

        return admissible, types


    def _build_graph_rep(self, action, ob_r):
        ''' Returns various graph-based representations of the current state. '''
        admissible_actions, objs = self._get_admissible_actions()

        admissible_actions_rep = [self.state_rep.get_action_rep_drqa(a['action']) \
                                  for a in admissible_actions] \
                                      if admissible_actions else [[0] * 20]
        try: # Gather additional information about the new state
            ob_l = self.env.look()
            ob_i = self.env.inventory()
            ob_t = self.env.taskdescription()
        except RuntimeError:
            print('RuntimeError: {}'.format(clean_obs(ob_r)))
            ob_l = ob_i = ''
        ob_rep = self.state_rep.get_obs_rep(ob_l, ob_i, ob_r, action, ob_t)
        cleaned_obs = clean_obs(ob_l)

        referents_to_type_id_lut = {}
        obj_id_to_type_id_lut = self.env.getAllObjectIdsTypesReferentsLUTJSON()
        for key, value in obj_id_to_type_id_lut.items():
            for referent in value["referents"]:
                referents_to_type_id_lut[referent] = value["type_id"]
        referents_to_type_id_lut[' '] = len(self.vocab_act)
        referents_to_type_id_lut['<s>'] = len(self.vocab_act) + 1
        type_to_obj_str_lut = self.env.getPossibleObjectReferentTypesLUT()

        rules = self.state_rep.step(cleaned_obs, ob_i, objs, referents_to_type_id_lut, action, cache=None, gat=self.gat)
        graph_state = self.state_rep.graph_state
        graph_state_rep = self.state_rep.graph_state_rep
        action_rep = self.state_rep.get_action_rep_drqa(action)

        

        return GraphInfo(objs, ob_rep, action_rep, graph_state, graph_state_rep,\
                         admissible_actions, admissible_actions_rep,\
                         referents_to_type_id_lut, type_to_obj_str_lut)


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_look = info['look']
        obs_inventory = info['inv']
        info['valid'] = done 
        info['steps'] = info['moves']
        if info['valid']:
            self.valid_steps += 1
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1
        info['history'] = self.env.getRunHistory()
        if info['score'] == -100:
            done = True

        if self.stuck_steps > self.max_stuck_steps:
            done = True

        graph_info = self._build_graph_rep(action, obs)
        if done:
            self.state_rep.visualize()
        return [obs, obs_look, obs_inventory], reward, done, info, graph_info


    def reset(self):
        self.state_rep = StateAction(self.spm_model, self.vocab_act, self.vocab_act_rev,
                                     self.vocab_kge, self.max_word_len)
        self.stuck_steps = 0
        self.valid_steps = 0
        self.episode_steps = 0        
        obs, info = self.env.reset()
        obs_look = info['look']
        obs_inventory = info['inv']
        info['valid'] = False
        info['steps'] = 0
        graph_info = self._build_graph_rep('look around', obs)
        info['history'] = self.env.getRunHistory()
        return [obs, obs_look, obs_inventory], info, graph_info

    def resetWithVariation(self, var_no):
        assert(self.taskName is not None) 
        self.env.load(self.taskName, var_no, self.simplification_str)
        return self.reset()

    def getRandomVariationTrain(self):
        return self.env.getRandomVariationTrain()    

    def getRandomVariationDev(self):
        return self.env.getRandomVariationDev()    

    def getRandomVariationTest(self):
        return self.env.getRandomVariationTest()

    def storeRunHistory(self, episodeIdxKey, notes):
        return self.env.storeRunHistory(episodeIdxKey, notes)

    def saveRunHistoriesBufferIfFull(self, filenameOutPrefix, maxPerFile=1000, forceSave=False):
        return self.env.saveRunHistoriesBufferIfFull(filenameOutPrefix, maxPerFile=maxPerFile, forceSave=forceSave)

    def close(self):
        print("Closing environment (Port " + str(self.env.portNum) + ")")
        self.env.shutdown()
