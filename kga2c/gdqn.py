import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import time
import pickle
import sentencepiece as spm
from statistics import mean
import random

from representations import StateAction
from models import KGA2C
from env import *
from vec_env import *

from scienceworld import BufferedHistorySaver

device = torch.device("cuda")





class KGA2CTrainer(object):
    '''

    KGA2C main class.


    '''
    def __init__(self, params):

        self.params = params
        self.output_dir = params['output_dir']


        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(params['spm_file'])

        self.init_envs(params['task_num'])

        self.model = KGA2C(params, self.templates, self.max_word_length, len(self.vocab_kge),
                           self.vocab_act, self.vocab_act_rev, len(self.sp), gat=self.params['gat']).cuda()
        self.batch_size = params['batch_size']
        if params['preload_weights']:
            self.model = torch.load(self.params['preload_weights'])['model']
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])

        self.loss_fn1 = nn.BCELoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss()
        self.loss_fn3 = nn.MSELoss()
        self.train_score_record = []
        self.dev_score_record = []
        

    def init_envs(self, curr_task_no):
        params = self.params

        kg_env = KGA2CEnv(params['rom_file_path'], params['seed'], self.sp,
                          max_word_len=50, step_limit=params['reset_steps'], stuck_steps=params['stuck_steps'], gat=params['gat'],
                          simplification_str=params["simplification_str"])
        kg_env.create(99, curr_task_no, 0)
        task_names = kg_env.env.getTaskNames()
        self.num_tasks = len(task_names)
        self.templates = kg_env.templates
        self.template_lut = kg_env.template_lut
        self.train_var_nos = list(kg_env.env.getVariationsTrain())
        self.dev_var_nos = list(kg_env.env.getVariationsDev())
        self.test_var_nos = list(kg_env.env.getVariationsTest())
        self.kg_env = kg_env
        if self.params['output_dir'].endswith('/'):
            self.params['output_dir'] = self.params['output_dir'][:-1]
        self.bufferedHistorySaverTrain = BufferedHistorySaver(filenameOutPrefix = f"{self.params['output_dir']}/kga2c-saveout" + "-seed" + str(params['seed']) + "-task" + str(curr_task_no) + "-train")
        self.bufferedHistorySaverEval = BufferedHistorySaver(filenameOutPrefix = f"{self.params['output_dir']}/kga2c-saveout" + "-seed" + str(params['seed']) + "-task" + str(curr_task_no) + "-eval")
        kg_env.close()


        self.vocab_act = kg_env.vocab_act
        self.vocab_act_rev = kg_env.vocab_act_rev
        self.vocab_kge = kg_env.vocab_kge
        self.max_word_length = kg_env.max_word_len

        self.train_vec_env = VecEnv(params['batch_size'], kg_env, params['task_num'], self.params['output_dir'], threadIdOffset=0)
        self.dev_vec_env = None

    def generate_targets(self, admissible, objs):
        '''
        Generates ground-truth targets for admissible actions.

        :param admissible: List-of-lists of admissible actions. Batch_size x Admissible
        :param objs: List-of-lists of interactive objects. Batch_size x Objs
        :returns: template targets and object target tensors

        '''
        tmpl_target = []
        type_targets = []
        for adm in admissible:
            type_t = set()
            cur_t = [0] * len(self.templates)
            for a in adm:
                if a['template_id'] in self.template_lut:
                    template_idx = self.template_lut[a['template_id']]
                    cur_t[template_idx] = 1
                type_t.update(a['type_ids'])
            tmpl_target.append(cur_t)
            type_targets.append(list(type_t))
        tmpl_target_tt = torch.FloatTensor(tmpl_target).cuda()

        
        idxs = np.array([i for i in range(len(self.vocab_act))])
        # Note: Adjusted to use the objects in the admissible actions only
        type_mask_target = []
        for typel in type_targets: # in objs
            cur_typet = [0] * (len(self.vocab_act))
            for t in typel:
                cur_typet[t] = 1
            sel_idx = idxs[np.random.choice(len(self.vocab_act), int(0.2 * len(self.vocab_act)), replace=False)]
            for t in sel_idx:
                cur_typet[t] = 1
            type_mask_target.append([[cur_typet], [cur_typet]])
        type_target_tt = torch.FloatTensor(type_mask_target).squeeze(2).cuda()
        return tmpl_target_tt, type_target_tt


    def generate_graph_mask(self, graph_infos):
        assert len(graph_infos) == self.batch_size
        mask_all = []
        for graph_info in graph_infos:
            mask = [0] * (len(self.vocab_act))
            
            if self.params['masking'] == 'kg':
                # Uses the knowledge graph as the mask.
                graph_state = graph_info.graph_state
                ents = set()
                for u, v in graph_state.edges:
                    ents.add(u)
                    ents.add(v)
                for ent in ents:
                    for ent_word in ent.split():
                        if ent_word[:self.max_word_length] in self.vocab_act_rev:
                            idx = self.vocab_act_rev[ent_word[:self.max_word_length]]
                            mask[idx] = 1

            elif self.params['masking'] == 'interactive':
                # Uses interactive objects ground truth as the mask.
                for o in graph_info.objs:
                    if o in self.vocab_act_rev.keys() and o != '':
                        mask[self.vocab_act_rev[o]] = 1
            elif self.params['masking'] == 'none':
                # No mask at all.
                mask = [1] * (len(self.vocab_act))
            else:
                assert False, 'Unrecognized masking {}'.format(self.params['masking'])
            if self.params['mask_dropout'] != 0.0:
                indices = [i for i in range(len(mask))]
                selected = random.sample(indices, int(len(mask) * self.params['mask_dropout']))
                for s in selected:
                    mask[s] = 1
            mask_all.append(mask)
        return torch.BoolTensor(mask_all).cuda().detach()


    def discount_reward(self, transitions, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(transitions))):
            _, _, values, rewards, done_masks, _, _, _, _, _, _ = transitions[t]
            R = rewards + self.params['gamma'] * R * done_masks
            adv = R - values
            returns.append(R)
            advantages.append(adv)
        return returns[::-1], advantages[::-1]

    def train(self, max_steps):
        start = time.time()
        transitions = []
        score_count = {}
        observations, infos, graph_infos = self.train_vec_env.reset(self.train_var_nos)
        episode = 0
        for step in range(1, max_steps + 1):
            print("Step: " + str(step))
            # tb.logkv('Step', step)
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]
            type_to_obj_str_luts = [g.type_to_obj_str_lut for g in graph_infos]
            scores = [info['score'] for info in infos]
            tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
                obs_reps, scores, graph_state_reps, graph_mask_tt,type_to_obj_str_luts)
            
            # Log the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [self.templates[t] for t in topk_tmpl_idxs.tolist()]
            tmpl_pred_str = ', '.join(['{} {:.3f}'.format(tmpl, prob) for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())])
            
            # Generate the ground truth and object mask
            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)
            

            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt, type_to_obj_str_luts)

            observations, rewards, dones, infos, graph_infos = self.train_vec_env.step(chosen_actions, self.train_var_nos)
            for n in range(len(graph_infos)):
                
                print('Environment {}, Step {}, Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}'.format(
                    n, step, chosen_actions[n], rewards[n], infos[n]['score'], dones[n], value[n].item()))
                if dones[n]:
                    score = infos[n]['score']
                    if score in score_count:
                        score_count[score] += 1
                    else:
                        score_count[score] = 1
                    self.train_score_record.append(score if score != -100 else 0)
                    if episode < 9:
                        last_10_score = mean(self.train_score_record)
                    else:
                        last_10_score = mean(self.train_score_record[int(len(self.train_score_record)*0.9):])
                    run_history = infos[n]['history']
                    self.bufferedHistorySaverTrain.storeRunHistory(run_history, episode, notes={'last_10_score': last_10_score, 'step': step})
                    self.bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=self.params['max_histories_per_file'])
                    episode += 1
                    
                    print(score_count)
                    print(f"Last 10% episodes average score: {last_10_score}")

                    
            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            
            transitions.append((tmpl_pred_tt, obj_pred_tt, value, rew_tt,
                                done_mask_tt, tmpl_gt_tt, dec_tmpl_tt,
                                dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps))

            if len(transitions) >= self.params['bptt']:
                print('StepsPerSecond', float(step) / (time.time() - start))
                self.model.clone_hidden()

                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                scores = [info['score'] for info in infos]
                returns, advantages = self.discount_reward(transitions[:-1], value)

                loss = self.update(transitions[:-1], returns, advantages)
                print(f"Loss: {loss.item()}")
                del transitions[:]
                self.model.restore_hidden()

            if step % self.params['checkpoint_interval'] == 0:
                parameters = { 'model': self.model }
                torch.save(parameters, os.path.join(self.params['output_dir'], 'kga2c.pt'))
            if step % self.params['test_interval'] == 0:
                with torch.no_grad():
                    self.test(global_steps=step,max_episodes=self.params['max_eval_episodes'], mode="test")


                observations, infos, graph_infos = self.train_vec_env.reset(self.train_var_nos)
                
        with open(os.path.join(self.output_dir, 'train_scores.pkl'), 'wb') as f:
            pickle.dump(self.train_score_record, f)
        with open(os.path.join(self.output_dir, 'dev_scores.pkl'), 'wb') as f:
            pickle.dump(self.dev_score_record, f)
        
        self.bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=self.params['max_histories_per_file'], forceSave=True)
        self.bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=self.params['max_histories_per_file'], forceSave=True)
        self.train_vec_env.close_extras()
        print(f"Total time: {time.time() - start}")

    def test(self, global_steps=0, max_episodes=10, mode="dev"):
        transitions = []
        score_count = {}
        print(f"##### {mode.upper()} BEGINS #####")
        if mode == 'dev':
            var_num = self.dev_var_nos
        else:
            var_num = self.test_var_nos
        self.dev_vec_env = VecEnv(self.params['batch_size'], self.kg_env, self.params['task_num'],self.params['output_dir'], threadIdOffset=100, is_train=False)
        obs, infos, graph_infos = self.dev_vec_env.reset(var_num)
        episode = 0
        while True:
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]
            type_to_obj_str_luts = [g.type_to_obj_str_lut for g in graph_infos]
            scores = [info['score'] for info in infos]
            tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
                obs_reps, scores, graph_state_reps, graph_mask_tt,type_to_obj_str_luts)
            # tb.logkv_mean('Value', value.mean().item())

            # Log the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [self.templates[t] for t in topk_tmpl_idxs.tolist()]
            tmpl_pred_str = ', '.join(['{} {:.3f}'.format(tmpl, prob) for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())])

            # Generate the ground truth and object mask
            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)

            # topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0,0]).topk(5)
            # topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
            # o1_pred_str = ', '.join(['{} {:.3f}'.format(o, prob) for o, prob in zip(topk_o1, topk_o1_probs.tolist())])

            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt, type_to_obj_str_luts)

            obs, rewards, dones, infos, graph_infos = self.dev_vec_env.step(chosen_actions, var_num)
            for n in range(len(graph_infos)):
                
                print('Environment {}, Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}'.format(
                    n, chosen_actions[n], rewards[n], infos[n]['score'], dones[n], value[n].item()))
                if dones[n]:
                    score = infos[n]['score']

                    if score in score_count:
                        score_count[score] += 1
                    else:
                        score_count[score] = 1
                    self.dev_score_record.append(score if score != -100 else 0)
                    if episode < 9:
                        last_10_score = mean(self.dev_score_record)
                    else:
                        last_10_score = mean(self.dev_score_record[int(len(self.dev_score_record)*0.9):])
                    run_history = infos[n]['history']
                    self.bufferedHistorySaverEval.storeRunHistory(run_history, f"{global_steps}-{episode}", notes={'last_10_score':last_10_score})
                    self.bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=self.params['max_histories_per_file'])
                    
                    print(score_count)
                    print(f"Last 10% episodes average score: {last_10_score}")

                    episode += 1
                    if episode >= max_episodes:
                        print(f"##### {mode.upper()} ENDS #####")
                        self.dev_vec_env.close_extras()
                        return                       
                    

            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append((tmpl_pred_tt, obj_pred_tt, value, rew_tt,
                                done_mask_tt, tmpl_gt_tt, dec_tmpl_tt,
                                dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps))

            if len(transitions) >= self.params['bptt']:
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                type_to_obj_str_luts = [g.type_to_obj_str_lut for g in graph_infos]
                scores = [info['score'] for info in infos]
                _, _, _, _, next_value, _ = self.model(obs_reps, scores, graph_state_reps, graph_mask_tt, type_to_obj_str_luts)
                returns, advantages = self.discount_reward(transitions, next_value)

                del transitions[:]
                self.model.restore_hidden()

        print(f"##### {mode.upper()} ENDS #####")
        self.dev_vec_env.close_extras()

    def update(self, transitions, returns, advantages):
        assert len(transitions) == len(returns) == len(advantages)
        loss = 0
        for trans, ret, adv in zip(transitions, returns, advantages):
            tmpl_pred_tt, obj_pred_tt, value, _, _, tmpl_gt_tt, dec_tmpl_tt, \
                dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps = trans

            # Supervised Template Loss
            tmpl_probs = F.softmax(tmpl_pred_tt, dim=1)
            template_loss = self.params['template_coeff'] * self.loss_fn1(tmpl_probs, tmpl_gt_tt)

            # Supervised Object Loss
            object_mask_target = obj_mask_gt_tt.permute((1, 0, 2))
            obj_probs = F.softmax(obj_pred_tt, dim=2)
            object_mask_loss = self.params['object_coeff'] * self.loss_fn1(obj_probs, object_mask_target)

            # Build the object mask
            o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
            for d, st in enumerate(dec_steps):
                if st > 1:
                    o1_mask[d] = 1
                    o2_mask[d] = 1
                elif st == 1:
                    o1_mask[d] = 1
            o1_mask = torch.FloatTensor(o1_mask).cuda()
            o2_mask = torch.FloatTensor(o2_mask).cuda()

            # Policy Gradient Loss
            policy_obj_loss = torch.FloatTensor([0]).cuda()
            cnt = 0
            for i in range(self.batch_size):
                if dec_steps[i] >= 1:
                    cnt = cnt + 1
                    batch_pred = obj_pred_tt[0, i, graph_mask_tt[i]]
                    action_log_probs_obj = F.log_softmax(batch_pred, dim=0)
                    dec_obj_idx = dec_obj_tt[0,i].item()
                    graph_mask_list = graph_mask_tt[i].nonzero().squeeze().cpu().numpy().flatten().tolist()
                    idx = graph_mask_list.index(dec_obj_idx)
                    log_prob_obj = action_log_probs_obj[idx]
                    policy_obj_loss = policy_obj_loss - log_prob_obj * adv[i].detach()
            if cnt > 0:
                policy_obj_loss = policy_obj_loss / cnt
            log_probs_obj = F.log_softmax(obj_pred_tt, dim=2)

            log_probs_tmpl = F.log_softmax(tmpl_pred_tt, dim=1)
            action_log_probs_tmpl = log_probs_tmpl.gather(1, dec_tmpl_tt).squeeze()

            policy_tmpl_loss = (-action_log_probs_tmpl * adv.detach().squeeze()).mean()

            policy_loss = policy_tmpl_loss + policy_obj_loss

            value_loss = self.params['value_coeff'] * self.loss_fn3(value, ret)
            tmpl_entropy = -(tmpl_probs * log_probs_tmpl).mean()
            object_entropy = -(obj_probs * log_probs_obj).mean()
            # Minimizing entropy loss will lead to increased entropy
            entropy_loss = self.params['entropy_coeff'] * -(tmpl_entropy + object_entropy)

            loss = loss + template_loss + object_mask_loss + value_loss + entropy_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()

        # Compute the gradient norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm = grad_norm + p.grad.data.norm(2).item()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])

        # Clipped Grad norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm = grad_norm + p.grad.data.norm(2).item()

        self.optimizer.step()
        
        return loss


    def decode_actions(self, decoded_templates, decoded_objects, type_to_referents_luts):
        '''
        Returns string representations of the given template actions.

        :param decoded_template: Tensor of template indices.
        :type decoded_template: Torch tensor of size (Batch_size x 1).
        :param decoded_objects: Tensor of o1, o2 object indices.
        :type decoded_objects: Torch tensor of size (2 x Batch_size x 1).

        '''
        decoded_actions = []
        for i in range(self.batch_size):
            decoded_template = decoded_templates[i].item()
            decoded_object1 = decoded_objects[0][i].item()
            decoded_object2 = decoded_objects[1][i].item()
            decoded_action = self.tmpl_to_str(decoded_template, decoded_object1, decoded_object2, type_to_referents_luts[i])
            decoded_actions.append(decoded_action)
        return decoded_actions


    def tmpl_to_str(self, template_idx, type_id1, type_id2, type_to_referents_lut):
        """ Returns a string representation of a template action. """
        template_str = self.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        type_id1 = str(type_id1)
        type_id2 = str(type_id2)
        if holes <= 0:
            return template_str
        elif holes == 1:
            word = self.type_to_word(type_id1, type_to_referents_lut)
            return template_str.replace('OBJ', word)
        else:
            object1 = self.type_to_word(type_id1, type_to_referents_lut)
            object2 = self.type_to_word(type_id2, type_to_referents_lut)
            return template_str.replace('OBJ', object1, 1)\
                               .replace('OBJ', object2, 1)

    def type_to_word(self, type_id, type_to_referents_lut):
        if type_id in type_to_referents_lut:
            referents = list(type_to_referents_lut[type_id].keys())
            referents.remove('desc')
            obj_id = random.choice(referents)
            return type_to_referents_lut[type_id][obj_id]
        # The model may want to use an obj that does not exist.
        else:
            return ''