import networkx as nx
import numpy as np
from observation_parser import parse_inventory, parse_observation


class StateAction(object):

    def __init__(self, spm, vocab, vocab_rev, vocab_kge, max_word_len):
        self.graph_state = nx.DiGraph()
        self.max_word_len = max_word_len
        self.graph_state_rep = []
        self.visible_state = ""
        self.drqa_input = ""
        self.vis_pruned_actions = []
        self.pruned_actions_rep = []
        self.sp = spm
        self.vocab_act = vocab
        self.vocab_act_rev = vocab_rev
        self.vocab_kge = vocab_kge
        self.adj_matrix = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        self.room = ""

    def visualize(self):
        # import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph_state)
        edge_labels = {e: self.graph_state.edges[e]['rel'] for e in self.graph_state.edges}
        print(edge_labels)
        nx.draw_networkx_edge_labels(self.graph_state, pos, edge_labels)
        nx.draw(self.graph_state, pos=pos, with_labels=True, node_size=200, font_size=10)
        nx.drawing.nx_pydot.write_dot(self.graph_state, 'kg.dot')
        #plt.show()

    def load_vocab_kge(self, tsv_file, use_electric=True):
        ent = {}
        idx_lut = {}
        idx = 0
        with open(tsv_file, 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                if not use_electric and "terminal" in e:
                    continue
                ent[e.strip()] = int(eid.strip())
                idx_lut[int(eid.strip())] = idx
                idx += 1
        return {'entity': ent, 'relation': ent}, idx_lut

    def update_state(self, visible_state, inventory_state, objs, referent_to_type_lut, prev_action=None, cache=None):

        rules = parse_observation(visible_state)
        inventory_rules = parse_inventory(inventory_state)

        add_rules = rules + inventory_rules

        for rule in add_rules:
            u = rule[0]
            v = rule[2]

            if u in referent_to_type_lut and v in referent_to_type_lut:
                
                self.graph_state.add_edge(self.vocab_kge[referent_to_type_lut[u]], self.vocab_kge[referent_to_type_lut[v]], rel=rule[1])


        return add_rules

    def get_state_rep_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))

        for u, v in self.graph_state.edges:

            if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
                break

            u_idx = self.vocab_kge[u]
            v_idx = self.vocab_kge[v]
            self.adj_matrix[u_idx][v_idx] = 1

            ret.append(self.vocab_kge[u])
            ret.append(self.vocab_kge[v])

        return list(set(ret))

    def get_state_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))

        for u, v in self.graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
                break

            u_idx = self.vocab_kge[u]
            v_idx = self.vocab_kge[v]
            self.adj_matrix[u_idx][v_idx] = 1

            ret.append(u)
            ret.append(v)

        return list(set(ret))

    def get_obs_rep(self, *args):
        ret = [self.get_visible_state_rep_drqa(ob) for ob in args]
        return pad_sequences(ret, maxlen=300)

    def get_visible_state_rep_drqa(self, state_description):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']

        for rm in remove:
            state_description = state_description.replace(rm, '')

        return self.sp.encode_as_ids(state_description)

    def get_action_rep_drqa(self, action):

        action_desc_num = 20 * [0]
        action = str(action)

        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_len]            
            action_desc_num[i] = self.vocab_act_rev[short_tok] if short_tok in self.vocab_act_rev else 0

        return action_desc_num

    def step(self, visible_state, inventory_state, objs, referent_to_type_lut, prev_action=None, cache=None, gat=True):
        ret = self.update_state(visible_state, inventory_state, objs, referent_to_type_lut, prev_action, cache)

        # self.pruned_actions_rep = [self.get_action_rep_drqa(a) for a in self.vis_pruned_actions]

        inter = self.visible_state #+ "The actions are:" + ",".join(self.vis_pruned_actions) + "."
        self.drqa_input = self.get_visible_state_rep_drqa(inter)

        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix

        return ret


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


