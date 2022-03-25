import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import DecoderRNN, DecoderRNN2, EncoderLSTM, GraphAttentionLayer, PackedEncoderRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        return x


class ObjectDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeddings, graph_dropout, k):
        super(ObjectDecoder, self).__init__()
        self.k = k
        self.decoder = DecoderRNN2(hidden_size, output_size, embeddings, graph_dropout)
        self.max_decode_steps = 2
        self.softmax = nn.Softmax()

    def forward(self, input, input_hidden, vocab, vocab_rev, decode_steps_t, graphs, type_to_obj_str_luts):
        all_outputs, all_words = [], []

        decoder_input = torch.tensor([vocab_rev['<s>']] * input.size(0)).cuda()
        decoder_hidden = input_hidden.unsqueeze(0)
        torch.set_printoptions(profile="full")

        masks = []
        for lut in type_to_obj_str_luts:
            mask = self.get_valid_object_mask(lut)
            masks.append(mask)

        for di in range(self.max_decode_steps):
            ret_decoder_output, decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, input, graphs)

            cur_objs = []

            for i in range(graphs.size(0)):
                all_output = decoder_output[i]
                valid = all_output[masks[i]]
                cur_obj_idx = torch.argmax(valid)
                cur_obj = masks[i][cur_obj_idx]
                cur_objs.append(cur_obj)

            decoder_input = torch.LongTensor(cur_objs).cuda()
            all_words.append(decoder_input)
            all_outputs.append(ret_decoder_output)

        return torch.stack(all_outputs), torch.stack(all_words)

    def flatten_parameters(self):
        self.encoder.gru.flatten_parameters()
        self.decoder.gru.flatten_parameters()

    def get_valid_object_mask(self, lut):
        mask = []
        for key in lut:
            if key != 'desc':
                mask.append(int(key))
        return mask


class KGA2C(nn.Module):
    def __init__(self, params, templates, max_word_length, vocab_kge_len, vocab_act,
                 vocab_act_rev, input_vocab_size, gat=True):
        super(KGA2C, self).__init__()
        self.templates = templates
        self.gat = gat
        self.max_word_length = max_word_length
        self.vocab = vocab_act
        self.vocab_rev = vocab_act_rev
        self.batch_size = params['batch_size']
        self.action_emb = nn.Embedding(len(vocab_act), params['embedding_size'])
        self.state_emb = nn.Embedding(input_vocab_size, params['embedding_size'])
        self.action_drqa = ActionDrQA(input_vocab_size, params['embedding_size'],
                                      params['batch_size'], params['recurrent'])
        self.state_gat = StateNetwork(params['gat_emb_size'],
                                      params['embedding_size'],
                                      params['dropout_ratio'], vocab_kge_len)
        self.template_enc = EncoderLSTM(input_vocab_size, params['embedding_size'],
                                        int(params['hidden_size'] / 2),
                                        params['padding_idx'], params['dropout_ratio'],
                                        self.action_emb)
        if not self.gat:
            self.state_fc = nn.Linear(110, 100)
        else:
            self.state_fc = nn.Linear(210, 100)
        # N policy heads, one for each separate task

        self.decoder_template = DecoderRNN(params['hidden_size'], len(templates))
        self.decoder_object = ObjectDecoder(50, 100, len(self.vocab), self.action_emb, params['graph_dropout'], params['k_object'])
        self.softmax = nn.Softmax(dim=1)
        self.critic = nn.Linear(100, 1)

    def get_action_rep(self, action):
        action = str(action)
        decode_step = action.count('OBJ')
        action = action.replace('OBJ', '')
        action_desc_num = 20 * [0]

        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_length]
            action_desc_num[i] = self.vocab_rev[short_tok] if short_tok in self.vocab_rev else 0

        return action_desc_num, decode_step

    def forward_next(self, obs, graph_rep):
        o_t, h_t = self.action_drqa.forward(obs)
        g_t = self.state_gat.forward(graph_rep)
        state_emb = torch.cat((g_t, o_t), dim=1)
        state_emb = F.relu(self.state_fc(state_emb))
        value = self.critic(state_emb)
        return value

    def forward(self, obs, scores, graph_rep, graphs, type_to_obj_str_luts):
        '''
        :param obs: The encoded ids for the textual observations (shape 4x300):
        The 4 components of an observation are: look - ob_l, inventory - ob_i, response - ob_r, and prev_action.
        :type obs: ndarray

        '''
        batch = self.batch_size
        o_t, h_t = self.action_drqa.forward(obs)

        src_t = []

        for scr in scores:
            #fist bit encodes +/-
            if scr >= 0:
                cur_st = [0]
            else:
                cur_st = [1]
            cur_st.extend([int(c) for c in '{0:09b}'.format(abs(int(scr)))])
            src_t.append(cur_st)

        src_t = torch.FloatTensor(src_t).cuda()

        if not self.gat:
            state_emb = torch.cat((o_t, src_t), dim=1)
        else:
            g_t = self.state_gat.forward(graph_rep)
            state_emb = torch.cat((g_t, o_t, src_t), dim=1)
        state_emb = F.relu(self.state_fc(state_emb))
        det_state_emb = state_emb.clone()#.detach()
        value = self.critic(det_state_emb)

        decoder_t_output, decoder_t_hidden = self.decoder_template(state_emb, h_t)

        templ_enc_input = []
        decode_steps = []

        topi = self.softmax(decoder_t_output).multinomial(num_samples=1)

        for i in range(batch):
            templ, decode_step = self.get_action_rep(self.templates[topi[i].squeeze().detach().item()])
            templ_enc_input.append(templ)
            decode_steps.append(decode_step)

        _, decoder_o_hidden_init0, _ = self.template_enc.forward(torch.tensor(templ_enc_input).cuda().clone())

        decoder_o_output, decoded_o_words = self.decoder_object.forward(decoder_o_hidden_init0.cuda(), decoder_t_hidden.squeeze(0).cuda(), self.vocab, self.vocab_rev, decode_steps, graphs, type_to_obj_str_luts)

        return decoder_t_output, decoder_o_output, decoded_o_words, topi, value, decode_steps#decoder_t_output#template_mask


    def clone_hidden(self):
        self.action_drqa.clone_hidden()

    def restore_hidden(self):
        self.action_drqa.restore_hidden()

    def reset_hidden(self, done_mask_tt):
        self.action_drqa.reset_hidden(done_mask_tt)


class StateNetwork(nn.Module):
    def __init__(self, gat_emb_size, embedding_size, dropout_ratio, max_types):
        super(StateNetwork, self).__init__()

        self.embedding_size = embedding_size
        self.dropout_ratio = dropout_ratio
        self.gat_emb_size = gat_emb_size
        #self.params = params
        self.gat = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)

        self.state_ent_emb = nn.Embedding.from_pretrained(torch.zeros((max_types, self.embedding_size)), freeze=False)
        self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 100)

    # def load_vocab_kge(self):
    #     ent = {}
    #     with open(tsv_file, 'r') as f:
    #         for line in f:
    #             e, eid = line.split('\t')
    #             ent[int(eid.strip())] = e.strip()
    #     return ent

    def forward(self, graph_rep):
        out = []
        for g in graph_rep:
            _, adj = g
            adj = torch.IntTensor(adj).cuda()
            x = self.gat.forward(self.state_ent_emb.weight, adj).view(-1)
            out.append(x.unsqueeze(0))
        out = torch.cat(out)
        ret = self.fc1(out)
        return ret


class ActionDrQA(nn.Module):
    def __init__(self, vocab_size, embedding_size, batch_size, recurrent=True):
        super(ActionDrQA, self).__init__()
        #self.opt = opt
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.recurrent = recurrent

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

        self.enc_look = PackedEncoderRNN(self.vocab_size, 100)
        self.h_look = self.enc_look.initHidden(self.batch_size)
        self.enc_inv = PackedEncoderRNN(self.vocab_size, 100)
        self.h_inv = self.enc_inv.initHidden(self.batch_size)
        self.enc_ob = PackedEncoderRNN(self.vocab_size, 100)
        self.h_ob = self.enc_ob.initHidden(self.batch_size)
        self.enc_preva = PackedEncoderRNN(self.vocab_size, 100)
        self.h_preva = self.enc_preva.initHidden(self.batch_size)
        self.enc_task = PackedEncoderRNN(self.vocab_size, 100)
        self.h_task = self.enc_task.initHidden(self.batch_size)

        self.fcx = nn.Linear(100 * 5, 100)
        self.fch = nn.Linear(100 * 5, 100)

    def reset_hidden(self, done_mask_tt):
        '''
        Reset the hidden state of episodes that are done.

        :param done_mask_tt: Mask indicating which parts of hidden state should be reset.
        :type done_mask_tt: Tensor of shape [BatchSize x 1]

        '''
        self.h_look = done_mask_tt.detach() * self.h_look
        self.h_inv = done_mask_tt.detach() * self.h_inv
        self.h_ob = done_mask_tt.detach() * self.h_ob
        self.h_preva = done_mask_tt.detach() * self.h_preva
        self.h_task = done_mask_tt.detach() * self.h_task

    def clone_hidden(self):
        ''' Makes a clone of hidden state. '''
        self.tmp_look = self.h_look.clone().detach()
        self.tmp_inv = self.h_inv.clone().detach()
        self.h_ob = self.h_ob.clone().detach()
        self.h_preva = self.h_preva.clone().detach()
        self.h_task = self.h_task.clone().detach()


    def restore_hidden(self):
        ''' Restores hidden state from clone made by clone_hidden. '''
        self.h_look = self.tmp_look
        self.h_inv = self.tmp_inv
        self.h_ob = self.h_ob
        self.h_preva = self.h_preva
        self.h_task = self.h_task

    def forward(self, obs):
        '''
        :param obs: Encoded observation tokens.
        :type obs: np.ndarray of shape (Batch_Size x 4 x 300)

        '''
        x_l, h_l = self.enc_look(torch.LongTensor(obs[:,0,:]).cuda(), self.h_look)
        x_i, h_i = self.enc_inv(torch.LongTensor(obs[:,1,:]).cuda(), self.h_inv)
        x_o, h_o = self.enc_ob(torch.LongTensor(obs[:,2,:]).cuda(), self.h_ob)
        x_p, h_p = self.enc_preva(torch.LongTensor(obs[:,3,:]).cuda(), self.h_preva)
        x_t, h_t = self.enc_task(torch.LongTensor(obs[:,4,:]).cuda(), self.h_task)

        if self.recurrent:
            self.h_look = h_l
            self.h_ob = h_o
            self.h_preva = h_p
            self.h_inv = h_i
            self.h_task = h_t

        x = F.relu(self.fcx(torch.cat((x_t, x_l, x_i, x_o, x_p), dim=1)))
        h = F.relu(self.fch(torch.cat((h_t, h_l, h_i, h_o, h_p), dim=2)))

        return x, h
