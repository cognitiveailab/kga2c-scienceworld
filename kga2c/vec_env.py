import random

import time
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env, id, task_num):
    parent_remote.close()
    curr_var_no = 0
    env.create(id, task_num, curr_var_no)
    try:
        done = False
        while True:
            cmd, data, var_nums = remote.recv()
            if cmd == 'step':
                if done:                   
                    # Samples random task variation from list of vars in current pool
                    curr_var_no = random.sample(var_nums, 1)[0]
                    ob, info, graph_info = env.resetWithVariation(curr_var_no)
                    rew = 0
                    done = False
                    
                else:
                    ob, rew, done, info, graph_info = env.step(data)
                remote.send((ob, rew, done, info, graph_info))
            elif cmd == 'reset':
                env.close()
                curr_var_no = random.sample(var_nums, 1)[0]
                env.create(id, task_num, curr_var_no)

                ob, info, graph_info = env.reset()
                remote.send((ob, info, graph_info))
            elif cmd == 'close':                
                env.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs, env, task_num, threadIdOffset=0):
        self.closed = False
        self.total_steps = 0
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ids = [i+threadIdOffset for i in range(self.num_envs)]
        self.ps = [Process(target=worker, args=(work_remote, remote, env, ids, task_num))
                   for (work_remote, remote, ids) in zip(self.work_remotes, self.remotes, self.ids)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
            time.sleep(1)
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, var_nums):
        self.total_steps += 1
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action, var_nums))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return zip(*results)

    def reset(self, var_nums):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None, var_nums))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None, None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
