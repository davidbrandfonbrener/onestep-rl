import numpy as np
import torch
import utils


class Transition():
    def __init__(self, state, action, reward, state_prime, action_prime=None, 
                    done=False, batched=False):
        self.s = state
        self.a = action
        self.r = reward
        self.sp = state_prime
        self.ap = action_prime
        self.d = done
        self.batched = batched

    def to_torch(self):
        s = utils.torch_single_precision(self.s)
        a = utils.torch_single_precision(self.a)
        r = utils.torch_single_precision(self.r)
        sp = utils.torch_single_precision(self.sp)
        if self.ap is not None:
            ap = utils.torch_single_precision(self.ap)
        else:
            ap = None
        d = utils.torch_single_precision(self.d)
        return Transition(s, a, r, sp, ap, d, batched=self.batched)

    def to_device(self, device):
        t = self.to_torch()
        t.s = t.s.to(device)
        t.a = t.a.to(device)
        t.r = t.r.to(device)
        t.sp = t.sp.to(device)
        if t.ap is not None:
            t.ap = t.ap.to(device)
        t.d = t.d.to(device)
        return t


class Replay:
    def __init__(self, state_shape, action_shape, has_next_action=False,
                 max_size=int(1e6)):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.s = np.zeros((max_size, *state_shape))
        self.a = np.zeros((max_size, *action_shape))
        self.r = np.zeros((max_size, 1))
        self.sp = np.zeros((max_size, *state_shape))
        self.d = np.zeros((max_size, 1))

        self.has_next_action = has_next_action
        if has_next_action:
            self.ap = np.zeros((max_size, *action_shape))
        self.max_size = max_size
        self.next_slot = 0
        self.length = 0

    def append(self, trans):
        index = self.next_slot
        if index < self.length:
            self._invalidate(index)
        self.s[index] = trans.s
        self.a[index] = trans.a
        self.r[index] = trans.r
        self.sp[index] = trans.sp
        self.d[index] = trans.d
        if self.has_next_action:
            self.ap[index] = trans.ap

        self.next_slot = (self.next_slot + 1) % self.max_size
        self.length = max(self.length, self.next_slot)
        return index

    def sample(self, n=128):
        indices = np.random.randint(0, self.length, size=(n,))
        return self.get_transitions(indices)

    # TODO: handle episodes correctly (only sample at least k from end)
    def sample_k(self, k, gamma, n=128):
        indices = np.random.randint(0, self.length, size=(n,))
        transitions = self.get_transitions(indices)
        for i in range(1,k):
            i_transitions = self.get_transitions(indices + i)
            transitions.r += np.power(gamma, i) * i_transitions.r
        if k > 1:
            k_transitions = self.get_transitions(indices + k)
            transitions.sp = k_transitions.s
            if self.has_next_action:
                transitions.ap = k_transitions.a
        return transitions

    def get_transitions(self, indices):
        transitions = Transition(self.s[indices],
                                 self.a[indices],
                                 self.r[indices],
                                 self.sp[indices],
                                 done=self.d[indices],
                                 batched=True)
        if self.has_next_action:
            transitions.ap = self.ap[indices]
        return transitions

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        transition = Transition(self.s[i],
                                self.a[i],
                                self.r[i],
                                self.sp[i],
                                done=self.d[i],
                                batched=False)
        if self.has_next_action:
            transition.ap = self.ap[i]
        return transition

    def subset(self, indices):
        new_replay = Replay(self.state_shape, self.action_shape, 
                            self.has_next_action, self.max_size)
        length = len(indices)
        new_replay.s[:length] = self.s[indices]
        new_replay.a[:length] = self.a[indices]
        new_replay.r[:length] = self.r[indices]
        new_replay.sp[:length] = self.sp[indices]
        new_replay.d[:length] = self.d[indices]
        if self.has_next_action:
            new_replay.ap[:length] = self.ap[indices]
        
        new_replay.length = length
        new_replay.next_slot = length
        return new_replay