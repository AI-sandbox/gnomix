import numpy as np
import time

from src.Gnofix.phasing import *

# Value function calculation
def seq_left_end(S, idx): 
  off_idx = np.where(S[:idx] != S[idx])[0]
  if len(off_idx) == 0:
    return -1
  return max(off_idx)

def seq_right_end(S, idx): 
  off_idx = np.where(S[idx:] != S[idx])[0]
  if len(off_idx) == 0:
    return len(S)
  return idx + min(off_idx)

def continuity_value(M,P,idx1,idx2,gen_pos=None):
  return int(M[idx1]==M[idx2]) + int(P[idx1]==P[idx2])

def exp_len_value(M,P,idx1,idx2,gen_pos=None):

  if gen_pos is not None:
    ml = gen_pos[idx1-1] - gen_pos[seq_left_end(M, idx1)-1]
    mr = gen_pos[seq_right_end(M, idx2)-1] - gen_pos[idx2-1]
    pl = gen_pos[idx1-1] - gen_pos[seq_left_end(P, idx1)-1]
    pr = gen_pos[seq_right_end(P, idx2)-1] - gen_pos[idx2-1]
  else:
    ml = idx1 - seq_left_end(M, idx1)
    mr = seq_right_end(M, idx2) - idx2 
    pl = idx1 - seq_left_end(P, idx1)
    pr = seq_right_end(P, idx2) - idx2

  if M[idx1]==M[idx2]:
    ml=ml+mr+1
    mr=0
  if P[idx1]==P[idx2]:
    pl=pl+pr+1
    pr=0

  # values larger sequences over small ones
  return sum(np.exp(0.01*np.array([ml, mr, pl, pr])))

def blurr_slack(S, slack):
  # overwrites sequences shorter than the slack, if located in a larger sequence

  if slack == 0:
    return S
  
  # for all the SNPs within slack away from the ends
  for i in range(slack, len(S)-2*slack+1):
    left  = S[i-slack:i]
    right = S[i+slack:i+2*slack]
    # if left and right would form a sequence if center would comply
    if np.all(np.concatenate([left,right]) == left[0]):
      # owerwrite center
      S[i:i+slack] = left

  return S

def is_steeling(M,P,i,w,one_checked=False):
  # if mismatch in M
  if M[i-1] != M[i]:
    # and P is solid at that point
    if P[i-1] == P[i] and np.all(P[i-w:i+w] == P[i-1]):
      # and M is trying to switch the same ancestry
      if np.all(M[i-w:i] == P[i]) or np.all(M[i:i+w] == P[i]):
        # then it's steeling
        return True
  if not one_checked:
    # check if P is steeling from M
    return is_steeling(P,M,i,w,one_checked=True)
  else:
    return False

def simple_switch(M_in, P_in, slack=1, animate=True, cont=False, gen_pos=None, verbose=True):
  """
  A simple switch algorithm. When encountering a change in sequence, compare the value 
  of the switch to the value of the current state, switch if it's more. The default value
  function sum(exp(length(adjoint sequences))) where length is measured in the input arrays.
  """

  start_time = time.time()

  M, P = np.copy(M_in), np.copy(P_in)
  M_track, P_track = np.zeros_like(M), np.ones_like(P)
  value_function = exp_len_value if not cont else continuity_value
  if animate:
    history = np.array([M,P])

  for w in range(slack+1):

    M, P = blurr_slack(M,w), blurr_slack(P,w) # if slack w, then sequences of length w don't make any sense
    if animate:
      history = np.dstack([history, [M,P]])
    for i in range(1,len(M)-w):
      if M[i] != M[i-1] or P[i] != P[i-1]:
        val = value_function(M,P,i-1,i,gen_pos)
        M_temp = np.concatenate([M[:i], [P[i+w]]*w, P[i+w:]])
        P_temp = np.concatenate([P[:i], [M[i+w]]*w, M[i+w:]])
        switch_val = value_function(M_temp,P_temp,i-1,i,gen_pos)
        if switch_val > val and not is_steeling(M,P,i,w):
          # print(i)
          M, P = np.copy(M_temp), np.copy(P_temp)
          M_track, P_track = track_switch(M_track, P_track, i)
          if animate:
            history = np.dstack([history, [M,P]])

  ani = None
  if animate:
    # make it stop on the end for a while
    for _ in range(20):
      history = np.dstack([history, [M,P]])
    ani = animate_history(history)

  if verbose:
    print("Solving time:", time.time()-start_time, "seconds")

  return M,P,M_track,P_track,ani