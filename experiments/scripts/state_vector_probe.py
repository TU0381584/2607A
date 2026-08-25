"""Non-invasive state-vector probe: wraps a policy INSTANCE's own
select_action bound method (plain instance-attribute override, standard
Python -- no frozen file touched, no existing script edited) to log the
exact 13-dim request_state array (encode_full_request_state's real
output: [urllc,embb,mmtc] x [prb_used_ratio, congestion_level,
queue_len_norm], then [slice_onehot(3), gnb_onehot(1)]) at every real
admission decision, offline or live, plus the action chosen.

Written because neither existing offline demand lever (arrivals/step,
mean_offered_ratio) changed a known-collapsing checkpoint's OFFLINE
behavior at all, and the raw live omega logs don't expose the actual
state vector (only downstream reward/margin), so there is no way to
even see what's different between the working (3 UE) and collapsing
(6 UE) conditions without capturing it directly at decision time.
"""
import json
import numpy as np


def wrap_policy_for_state_logging(policy, out_path):
    original = policy.select_action
    fh = open(out_path, "w")

    def logging_select_action(req_state, training=False):
        action, info = original(req_state, training=training)
        row = {"state": [float(x) for x in np.asarray(req_state).tolist()],
               "action": int(action)}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        return action, info

    policy.select_action = logging_select_action
    return fh  # caller closes when done
