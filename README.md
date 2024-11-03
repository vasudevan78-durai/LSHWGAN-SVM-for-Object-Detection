# LSHWGAN-SVM-for-Object-Detection

# Algorithm I Dynamic data selection strategy of the LSHWGAN system.	

import numpy as np

def dynamic_data_selection_strategy(ENs, Ts, xi):
    """
    Dynamic data selection strategy for LSHWGAN system.

    Parameters:
    ENs: list of edge nodes (edge pixels).
    Ts: list of training times of ENs in the current migration period.
    xi: offset for data migration matching.

    Returns:
    DMs: List of data migration matching results.
    """
    # Step 1: Calculate average execution time T̅
    T_avg = np.mean(Ts)

    # Initialize lists for migration
    L_Out = []  # Outgoing migration list
    L_In = []   # Incoming migration list
    DMs = []    # Data migration matches

    # Step 2-5: Calculate migration amounts and build L_Out and L_In
    for j, T_j in enumerate(Ts):
        t_j = Ts[j]
        delta_n_j = int((T_avg - T_j) / t_j)

        if delta_n_j < 0 and abs(delta_n_j) > xi:
            L_Out.append((ENs[j], abs(delta_n_j)))  # Exceeds xi, needs outgoing migration
        elif delta_n_j > 0 and abs(delta_n_j) > xi:
            L_In.append((ENs[j], delta_n_j))        # Exceeds xi, needs incoming migration

    # Step 6-11: Execute migrations based on L_Out and L_In
    while L_Out and L_In:
        # Step 7-10: Match each EN_out in L_Out with EN_in in L_In
        for out_idx, (EN_out, n_out) in enumerate(L_Out[:]):
            for in_idx, (EN_in, n_in) in enumerate(L_In[:]):
                if abs(n_out + n_in) <= xi:
                    DMs.append((EN_out, EN_in, min(n_in, n_out)))
                    L_Out.pop(out_idx)
                    L_In.pop(in_idx)
                    break  # Break to re-evaluate lists

        # Step 12-18: Find max(n_out) from L_Out and max(n_in) from L_In and continue migration
        if L_Out and L_In:
            max_out_idx = np.argmax([n for _, n in L_Out])
            max_in_idx = np.argmax([n for _, n in L_In])

            EN_out, n_out = L_Out[max_out_idx]
            EN_in, n_in = L_In[max_in_idx]

            if abs(n_in) > abs(n_out):
                DMs.append((EN_out, EN_in, n_out))
                L_Out.pop(max_out_idx)
                L_In[max_in_idx] = (EN_in, n_in - n_out)
            else:
                DMs.append((EN_out, EN_in, n_in))
                L_In.pop(max_in_idx)
                L_Out[max_out_idx] = (EN_out, n_out - n_in)

    return DMs


# Algorithm II Hierarchical Walk operation in LSGAN

import numpy as np
import torch
import torch.nn.functional as F

def hierarchical_walk_operation(X, F):
    """
    Hierarchical Walk operation in LSGAN.

    Parameters:
    X: numpy array or torch tensor, the input matrix of training video frames.
    F: numpy array or torch tensor, filter parameter matrix of the current LSGAN sub-model.

    Returns:
    A: numpy array or torch tensor, feature map with probabilities of the current LSGAN sub-model.
    """
    # Ensure inputs are torch tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(F, torch.Tensor):
        F = torch.tensor(F, dtype=torch.float32)

    # Step 1: Compute the shape (D_a, H_a, W_a) of feature map A based on X and F
    D_a, H_a, W_a = X.shape[0], X.shape[1] - F.shape[0] + 1, X.shape[2] - F.shape[1] + 1
    A = torch.empty((D_a, H_a, W_a))

    # Step 2: Compute the number of parallel tasks P_LSGAN = D_a × H_a × W_a
    P_LSGAN = D_a * H_a * W_a

    # Step 3: For each parallel task in P_LSGAN
    for d in range(D_a):
        for h in range(H_a):
            for w in range(W_a):
                # Step 4: Extract the important region area X_i
                r_s, r_e = h, h + F.shape[0]
                c_s, c_e = w, w + F.shape[1]
                X_i = X[d, r_s:r_e, c_s:c_e]

    # Step 5: Execute LSGAN operation (MaxPooling over the convolution)
                a_i = F.max_pool2d(F.conv2d(X_i.unsqueeze(0).unsqueeze(0), F.unsqueeze(0).unsqueeze(0)), kernel_size=2)
                
                # Step 6: Append result to the feature map A
                A[d, h, w] = a_i.squeeze()

    # Step 7: Return the feature map A
    return A

# Example usage
X = np.random.rand(10, 28, 28)    # Example input matrix for video frames with shape (D, H, W)
F = np.random.rand(3, 3)          # Example filter matrix

feature_map = hierarchical_walk_operation(X, F)
print("Feature Map:\n", feature_map)

# Algorithm III Weight parameter updating process and model synchronization in LSHWGAN


import numpy as np

def update_global_weights(W_local, Q):
    """
    Weight parameter updating process and model synchronization in LSHWGAN.

    Parameters:
    W_local: list of lists, where each inner list W_j contains the weights of a local model.
    Q: list of batch sizes for each local model (or edge node).

    Returns:
    W_global: list representing the new version of the global weight set.
    """
    # Step 1: Initialize the global weights W^(i) to zero with the same structure as W_local
    num_weights = len(W_local[0])
    W_global = np.zeros(num_weights)

    # Step 2-4: Update global weight parameters by iterating through each local weight set
    for j, W_j in enumerate(W_local):  # Loop through each local weight set W_j^(i)
        for k, w_jk in enumerate(W_j):  # Loop through each weight parameter w_(j,k)
            W_global[k] += w_jk * Q[j]  # Update global weight with weighted sum

    # Step 5: Return the global weight set W^(i)
    return W_global

# Example usage
# List of local weights from 3 edge nodes (each has a list of weights)
W_local = [
    [0.1, 0.2, 0.3],  # Weights from node 1
    [0.4, 0.5, 0.6],  # Weights from node 2
    [0.7, 0.8, 0.9]   # Weights from node 3
]

# Batch sizes for each node
Q = [10, 20, 30]

# Compute the updated global weights
W_global = update_global_weights(W_local, Q)
print("Updated Global Weights:", W_global)



