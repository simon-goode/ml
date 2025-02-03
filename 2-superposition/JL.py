import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# list of vectors in some dimension, with many more vectors than dimensions
num_vectors = 1000
vector_len = 100
big_matrix = torch.randn(num_vectors, vector_len)
big_matrix /= big_matrix.norm(p=2, dim=1, keepdim=True) #normalize
big_matrix.requires_grad_(True)

# angle distribution
dot_products = big_matrix @ big_matrix.T
norms = torch.sqrt(torch.diag(dot_products))
normed_dot_products = dot_products / torch.outer(norms, norms)
angles_degrees = torch.rad2deg(torch.acos(normed_dot_products.detach()))
# use this to ignore self-orthogonality
self_orthogonality_mask = ~(torch.eye(num_vectors, num_vectors).bool())
plt.hist(angles_degrees[self_orthogonality_mask].numpy().ravel(), bins=1000, range=(0, 180))
plt.grid(1)
plt.show()

# set up an optimization loop to create near-perpendicular vectors
optimizer = torch.optim.Adam([big_matrix], lr=0.01)
num_steps = 250

losses=[]

dot_diff_cutoff = 0.001 #epsilon
big_id = torch.eye(num_vectors, num_vectors)

for step_num in tqdm(range(num_steps)):
    optimizer.zero_grad()

    dot_products = big_matrix @ big_matrix.T
    # punish deviation from orthogonal
    diff = dot_products - big_id
    loss = (diff.abs() - dot_diff_cutoff).relu().sum()

    # extra incentive to keep rows normalized
    loss += num_vectors * diff.diag().pow(2).sum()

    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# loss curve
plt.plot(losses)
plt.grid(1)
plt.show()

# angle distribution
dot_products = big_matrix @ big_matrix.T
norms = torch.sqrt(torch.diag(dot_products))
normed_dot_products = dot_products / torch.outer(norms, norms)
angles_degrees = torch.rad2deg(torch.acos(normed_dot_products.detach()))
# use this to ignore self-orthogonality
self_orthogonality_mask = ~(torch.eye(num_vectors, num_vectors).bool())
plt.hist(angles_degrees[self_orthogonality_mask].numpy().ravel(), bins=1000, range=(88, 92))
plt.grid(1)
plt.show()