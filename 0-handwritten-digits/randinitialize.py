import numpy as np

def initialize(a, b):
    epsilon = 0.15
    # randomizes initial values between [-epsilon, +epsilon]
    c = np.random.rand(a, b + 1) * (2 * epsilon) - epsilon
    return c