# Module imports
from training_utils import *


transitions = 60
pars_to_loop = {"buffer_size" : [20*transitions, 50*transitions, 150*transitions],
                "batch_size" : [2*transitions, 5*transitions, 10*transitions],
                "lr" : [10**(-7), 10**(-6), 10**(-5)],
                "n_iters": [10,100,1000]
                }

stat_wiggles = []
slow_increase = []
past_peak = []
peaky_bad = []
peaky_good = []

# Get all ... options
print("All following options had stationary wiggle")
option_counter = 0
n_options = len(pars_to_loop["buffer_size"])*len(pars_to_loop["batch_size"])*len(pars_to_loop["lr"])*len(pars_to_loop["n_iters"])
for buff_size in pars_to_loop["buffer_size"]:
    for batch_size in pars_to_loop["batch_size"]:
        for lr in pars_to_loop["lr"]:
            for n_iters in pars_to_loop["n_iters"]:
                option_counter += 1
                if option_counter in stat_wiggles:
                    print(f"Option: {option_counter}, buff size: {buff_size}, batch size: {batch_size}, lr: {lr}, n_iters {n_iters}")

