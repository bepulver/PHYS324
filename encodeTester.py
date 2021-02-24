import dwave
import dimod
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.embedding.pegasus import find_clique_embedding
import time
import matplotlib.pyplot as plt

token = "put your token here :)"
computer = "Advantage_system1.1"

sampler = DWaveSampler(token=token, solver=computer)

index = 64
while index < 80:
    print("Embedding with " + str(index) + " qubits")
    start = time.process_time()
    hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
    embedding = dwave.embedding.pegasus.find_clique_embedding(index, target_graph=hardware_graph) #Creates a dictionary with the key as the logical Qubit ID and a list of the physical qubit IDs that belong to the logical qubit
    print("It took " + str(time.process_time() - start) + " seconds to embed...")
    #AnnealOffsets.printQubitGraph()
    index += 4