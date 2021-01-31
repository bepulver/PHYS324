#=================================================IMPORTS=========================================================#
import dwave
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from dwave.system.samplers import DWaveSampler
from dwave.embedding.pegasus import find_clique_embedding
from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.util import get_hardware_adjacency

url = "https://cloud.dwavesys.com/sapi"
token = "put your token here :)"
computer = "Advantage_system1.1"
conn = RemoteConnection(url, token)

#===============================================QUBIT MAPPER========================================================#

sampler = DWaveSampler(token=token, solver=computer)
hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
embedding = dwave.embedding.pegasus.find_clique_embedding(4, target_graph=hardware_graph) #Creates a dictionary with the key as the logical Qubit ID and a list of the physical qubit IDs that belong to the logical qubit
print(embedding)

plt.figure(figsize=(40,40))
dnx.draw_pegasus_embedding(hardware_graph, emb=embedding, node_size=100, width=2, unused_color=(0,0,0,.3))
plt.savefig('2V2T_Advantage.png')
plt.close()

