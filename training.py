##TRAINING##

##Import##
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd
import optax
import numpy.typing as npt
from typing import Optional
import pathlib
import copy
import flax
import flax.linen as nn
import time
from jax.nn.initializers import uniform, normal
import netket.experimental as nkx
from netket.operator.spin import sigmaz, sigmax, sigmay
import csv
import os
from typing import Any
import pickle


####Importamos de .py
from utils import BestIterKeeper,make_extract_metrics, KitaevTransverse_H
from basic_selfatt import MultiHead_Att


##Declaring KITAEV 
kitaev_graph= nk.graph.KitaevHoneycomb(extent=[3,3], pbc = True)
N = kitaev_graph.n_nodes
adj_list = kitaev_graph.adjacency_list()
direcciones = kitaev_graph.edge_colors
bonds = kitaev_graph.edges()

hi = nk.hilbert.Spin(s=1 / 2, N=kitaev_graph.n_nodes)
kitaev_graph.hi=hi
h = 1

#Declare Observables--------------------------------#
renyi = nkx.observable.Renyi2EntanglementEntropy(
    hi, np.arange(0, N / 2 + 1, dtype=int)
)
mags = sum([(-1) ** i * sigmaz(hi, i) / N for i in range(N)])
magnet = sum([sigmaz(hi, i) / N for i in range(N)])

#---------------------------------------------------#


#Training--------------------------------------------#


####REGLAS d MonteCarlo
rule1 = nk.sampler.rules.LocalRule()# flip de un solo spin para proponer nuevo estado
rule2 = nk.sampler.rules.GaussianRule()# flip de todos los spins para proponer nuevo estado
rules = [rule1]
sampler = nk.sampler.MetropolisSampler(
    hi, nk.sampler.rules.MultipleRules([rule1],[1.0])
)
#############################################

epochs = 1

# definir los inicializadores según el artículo
weights_init = normal(stddev=0.01)
bias_init = normal(stddev=0.1)
vstate_init =  []

jz_values = np.linspace(0, 1, 11)
energies_exact = []
matrics_history = {}


###POSIBLES VALORES LEARNING RATE###
lr = 0.1
ramp_iter = 50
lrmax = 0.05
epsilon = 1e-7

lr_schedule_try = optax.warmup_exponential_decay_schedule(
    init_value=0.01, # bajamos el learning rate para lowlr:1e-3
    peak_value=0.05,
    warmup_steps=30,
    transition_steps=100,
    decay_rate=0.90
)
####################################

lr = lr_schedule_try
lr_name = f'{lr}'
if lr == lr_schedule_try:
    lr_name = 'sched'

####BUCLE DE ENTRENAMIENTO Y GUARDADO#################
energies_rbm = np.zeros((len(jz_values),2))
np.copyto(energies_rbm[:,0], jz_values)
layers = 1
for jz in jz_values:

    path_metrics = f'SelfAtt_metrics{layers}_{jz:.2f}_{lr_name}.csv' 
    filename = f"SelfAtt{layers}_{jz:.2f}_{lr_name}.mpack"
    vstate_path = f"vstate_SelfAtt{layers}_{jz:.2f}_{lr_name}.pkl"
    obs_path = f'obs_layers{layers}_{lr_name}.csv'



    ##----------------------------------------------------------------------------------------------------------##


    print(f"\n--- Entrenando para Jz = {jz:.2f} ---")
    jx = jy = (1 - jz) / 2

    H = KitaevTransverse_H(direcciones, bonds, Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi)

    ##Cambiar por cargar las energías##

    evals = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)
    energies_exact.append(evals[0] / N)


    RBM_name, RBM = f'SelfAtt_{1}', MultiHead_Att(layers = layers, heads = 1, dk=1 )

    vstate = nk.vqs.MCState(sampler, model=RBM, n_samples=2048)
    vstate_init.append(vstate.expect(H))

    optimizer = nk.optimizer.AdaGrad(learning_rate=lr, epscut=1e-7)
    driver = nk.driver.VMC(H, optimizer, variational_state=vstate)

    # Reiniciar el Keeper para este punto
    keeper = BestIterKeeper(H, N, 1e-8)

    log = nk.logging.RuntimeLog()
    metrics_history = {'step': [], 'energy': [], 'energy_error': [], 'loss': [], 'variance': []}
    callback_fn = [keeper.update, make_extract_metrics(metrics_history, H)]

    driver.run(n_iter=epochs, out=log, callback=callback_fn, show_progress=True)

    # guardar energía final de la RBM


    # guardar como CSV usando Pandas
    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv(path_metrics, index=False, sep='\t')

    with open(filename, "wb") as f:
        f.write(flax.serialization.to_bytes(keeper.best_state.parameters))


    print(f"Pesos guardados en: {filename}")

    params = vstate.parameters

    with open(vstate_path, "wb") as f:
        pickle.dump(params, f)

    print("Estado guardado correctamente.")

    header = ['Jz', 'Energy', 'S', 'm', 'ms', 'fluct', 'fluct_s']
    best = keeper.best_state
    # Definimos los observables a medir
    obs = [renyi, magnet, mags, magnet @ magnet, mags @ mags]
    # Calculamos resultados y los metemos en una lista junto a jz y energía
    results = [jz, keeper.best_energy/N] + [np.real(best.expect(o).mean) for o in obs]

    # Escribimos (si el archivo no existe, escribe cabecera + datos; si existe, solo datos)
    file_exists = os.path.isfile(obs_path)
    with open(obs_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(header)
        writer.writerow(results)

################################################