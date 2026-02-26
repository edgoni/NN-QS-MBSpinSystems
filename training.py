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

#-----------------------Declare Observables------------------------------#
renyi = nkx.observable.Renyi2EntanglementEntropy(
    hi, np.arange(0, N / 2 + 1, dtype=int)
)
mags = sum([(-1) ** i * sigmaz(hi, i) / N for i in range(N)])
magnet = sum([sigmaz(hi, i) / N for i in range(N)])

idx = [1, 2, 3, 8, 7, 6]
Wp_op = (sigmax(hi, idx[0]) * sigmay(hi, idx[1]) * sigmaz(hi, idx[2]) * sigmax(hi, idx[3]) * sigmay(hi, idx[4]) * sigmaz(hi, idx[5]))####Plaqueta

#---------------------------------------------------#

####------------/REGLAS d MonteCarlo/--------------####
rule1 = nk.sampler.rules.LocalRule()# flip de un solo spin para proponer nuevo estado
rule2 = nk.sampler.rules.GaussianRule()# flip de todos los spins para proponer nuevo estado
rules = [rule1]
sampler = nk.sampler.MetropolisSampler(
    hi, nk.sampler.rules.MultipleRules([rule1],[1.0])
)
#############################################

epochs = 7

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

#### BUCLE DE ENTRENAMIENTO Y GUARDADO CON TRANSFER LEARNING #################

energies_rbm = np.zeros((len(jz_values), 2))
np.copyto(energies_rbm[:, 0], jz_values)

maxlayers = 3
maxheads = 4

# 1. INICIALIZACIÓN FUERA DEL BUCLE (Transfer Learning)
# Al declarar vstate aquí, conservamos los pesos entre iteraciones de Jz

    
for layers in range(1,maxlayers+1):
    for heads in range(1,maxheads+1):
        
        for i, jz in enumerate(jz_values):
            # Definición de paths incluyendo layers y heads
            path_metrics = f'SelfAtt_metrics{layers}_head{heads}_{jz:.2f}_{lr_name}.csv'
            filename = f"SelfAtt{layers}_head{heads}_{jz:.2f}_{lr_name}.mpack"
            vstate_path = f"vstate_SelfAtt{layers}_head{heads}_{jz:.2f}_{lr_name}.pkl"
            obs_path = f'obs_layers{layers}_head{heads}_{lr_name}.csv'
            path_energies = 'energies_eigenvecs'

            print(f"\n--- Entrenando para Jz = {jz:.2f} ---")

            ##########DECLARAMOS MODELO EN CADA ITERACION############
            RBM = MultiHead_Att(layers=layers, heads=heads, dk=1)
            vstate = nk.vqs.MCState(sampler, model=RBM, n_samples=2048)


            jx = jy = (1 - jz) / 2
            H = KitaevTransverse_H(direcciones, bonds, Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi)

        # El optimizer y el driver se vinculan al vstate que ya tiene los pesos del Jz anterior
            optimizer = nk.optimizer.AdaGrad(learning_rate=lr, epscut=1e-7)
            driver = nk.driver.VMC(H, optimizer, variational_state=vstate)

        # El Keeper evalúa la mejor energía para el Hamiltoniano actual
            keeper = BestIterKeeper(H, N, 1e-8)

            log = nk.logging.RuntimeLog()
            metrics_history = {'step': [], 'energy': [], 'energy_error': [], 'loss': [], 'variance': []}

        # make_extract_metrics debe recibir H para calcular correctamente en cada paso
            callback_fn = [keeper.update, make_extract_metrics(metrics_history, H)]

        # ENTRENAMIENTO
            driver.run(n_iter=epochs, out=log, callback=callback_fn, show_progress=True)

        # ACTUALIZACIÓN ADIABÁTICA (Transfer Learning explícito)
        # Forzamos que el vstate para el PRÓXIMO Jz empiece con los MEJORES pesos de este Jz
            vstate.parameters = keeper.best_state.parameters

        # Guardar pesos optimizados
            with open(filename, "wb") as f:
                f.write(flax.serialization.to_bytes(vstate.parameters))

        # Guardar estado en formato pickle
            with open(vstate_path, "wb") as f:
                pickle.dump(vstate.parameters, f)

        # Cálculo de observables con el mejor estado encontrado
            header = ['Jz', 'Energy', 'S', 'm', 'ms', 'fluct', 'fluct_s', 'Wp','overlap']
            best = keeper.best_state
            wp_val = np.real(best.expect(Wp_op).mean)
            obs = [renyi, magnet, mags, magnet @ magnet, mags @ mags]

            data = np.load(f"{path_energies}.npz")
            psi_exact = data['vecs'][:, 0] # El primer eigenvector
            psi_nqs = vstate.to_array()
            psi_nqs /= np.linalg.norm(psi_nqs)
            overlap =  np.abs(np.vdot(psi_nqs, psi_exact))**2

            results = [jz, keeper.best_energy/N] + [np.real(best.expect(o).mean) for o in obs] + [wp_val] +[overlap]

            file_exists = os.path.isfile(obs_path)
            with open(obs_path, 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(results)

            df_metrics = pd.DataFrame(metrics_history)

            # Guardamos a CSV (usamos index=False para no guardar los números de fila)
            df_metrics.to_csv(path_metrics, index=False)

###############################################################################
