
import jax
import jax.numpy as jnp
import jaxlib
from jax.nn.initializers import uniform, normal

import flax
import flax.linen as nn
import optax

import netket as nk
import netket.experimental as nkx
from netket.operator.spin import sigmaz, sigmax, sigmay


import numpy as np
import numpy.typing as npt
from typing import Optional
import pathlib
import copy

class BestIterKeeper:
    """Almacena los valores de varias cantidades de la mejor iteración.

    "Mejor" se define en el sentido de menor energía.

    Argumentos:
        Hamiltoniano: Un array que contiene la matriz del Hamiltoniano.
        N: Número de espines en la cadena.
        baseline: Un límite inferior para la puntuación V. Si la puntuación V
            de la mejor iteración cae por debajo de este umbral, el proceso
            se detendrá antes.
        filename: Puede ser None o un archivo donde se escribirá el mejor estado.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None

    def update(self, step, log_data, driver):
        """Actualiza las cantidades almacenadas si es necesario.

        Esta función está diseñada para actuar como una función de *callback* para NetKet.
        Por favor, consulta la documentación de su API para una explicación detallada.
        """

        vstate = driver.state
        energystep = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        varstep = self.N * var / mean**2

        if self.best_energy > energystep:
            self.best_energy = energystep
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(
                driver.state.parameters
            )
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))

        return self.vscore > self.baseline


def make_extract_metrics(metrics_history, H, N):
  '''
  Function that extractus some metrics from the training proccess of the NQS.
  Please refer to NetKet documentation to learn more about the structure of this type of function.

  Incluye el V-score (N * Var[E_loc] / <E>^2, misma formula que BestIterKeeper),
  calculado con loss/variance de log_data para que coincida exactamente con el
  criterio de parada de BestIterKeeper -- no es una estimacion nueva.
  '''
  def extract_metrics(step, log_data, driver):
      stats = driver.state.expect(H)
      energy = float(jnp.real(stats.mean))
      energy_error = float(jnp.real(stats.error_of_mean))

      loss = float(jnp.real(getattr(log_data[driver._loss_name], "mean")))
      variance = float(jnp.real(getattr(log_data[driver._loss_name], "variance")))
      vscore = N * variance / loss**2 if loss != 0 else float('inf')

      metrics_history['step'].append(step)
      metrics_history['energy'].append(energy)
      metrics_history['energy_error'].append(energy_error)
      metrics_history['loss'].append(loss)
      metrics_history['variance'].append(variance)
      metrics_history['vscore'].append(vscore)

      print(f"Step {step}: Energy = {energy:.6f} ± {energy_error:.2e}, "
            f"Loss = {loss:.4f}, Variance = {variance:.4f}, V-score = {vscore:.2e}")
      return True

  return extract_metrics

def find_plaquettes(graph):
  '''
  Encuentra las plaquetas hexagonales de un KitaevHoneycomb a partir de la
  estructura de bonds coloreados (0=x, 1=y, 2=z), caminando por bonds
  alternando colores x,y,z,x,y,z (y su reverso) desde cada sitio hasta cerrar
  un lazo de 6 sitios distintos.

  Verificado numericamente (vector aleatorio, N=8 y N=18): [H, Wp] = 0 EXACTO
  y Wp^2 = identidad para cada plaqueta encontrada de esta forma, para el
  Kitaev puro (h=0).

  :param graph: grafo de NetKet (p.ej. nk.graph.KitaevHoneycomb) con
      .edges() y .edge_colors
  :return: lista de (loop, order), loop = lista de 6 sitios, order = colores
      de los bonds recorridos entre sitios consecutivos del loop
  '''
  nbr = {0: {}, 1: {}, 2: {}}
  for (a, b), c in zip(graph.edges(), graph.edge_colors):
    nbr[c][a] = b
    nbr[c][b] = a

  found = {}
  for s0 in range(graph.n_nodes):
    for order in [(0, 1, 2, 0, 1, 2), (2, 1, 0, 2, 1, 0)]:
      path = [s0]
      ok = True
      cur = s0
      for c in order:
        if cur not in nbr[c]:
          ok = False
          break
        cur = nbr[c][cur]
        path.append(cur)
      if ok and path[-1] == s0 and len(set(path[:-1])) == 6:
        found[frozenset(path[:-1])] = (path[:-1], order)
  return list(found.values())


def wp_operator(hi, loop, order):
  '''
  Construye el operador de plaqueta Wp para un lazo hexagonal.

  El operador Pauli en cada sitio del lazo es el color OPUESTO al par de
  bonds del hexagono que tocan ese sitio (el bond entrante y el saliente),
  no un patron fijo por posicion -- ese es el error facil de cometer aqui.

  :param hi: espacio de Hilbert de NetKet
  :param loop: lista de 6 sitios (de find_plaquettes)
  :param order: colores de los bonds recorridos (de find_plaquettes)
  '''
  ops = {0: sigmax, 1: sigmay, 2: sigmaz}
  W = None
  for k in range(6):
    c_in = order[(k - 1) % 6]
    c_out = order[k % 6]
    missing = ({0, 1, 2} - {c_in, c_out}).pop()
    term = ops[missing](hi, loop[k])
    W = term if W is None else W @ term
  return W


def all_plaquettes(hi, graph):
  '''
  Devuelve la lista de operadores Wp para TODAS las plaquetas del grafo.
  '''
  return [wp_operator(hi, loop, order) for loop, order in find_plaquettes(graph)]


def KitaevTransverse_H(colores, enlaces,Jx,Jy,Jz,h,hi):
  '''
  Function to define a Kitaev Hamiltonian.
  
  :param colores: Direction of ñthe bonds in the graph
  :param enlaces: Connection in the graph
  :param Jx: coupling X-bond
  :param Jy: coupling Y-bond
  :param Jz: coupling Z-bond
  :param h: External magnetic field
  :param hi: NetKet hilbert space
  '''
  H = nk.operator.LocalOperator(hi, dtype=complex)
  for i, color in enumerate(colores):
    if color == 0:
      bond = enlaces[i]
      H -= Jx * nk.operator.spin.sigmax(hi, bond[0])@nk.operator.spin.sigmax(hi, bond[1])
      H-= h * (nk.operator.spin.sigmax(hi, bond[0]) + nk.operator.spin.sigmay(hi, bond[0]) + nk.operator.spin.sigmaz(hi, bond[0]))
    elif color == 1:
      bond = enlaces[i]
      H -= Jy * nk.operator.spin.sigmay(hi, bond[0])@nk.operator.spin.sigmay(hi, bond[1])
      H-= h * (nk.operator.spin.sigmax(hi, bond[0]) + nk.operator.spin.sigmay(hi, bond[0]) + nk.operator.spin.sigmaz(hi, bond[0]))
    elif color == 2:
      bond = enlaces[i]
      H -= Jz * nk.operator.spin.sigmaz(hi, bond[0])@nk.operator.spin.sigmaz(hi, bond[1])
      H-= h * (nk.operator.spin.sigmax(hi, bond[0]) + nk.operator.spin.sigmay(hi, bond[0]) + nk.operator.spin.sigmaz(hi, bond[0]))
    else:
      print(f'Error, not implemented color {color}')

  return H