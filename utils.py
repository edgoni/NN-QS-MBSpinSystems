##Utils##

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


def make_extract_metrics(metrics_history, H):
  '''
  Function that extractus some metrics from the training proccess of the NQS.
  Please refer to NetKet documentation to learn more about the structure of this type of function.
  '''
  def extract_metrics(step, log_data, driver):
      stats = driver.state.expect(H)
      energy = float(jnp.real(stats.mean))
      energy_error = float(jnp.real(stats.error_of_mean))

      loss = float(jnp.real(getattr(log_data[driver._loss_name], "mean")))
      variance = float(jnp.real(getattr(log_data[driver._loss_name], "variance")))

      metrics_history['step'].append(step)
      metrics_history['energy'].append(energy)
      metrics_history['energy_error'].append(energy_error)
      metrics_history['loss'].append(loss)
      metrics_history['variance'].append(variance)

      print(f"Step {step}: Energy = {energy:.6f} ± {energy_error:.2e}, Loss = {loss:.4f}, Variance = {variance:.4f}")
      return True

  return extract_metrics

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