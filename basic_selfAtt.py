{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMDI/B830blBTJerO1K1KRD",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/edgoni/NN-QS-MBSpinSystems/blob/main/basic_selfAtt.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 75,
      "metadata": {
        "id": "-mC0p_6K2t9H"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import netket as nk\n",
        "\n",
        "import jax\n",
        "import jax.numpy as jnp\n",
        "\n",
        "import flax\n",
        "import flax.linen as nn"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class MultiHead_Att(nn.Module):\n",
        "  ### Parametros\n",
        "  layers: int #capas\n",
        "  heads: int # cabezas de attention\n",
        "  dk: int #dimension matrices entrenables\n",
        "\n",
        "  ###call\n",
        "  @nn.compact\n",
        "  def __call__(self, x):\n",
        "    dimension = x.shape[-1]\n",
        "    dimension_head = self.heads * self.dk\n",
        "    w_shape = (dimension, dimension, self.heads)\n",
        "\n",
        "    #definimos matrices de entrenamiento\n",
        "    for i in range(self.layers):\n",
        "\n",
        "      #-----------------------------------------------------------------------------------------------#\n",
        "\n",
        "      weight_Q = self.param(f'weight_Q_head{i}', nn.initializers.xavier_uniform(), w_shape)\n",
        "      weight_K = self.param(f'weight_K_head{i}', nn.initializers.xavier_uniform(), w_shape)\n",
        "      weight_V = self.param(f'weight_V_head{i}', nn.initializers.xavier_uniform(), w_shape)\n",
        "\n",
        "      # Bloque Self_Attention\n",
        "      Q = jnp.einsum('bd,ddh->bhd', x, weight_Q)\n",
        "      K = jnp.einsum('bd,ddh->bhd', x, weight_K)\n",
        "      V = jnp.einsum('bd,ddh->bhd', x, weight_V)\n",
        "\n",
        "\n",
        "      ##DENTRO DE CADA CABEZA##\n",
        "      logits = jnp.einsum('bhd,bhd->bh', Q, K) / jnp.sqrt(dimension)\n",
        "      weights = jax.nn.softmax(logits, axis=-1)\n",
        "\n",
        "\n",
        "      att_output = jnp.einsum('bh,bhd->bhd', weights, V)\n",
        "\n",
        "      att_output = jnp.mean(att_output, axis=1)\n",
        "\n",
        "\n",
        "      #-----------------------------------------------------------------------------------------------#\n",
        "      #ESTAMOS EN UN BUCLE IMPORTANTE PONER NOMBRES\n",
        "      ##Residuo2##\n",
        "      x = x + att_output\n",
        "      x = nn.LayerNorm(name=f'ln1_{i}')(x)\n",
        "\n",
        "      ##MLP##\n",
        "      mlp = nn.Dense(dimension * 4, name=f'mlp_up_{i}')(x)\n",
        "      mlp = nn.gelu(mlp)\n",
        "      mlp = nn.Dense(dimension, name=f'mlp_down_{i}')(mlp)\n",
        "\n",
        "      ##Residuo2##\n",
        "      x = x + mlp\n",
        "      x = nn.LayerNorm(name=f'ln2_{i}')(x)\n",
        "\n",
        "\n",
        "    output = nn.Dense(1)(x) # log amplitud final\n",
        "    output = output.squeeze(-1) # Squeeze the last dimension\n",
        "\n",
        "    return output"
      ],
      "metadata": {
        "id": "-Q8Uw69iw234"
      },
      "execution_count": 76,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "vvUml2FY1nc6"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}