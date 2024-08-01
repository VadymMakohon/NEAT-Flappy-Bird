from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def check_matplotlib():
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return False
    return True


def check_graphviz():
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return False
    return True


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """Plots the population's average and best fitness."""
    if not check_matplotlib():
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """Plots the trains for a single spiking neuron."""
    if not check_matplotlib():
        return

    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    fig.suptitle("Izhikevich's spiking neuron model" if title is None else f"Izhikevich's spiking neuron model ({title})")

    axs[0].plot(t_values, v_values, "g-")
    axs[0].set_ylabel("Potential (mv)")
    axs[0].set_xlabel("Time (in ms)")
    axs[0].grid()

    axs[1].plot(t_values, f_values, "r-")
    axs[1].set_ylabel("Fired")
    axs[1].set_xlabel("Time (in ms)")
    axs[1].grid()

    axs[2].plot(t_values, u_values, "r-")
    axs[2].set_ylabel("Recovery (u)")
    axs[2].set_xlabel("Time (in ms)")
    axs[2].grid()

    axs[3].plot(t_values, I_values, "r-o")
    axs[3].set_ylabel("Current (I)")
    axs[3].set_xlabel("Time (in ms)")
    axs[3].grid()

    if filename is not None:
        plt.savefig(filename)
    if view:
        plt.show()
        plt.close()
    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """Visualizes speciation throughout evolution."""
    if not check_matplotlib():
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """Draws a neural network with arbitrary topology."""
    if not check_graphviz():
        return

    if node_names is None:
        node_names = {}
    if node_colors is None:
        node_colors = {}

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)

    for k in inputs:
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    for k in outputs:
        name = node_names.get(k, str(k))
        output_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=output_attrs)

    used_nodes = set(genome.nodes.keys())
    if prune_unused:
        connections = {cg.key for cg in genome.connections.values() if cg.enabled or show_disabled}
        used_nodes = outputs.copy()
        pending = outputs.copy()
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending

    for n in used_nodes:
        if n not in inputs and n not in outputs:
            attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
            dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
    return dot