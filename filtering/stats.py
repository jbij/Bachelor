# %% [markdown]
# ## Get network stats

# %%
import pickle
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import powerlaw
import matplotlib.pyplot as plt

# %%
def graph_stats(G):
    """Returns various statistics of a given graph G."""
    stats = {}
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)
    
    degrees = [d for n, d in G.degree()]
    stats['mean_degree'] = np.mean(degrees)
    stats['std_degree'] = np.std(degrees)
    stats['global_clustering_coeff'] = nx.transitivity(G)
    
    stats['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    
    # Giant Connected Component (GCC)
    if nx.is_directed(G) == False:
        largest_cc = max(nx.connected_components(G), key=len)
        GCC = G.subgraph(largest_cc)
        stats['GCC_size'] = len(GCC)
    
    # Power-law fitting
    try:
        fit = powerlaw.Fit(degrees)
        stats['power_law_alpha'] = fit.alpha  # Power-law exponent
        stats['power_law_xmin'] = fit.xmin  # Minimum value where power law applies
        stats['ks_test_statistic'] = fit.D  # KS test statistic (lower is better)
        
        # Compare power law with other distributions
        distribution_list = ['lognormal', 'exponential', 'truncated_power_law']
        comparison_results = {}
        
        for dist in distribution_list:
            R, p = fit.distribution_compare('power_law', dist)
            comparison_results[dist] = (R, p)
        
        # Find the best alternative distribution (lowest R and p-value)
        best_fit = min(comparison_results.items(), key=lambda x: (x[1][0], x[1][1]))
        best_dist, (best_R, best_p) = best_fit
        
        stats['best_powerlaw_comparison'] = f"power law vs {best_dist}: R = {best_R:.3f}, p = {best_p:.3f}"
    
    except Exception as e:
        stats['power_law_test_error'] = str(e)
    
    return stats