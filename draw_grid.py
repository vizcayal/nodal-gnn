import h5py
import pandapower as pp
import pandapower.networks as networks
import pandapower.plotting as plotting
import matplotlib.pyplot as plt
import os
import numpy as np

def draw_grid_sample(h5_path, sample_idx=0):
    print(f"Reading sample {sample_idx} from {h5_path}")
    
    # Load standard IEEE 30-bus network
    net = networks.case30()
    
    with h5py.File(h5_path, 'r') as f:
        # Get statuses for the sample
        branch_status = f['branch_status'][sample_idx]
        gen_status = f['gen_status'][sample_idx]
        pd = f['pd'][sample_idx]
        
        # 1. Update line statuses
        # PGLib/MATPOWER order for branches in IEEE 30 is usually consistent
        # We assume 41 branches (lines + transformers)
        for i in range(len(branch_status)):
            if i < len(net.line):
                net.line.at[i, 'in_service'] = bool(branch_status[i])
            else:
                # The rest are likely transformers
                trafo_idx = i - len(net.line)
                if trafo_idx < len(net.trafo):
                    net.trafo.at[trafo_idx, 'in_service'] = bool(branch_status[i])

        # 2. Update generator statuses
        # In case30, there is 1 external grid and 5 generators
        # We need to map 6 statuses to (ext_grid + 5 gens)
        net.ext_grid.at[0, 'in_service'] = bool(gen_status[0])
        for i in range(1, len(gen_status)):
            gen_idx = i - 1
            if gen_idx < len(net.gen):
                net.gen.at[gen_idx, 'in_service'] = bool(gen_status[i])

        # 3. Update loads
        # 21 loads in input.h5. In case30, there are 21 loads.
        for i in range(len(pd)):
            if i < len(net.load):
                # Scale load or just set in-service
                net.load.at[i, 'p_mw'] = pd[i] * 100 # Assuming p.u. to MW
                net.load.at[i, 'in_service'] = pd[i] > 0

    # Draw the grid
    print("Generating plot...")
    plt.figure(figsize=(12, 10))
    
    # Simple plot using pandapower plotting
    # We'll use a specific layout if available, otherwise generic
    try:
        # Try to use internal coordinates if they exist
        plotting.simple_plot(net, show_plot=False, plot_loads=True, plot_gens=True)
    except:
        # Fallback to manual networkx plot if coords are missing
        import networkx as nx
        G = pp.topology.create_nxgraph(net)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')
    
    plt.title(f"IEEE 30-Bus Grid - Sample {sample_idx}\n(PGLearn-Small)")
    
    output_img = "grid_plot.png"
    plt.savefig(output_img)
    print(f"Grid plot saved to: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    h5_path = r"C:\Users\i34005\OneDrive - Wood Mackenzie Limited\CSML\PGLearn-Small-30_ieee\train\input.h5"
    if os.path.exists(h5_path):
        draw_grid_sample(h5_path, sample_idx=0)
    else:
        print("Input file not found.")
