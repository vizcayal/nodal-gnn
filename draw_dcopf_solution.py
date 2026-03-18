import h5py
import pandapower as pp
import pandapower.networks as networks
import pandapower.plotting as plotting
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.colors as colors

def draw_dcopf_solution(base_path, sample_idx=1):
    print(f"Drawing DCOPF solution for sample {sample_idx}...")
    
    input_path = os.path.join(base_path, "input.h5")
    primal_path = os.path.join(base_path, "DCOPF", "primal.h5")
    dual_path = os.path.join(base_path, "DCOPF", "dual.h5")
    
    # Load standard IEEE 30-bus network
    net = networks.case30()
    
    try:
        with h5py.File(input_path, 'r') as f_in, \
             h5py.File(primal_path, 'r') as f_pri, \
             h5py.File(dual_path, 'r') as f_dua:
            
            # 1. Update Topology from input.h5
            branch_status = f_in['branch_status'][sample_idx]
            gen_status = f_in['gen_status'][sample_idx]
            pd = f_in['pd'][sample_idx]
            
            # Map branch status
            for i in range(len(branch_status)):
                if i < len(net.line): net.line.at[i, 'in_service'] = bool(branch_status[i])
                else:
                    trafo_idx = i - len(net.line)
                    if trafo_idx < len(net.trafo): net.trafo.at[trafo_idx, 'in_service'] = bool(branch_status[i])

            # Map generation status
            net.ext_grid.at[0, 'in_service'] = bool(gen_status[0])
            for i in range(1, len(gen_status)):
                gen_idx = i - 1
                if gen_idx < len(net.gen): net.gen.at[gen_idx, 'in_service'] = bool(gen_status[i])

            # 2. Add Results from primal.h5
            pf = f_pri['pf'][sample_idx]
            pg = f_pri['pg'][sample_idx]
            
            # Add results to the net dataframe for plotting tools to find them
            # pf is flow on lines
            net.res_line = net.line.copy()
            net.res_trafo = net.trafo.copy()
            for i in range(len(pf)):
                if i < len(net.line): net.res_line.at[i, 'p_mw'] = pf[i] * 100
                else: 
                    t_idx = i - len(net.line)
                    if t_idx < len(net.trafo): net.res_trafo.at[t_idx, 'p_mw'] = pf[i] * 100

            # 3. Add Congestion Info from dual.h5
            kcl_p = f_dua['kcl_p'][sample_idx] # LMPs
            pf_dual = f_dua['pf'][sample_idx] # Line duals (non-zero means congested)
            
            # Store LMPs in bus table for plotting
            net.bus['lmp'] = kcl_p
            
            # Identify congested lines (dual > 0)
            congested_lines = np.where(np.abs(pf_dual) > 1e-4)[0]
            print(f"Congested lines detected at indices: {congested_lines}")

        # --- PLOTTING ---
        plt.figure(figsize=(14, 12))
        
        # Color nodes by LMP
        node_values = net.bus['lmp'].values
        norm = colors.Normalize(vmin=min(node_values), vmax=max(node_values))
        cmap = plt.get_cmap('jet')
        node_colors = [cmap(norm(v)) for v in node_values]
        
        # Create plotting collections
        # Buses
        bc = plotting.create_bus_collection(net, buses=net.bus.index, color=node_colors, zorder=2, size=0.1)
        
        # Lines (thicker if flow is high)
        line_widths = np.abs(net.res_line.p_mw.values) / 10 + 1
        lc = plotting.create_line_collection(net, net.line.index, color='gray', linewidths=line_widths, use_bus_geodata=True, zorder=1)
        
        # Highlight congested lines in Red
        clc = None
        if len(congested_lines) > 0:
            c_lines = [i for i in congested_lines if i < len(net.line)]
            if c_lines:
                clc = plotting.create_line_collection(net, c_lines, color='red', linewidths=5, label='Congested Line', zorder=3)

        # Plot
        ax = plt.gca()
        plotting.draw_collections([bc, lc] + ([clc] if clc else []), ax=ax)
        
        sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([]) # Required for matplotlib 3.1+
        plt.colorbar(sm, ax=ax, label='LMP (Nodal Price)')
        
        plt.title(f"DCOPF Solution - IEEE 30-Bus System (Sample {sample_idx})\nRed lines = Congested | Node Color = Price")
        
        output_img = "dcopf_solution_plot.png"
        plt.savefig(output_img)
        print(f"Solution plot saved to: {os.path.abspath(output_img)}")
        
    except Exception as e:
        print(f"Error drawing solution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    base_path = r"C:\Users\i34005\OneDrive - Wood Mackenzie Limited\CSML\PGLearn-Small-30_ieee\train"
    draw_dcopf_solution(base_path, sample_idx=1)
