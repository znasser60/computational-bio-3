import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import PillowWriter
from scipy.sparse import diags, linalg
from matplotlib.animation import FuncAnimation
import random
import datetime
import pickle 

species_params = {
    "A": {
        "V_max_P": 0.5,  # Maximum uptake rate of phosphate for species A
        "branch_prob": 0.1,  # Probability of branching for species A
        "max_branch_depth": 5,  # Maximum branching depth for species A
    },
    "B": {
        "V_max_P": 0.7,  # Maximum uptake rate of phosphate for species B
        "branch_prob": 0.15,  # Probability of branching for species B
        "max_branch_depth": 5,  # Maximum branching depth for species B
    },
}

params = {
    "runs": 15, # Number of runs for simulation
    "num_frames": 400, # Number of frames to simulate
    "grid_size": 100,  # Size of the simulation grid
    "dt": 0.2,  # Time step for the simulation
    "D_P": 0.5,  # Diffusion coefficient for phosphate
    "K_m_P": 0.3, # Half-saturation constant for phosphate
    "adhesion": 0.01,  # Adhesion parameter for cells
    "volume_constraint": 0.01,  # Constraint on cell volume
    "nutrient_threshold": 0.7,  # Threshold for nutrient concentration
    "chemotaxis_strength": 3.0,  # Strength of chemotaxis
    "target_volume": 5000,  # Target volume for the cells   
    "P_source_loc": (1/2, 1/2),  # Location of phosphate source as a fraction of grid size
    "P_conc": 1.0  # Initial concentration of phosphate
}

def initialise_grids(grid_size):
    """
    Initialise the nutrients and biomass of the two species on the grid.
    """
    phosphate = np.zeros((grid_size, grid_size))
    root1_grid = np.zeros((grid_size, grid_size), dtype=int)
    root2_grid = np.zeros((grid_size, grid_size), dtype=int)
    tip1_map = {}
    tip2_map = {}

    center = grid_size // 2
    x1 = center
    y1 = 0
    x2 = center
    y2 = grid_size - 1
    root1_grid[x1, y1] = 1
    root2_grid[x2, y2] = 1 
    tip1_map[1] = (x1, y1, 0, True)
    tip2_map[1] = (x2, y2, 0, True)

    p_i, p_j = int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size)
    phosphate[p_i, p_j] = params["P_conc"]

    return phosphate, root1_grid, root2_grid, tip1_map, tip2_map

def build_laplacian_matrix(grid_size, D):
    """
    Build a Laplacian matrix for a 2D grid with Dirichlet boundary conditions.
    """
    N = grid_size * grid_size
    main_diag = -4 * np.ones(N)
    side_diag = np.ones(N - 1)
    side_diag[np.arange(1, N) % grid_size == 0] = 0  
    up_down_diag = np.ones(N - grid_size)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    offsets = [0, -1, 1, -grid_size, grid_size]
    L = diags(diagonals, offsets, shape=(N, N), format='lil')

    L = D * L
    return L

def steady_state_nutrient(C_init, biomass1, biomass2, params, species_params, species_type1, species_type2, nutrient_type='P', tol=1e-4, max_iter=50):
    """
    Solve the steady-state nutrient field using a finite difference method.

    C_init: Initial concentration of the nutrient.
    biomass1, biomass2: Current biomass grids for the two species.
    params: Simulation parameters.
    nutrient_type: 'P' for phosphate, 'N' for nitrogen.
    tol: Tolerance for convergence.
    max_iter: Maximum number of iterations.

    Returns the updated nutrient concentration grid.
    """
    grid_size = C_init.shape[0]
    C = C_init.copy()
    
    D = params["D_P"]
    V_max1 = species_params[species_type1]["V_max_P"]
    V_max2 = species_params[species_type2]["V_max_P"]
    K_m = params["K_m_P"]
    source = [(int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size))]

    N = grid_size * grid_size
    L = build_laplacian_matrix(grid_size, D).tolil()

    source_indices = [x * grid_size + y for (x, y) in source]

    for iter in range(max_iter):
        uptake_coeff1 = (V_max1 * biomass1) / (K_m + C)
        uptake_coeff2 = (V_max2 * biomass2) / (K_m + C)
        uptake_diag = diags((uptake_coeff1 + uptake_coeff2).flatten(), 0)

        # Build system matrix: D * Laplacian - uptake
        A = (L - uptake_diag).tocsr()

        # RHS is zero except for Dirichlet boundaries
        b = np.zeros(N)
        for idx in source_indices:
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = 1.0

        # Solve matrices for steady-state nutrient concentration
        C_new = linalg.spsolve(A, b).reshape((grid_size, grid_size))
        C_new = np.clip(C_new, 0, 1)

        if np.linalg.norm(C_new - C) < tol:
            break
        C = C_new

    return C


def get_neighbors(i, j, grid_size, M2, species_type):
    """
    Get valid neighbors for a given cell position (i, j). Moore Neighborhood without growing backwards.
    M2 - the grid of the species that is not growing. you need to know what is occupied by it so that you don't go there.
    species_type - you need to let this function know whic species you're modeling so that it knows which direction is backwards because we dont want to grow there
    """

    # Dont let the fungi grow backward
    if species_type == "A":
        neighbors = [(i + di, j + dj) for di, dj in [(1, 0), (0, 1), (-1, 0), (-1,1), (1,1)] if 0 <= i + di < grid_size and 0 <= j + dj < grid_size and M2[i + di, j + dj] == 0]
    else:
        neighbors = [(i + di, j + dj) for di, dj in [(1, 0), (0, -1), (-1, 0), (-1, -1), (1, -1)] if 0 <= i + di < grid_size and 0 <= j + dj < grid_size and M2[i + di, j + dj] == 0]

    # Detect if there is a cell in the second level of von Neumann neighborhood that is occupied by the other species and get rid of the closest neighbors to that cell so that it doesn't grow in that direction
    for di, dj in [(-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2), (-2, -1), (2, -1), (-2, 0), (2, 0), (-2, 1), (2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2)]:
            if 0 <= i + di < grid_size and 0 <= j + dj < grid_size and M2[i + di, j + dj] == 1:
                distances = [(np.sqrt(abs(x[0] - (i + di))**2 + abs(x[1] - (j + dj))**2), x) for x in neighbors]
                # distances = [(abs(x[0] - (i + di)) + abs(x[1] - (j + dj)), x) for x in neighbors]
                distances.sort(key=lambda x: x[0])

                if len(distances) > 1 and distances[0][0] == distances[1][0]:
                    to_remove = [distances[0][1], distances[1][1]]
                elif len(distances) > 0:
                    to_remove = [distances[0][1]]
                else:
                    continue

                if len(to_remove) > 0:
                    neighbors = [n for n in neighbors if n not in to_remove]
            else:    
                continue

    return neighbors

def calculate_energy(i, j, P, params, grid=None):
    """
    Calculate the energy of a cell at position (i, j) based on nutrient concentration, adhesion, and volume constraints.
    The energy function is based on the cellular Potts model.
    """
    chemotaxis = params["chemotaxis_strength"] * P[i, j]
    adhesion = 0
    if grid is not None:
        grid_size = params.get("grid_size")
        sigma_i = 1
        tau = lambda sigma: 1 if sigma > 0 else 0
        J = lambda tau1, tau2: params["adhesion"] if tau1 != tau2 else 0
        neighbors = [(i+di, j+dj) for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] if 0 <= i+di < grid_size and 0 <= j+dj < grid_size]
        for ni, nj in neighbors:
            sigma_j = grid[ni, nj]
            delta = 1 if sigma_i == sigma_j else 0
            adhesion += J(tau(sigma_i), tau(sigma_j)) * (1 - delta) * np.random.uniform(0, 1)
    volume_penalty = 0
    if grid is not None:
        current_volume = np.sum(grid > 0)
        target_volume = min(current_volume+5, params.get("target_volume"))
        volume_penalty = params["volume_constraint"] * (current_volume - target_volume) ** 2 * np.random.uniform(0, 1)

    return -chemotaxis + adhesion + volume_penalty

def grow_tips(grid1, grid2, P, tips, params, species_params, species_type):
    """
    Grow the tips of the mycelium based on nutrient uptake.

    grid1 - the grid where the tips are growing
    grid2 - the grid of the other species. this info needs to be known so you dont grow into a grid that's already occupied.
    """
    new_tips = {}
    cell_id = max(tips.keys()) + 1 if tips else 2
    grid_size = grid1.shape[0]

    
    for tid, (i, j, gen, is_main) in tips.items():
        if P[i, j]>= 0.3: 
            branching_probability = species_params[species_type]["branch_prob"] * 2
        else:
            branching_probability = species_params[species_type]["branch_prob"]
    
    for tid, (i, j, gen, is_main) in tips.items():
        if P[i, j] > params["nutrient_threshold"]:
            continue

        neighbors = get_neighbors(i, j, grid_size, grid2, species_type)
        candidates = [pos for pos in neighbors if grid1[pos] == 0]
        if not candidates:
            continue

        scored = [(pos, calculate_energy(pos[0], pos[1], P, params, grid1)) for pos in candidates]
        scored.sort(key=lambda x: x[1])
        best = scored[0][0]
        grid1[best] = 1
        new_tips[cell_id] = (best[0], best[1], gen, is_main)
        cell_id += 1

        if np.random.rand() < branching_probability and gen < species_params[species_type]["max_branch_depth"]:
            branch_dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            random.shuffle(branch_dirs)
            for di, dj in branch_dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size and grid1[ni, nj] == 0:
                    grid1[ni, nj] = 1
                    new_tips[cell_id] = (ni, nj, gen + 1, False)
                    cell_id += 1
                    break

    return grid1, new_tips

def pushDataExport(data_export, frame, M1, tips1, M2, tips2, P):
    # general: frame / M_1 / TIPS_1 / M_2 / TIPS_2 / P
    result = [frame, np.sum(M1), len(tips1), np.sum(M2), len(tips2), np.sum(P)]
    data_export['time'].append(result)

    # tips
    result1 = [[t[0], t[1]] for t in tips1.items()]
    result2 = [[t[0], t[1]] for t in tips2.items()]

    data_export['tips'][0].append(result1)
    data_export['tips'][1].append(result2)

    return data_export

def animate_simulation(P, M1, M2, tips1, tips2, params, species_params, species_type1, species_type2, image_filename, data_export, num_frames=400):
    """
    Animate the simulation of mycelium growth over time.
    """    
    #have to do everything twice, once for each species
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((params["grid_size"], params["grid_size"], 3)))

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # To store key frames
    snapshots = {}
    frame1, frame2, frame3 = 5, 50, num_frames - 1
    capture_frames = [frame1, frame2, frame3]

    """
    Update the state of the two species (M1 and M2) and nutrient concentrations (P and N) for one frame of the animation.
    """
    def update(frame):
        nonlocal P, M1, M2, tips1, tips2, snapshots, data_export

        P = steady_state_nutrient(P, M1, M2, params, species_params, species_type1, species_type2, nutrient_type='P')
        M1, tips1 = grow_tips(M1, M2, P, tips1, params, species_params, species_type1)
        M2, tips2 = grow_tips(M2, M1, P, tips2, params, species_params, species_type2)

        # collect data
        data_export = pushDataExport(data_export, frame, M1, tips1, M2, tips2, P)

        # create images
        rgb_image = np.ones((params["grid_size"], params["grid_size"], 3)) * [0.4, 0.26, 0.13]
        rgb_image[..., 0] += P
        rgb_image = np.clip(rgb_image, 0, 1)

        for i in range(params["grid_size"]):
            for j in range(params["grid_size"]):
                if M1[i, j] > 0:
                    rgb_image[i, j] = [0.0, 1.0, 0.0]  # green
                if M2[i, j] > 0:
                    rgb_image[i, j] = [0.0, 1.0, 1.0]  # cyan

        for tid, (i, j, _, _) in tips1.items():
            rgb_image[i, j] = [1, 1, 1]
        for tid, (i, j, _, _) in tips2.items():
            rgb_image[i, j] = [1, 1, 1]

        if frame in capture_frames:
            snapshots[frame] = rgb_image.copy()

        im.set_array(rgb_image)
        return [im]

    # Save animation
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    now = datetime.datetime.now()
    filename = f"results/mycelium_growth_{image_filename}_{species_type1}_{species_type2}.gif"
    ani.save(filename, writer=PillowWriter(fps=20))
    print(f"Animation saved to {filename}")

    # Create subplot of beginning, middle, end
    fig_snap, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig_snap.suptitle(f"Species1={species_type1}, Species2={species_type2}, P={params['P_conc']:.2f}")
    for ax, idx in zip(axs, capture_frames):
        ax.imshow(snapshots[idx])
        ax.set_title(f"Frame {idx}")
        ax.axis('off')
    fig_snap.savefig(filename.replace(".gif", "_snapshots.png"))
    print(f"Snapshots saved.")

    plt.close(fig)
    plt.close(fig_snap)

    return data_export

def getTimestamp():
    now = datetime.datetime.now()
    return now.strftime('%m-%d_%H-%M')

if __name__ == "__main__":
    # simulation sweep data object
    sweep_timestamp = getTimestamp()
    data_export_folder = "data"
    data_export_file = f"{data_export_folder}/{sweep_timestamp}"

    data_export = {
        'params': params,
        'runs': [],
        'timestamp': sweep_timestamp
    }

    # sweep loop
    for ii in range(params["runs"]):
        image_filename = f'{sweep_timestamp}-run-{ii}'

        simulation_data = {
            'tips' : [[],[]],
            'time' : [],
            'run_id': ii,
            'start': getTimestamp(),
            'image_filename': image_filename
        }

        P, M1, M2, tips1, tips2 = initialise_grids(params["grid_size"])
        simulation_data = animate_simulation( P, M1, M2, tips1, tips2, params, species_params, "A", "B", image_filename, simulation_data, num_frames=params["num_frames"])

        data_export['runs'].append(simulation_data)
        
    # Ensure results directory exists
    os.makedirs(data_export_folder, exist_ok=True)

    # export
    with open(f'{data_export_file}.pickle', 'wb') as f:
        pickle.dump(data_export, f, pickle.HIGHEST_PROTOCOL)