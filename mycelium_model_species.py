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
        "V_max_P": 0.4,  # Maximum uptake rate of phosphate for species B
        "branch_prob": 0.15,  # Probability of branching for species B
        "max_branch_depth": 4,  # Maximum branching depth for species B
    },
}

params = {
    "runs": 15, # Number of runs for simulation
    "num_frames": 400, # Number of frames to simulate
    "grid_size": 100, # Size of the grid
    "dt": 0.2, # Time step for the simulation
    "D_P": 0.5, # Diffusion coefficient 
    "K_m_P": 0.3, # Half-saturation constant 
    "adhesion": 0.01, # Adhesion strength
    "volume_constraint": 0.01, # Volume constraint for the cells
    "chemotaxis_strength": 3.0, # Strength of chemotaxis
    "target_volume": 5000, # Target volume for the cells
    "nutrient_threshold": 0.7, # Threshold for nutrient concentration to stop growth
    "P_source_loc": (0.5, 0.5), # Relative location of nutrient source as fractions of grid size
    "P_conc": 1.0, # Initial source nutrient of phosphate
}

def initialise_grids(grid_size):
    """
    Initialise the nutrients and biomass on the grid. A phosphate source is placed at the center of the grid, 
    and the root grid is initialized with a single cell at the top center.
    """
    phosphate = np.zeros((grid_size, grid_size))
    root_grid = np.zeros((grid_size, grid_size), dtype=int)
    tip_map = {}

    center = grid_size // 2
    root_grid[0, center] = 1
    tip_map[1] = (0, center, 0, True)

    p_i, p_j = int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size)
    phosphate[p_i, p_j] = params["P_conc"]

    return phosphate, root_grid, tip_map

def build_laplacian_matrix(grid_size, D):
    """
    Build a Laplacian matrix for a 2D grid with Dirichlet boundary conditions.
    The matrix is used to model the diffusion of nutrients across the grid.
    """
    N = grid_size * grid_size
    main_diag = -4 * np.ones(N)
    side_diag = np.ones(N - 1)
    side_diag[np.arange(1, N) % grid_size == 0] = 0
    up_down_diag = np.ones(N - grid_size)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    offsets = [0, -1, 1, -grid_size, grid_size]
    L = diags(diagonals, offsets, shape=(N, N), format='lil')

    return D * L

def steady_state_phosphate(C_init, biomass, params, species_params, species_type='A', tol=1e-4, max_iter=50):
    """
    Calculate the steady-state concentration of phosphate in the grid using a finite difference method.
    """
    grid_size = C_init.shape[0]
    C = C_init.copy()
    D = params["D_P"]
    V_max = species_params[species_type]["V_max_P"]
    K_m = params["K_m_P"]
    source = [(int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size))]

    N = grid_size * grid_size
    L = build_laplacian_matrix(grid_size, D).tolil()
    source_indices = [x * grid_size + y for (x, y) in source]

    for _ in range(max_iter):
        uptake_coeff = (V_max * biomass) / (K_m + C)
        uptake_diag = diags(uptake_coeff.flatten(), 0)

        A = (L - uptake_diag).tocsr()
        b = np.zeros(N)
        for idx in source_indices:
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = 1.0

        C_new = linalg.spsolve(A, b).reshape((grid_size, grid_size))
        C_new = np.clip(C_new, 0, 1)

        if np.linalg.norm(C_new - C) < tol:
            break
        C = C_new

    return C

def get_neighbors(i, j, grid_size):
    """
    Get the valid neighbors of a cell in the grid. Up directions are removed due to energy conservation.
    """
    return [(i + di, j + dj) for di, dj in [(1, 0), (0, -1), (0, 1), (1,-1), (1,1)] if 0 <= i + di < grid_size and 0 <= j + dj < grid_size]

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

def grow_tips(grid, P, tips, params, species_params, species_type):
    """
    Grow the tips of the mycelium based on the nutrient concentration and other parameters.
    """
    new_tips = {}
    cell_id = max(tips.keys()) + 1 if tips else 2
    grid_size = grid.shape[0]

    for tid, (i, j, gen, is_main) in tips.items():
        if P[i, j]>= 0.3: 
            branching_probability = species_params[species_type]["branch_prob"] * 2
        else:
            branching_probability = species_params[species_type]["branch_prob"]

    for tid, (i, j, gen, is_main) in tips.items():
        if i == grid_size - 1 or P[i, j] > params["nutrient_threshold"]:
            continue

        neighbors = get_neighbors(i, j, grid_size)
        candidates = [pos for pos in neighbors if grid[pos] == 0]
        if not candidates:
            continue

        if is_main:
            candidates = [pos for pos in candidates if pos[0] > i] or candidates

        scored = [(pos, calculate_energy(pos[0], pos[1], P, params, grid)) for pos in candidates]
        best = min(scored, key=lambda x: x[1])[0]

        current_energy = calculate_energy(i, j, P, params, grid)
        new_energy = calculate_energy(best[0], best[1], P, params, grid)
        delta_E = new_energy - current_energy

        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E):
            grid[best] = 1
            new_tips[cell_id] = (best[0], best[1], gen, is_main)
            cell_id += 1

        if np.random.rand() < branching_probability and gen < species_params[species_type]["max_branch_depth"]:
            random.shuffle(neighbors)
            for ni, nj in neighbors:
                if grid[ni, nj] == 0:
                    grid[ni, nj] = 1
                    new_tips[cell_id] = (ni, nj, gen + 1, False)
                    cell_id += 1
                    break

    return grid, new_tips

def pushDataExport(data_export, frame, M, P, tips):
    # general: frame / M_1 / P / TIPS_1
    entry = [frame, np.sum(M), np.sum(P), len(tips)]
    data_export['time'].append(entry)

    # tips
    result = [[t[0], t[1]] for t in tips.items()]
    data_export['tips'][0].append(result)

    return data_export

def animate_simulation(P, M, tips, params, species_params, species_type, image_filename, data_export, num_frames=400):
    """
    Animate the simulation of mycelium growth over time.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((params["grid_size"], params["grid_size"], 3)))

    os.makedirs("results", exist_ok=True)
    snapshots = {}
    capture_frames = [5, 50, num_frames - 1]

    def update(frame):
        nonlocal P, M, tips, snapshots, data_export
        P = steady_state_phosphate(P, M, params, species_params, species_type)
        M, tips = grow_tips(M, P, tips, params, species_params, species_type)

        # collect data
        data_export = pushDataExport(data_export, frame, M, P, tips)

        # update images
        rgb_image = np.ones((params["grid_size"], params["grid_size"], 3)) * [0.4, 0.26, 0.13]
        rgb_image[..., 0] += P
        rgb_image = np.clip(rgb_image, 0, 1)

        for i in range(params["grid_size"]):
            for j in range(params["grid_size"]):
                if M[i, j] > 0:
                    rgb_image[i, j] = [0.8, 0.52, 0.25]

        for tid, (i, j, _, _) in tips.items():
            rgb_image[i, j] = [1, 1, 1]

        if frame in capture_frames:
            snapshots[frame] = rgb_image.copy()

        im.set_array(rgb_image)
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    now = datetime.datetime.now()
    filename = f"results/mycelium_growth_{image_filename}_{species_type}.gif"
    ani.save(filename, writer=PillowWriter(fps=20))
    print(f"Animation saved to {filename}")

    fig_snap, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig_snap.suptitle(f"Species={species_type}, P={params['P_conc']:.2f}")
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

        P, M, tips = initialise_grids(params["grid_size"])
        simulation_data = animate_simulation(P, M, tips, params, species_params, "A", image_filename, simulation_data, num_frames=params["num_frames"])

        data_export['runs'].append(simulation_data)

    # Ensure results directory exists
    os.makedirs(data_export_folder, exist_ok=True)

    # export
    with open(f'{data_export_file}.pickle', 'wb') as f:
        pickle.dump(data_export, f, pickle.HIGHEST_PROTOCOL)