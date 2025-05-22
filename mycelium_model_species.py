import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import PillowWriter
from scipy.sparse import diags, linalg
from matplotlib.animation import FuncAnimation
import random
import datetime

species_params = {
    "A": {
        "V_max_P": 0.6,  # Maximum uptake rate of phosphate for species A
        "V_max_N": 0.8,  # Maximum uptake rate of nitrogen for species A
        "branch_prob": 0.07,  # Probability of branching for species A
        "max_branch_depth": 5,  # Maximum branching depth for species A
    },
    "B": {
        "V_max_P": 0.4,  # Maximum uptake rate of phosphate for species B
        "V_max_N": 0.7,  # Maximum uptake rate of nitrogen for species B
        "branch_prob": 0.1,  # Probability of branching for species B
        "max_branch_depth": 4,  # Maximum branching depth for species B
    },
}

params = {
    "grid_size": 100,  # Size of the simulation grid
    "dt": 0.2,  # Time step for the simulation
    "D_P": 0.5,  # Diffusion coefficient for phosphate
    "D_N": 0.6,  # Diffusion coefficient for nitrogen
    "K_m_P": 0.3, # Half-saturation constant for phosphate
    "K_m_N": 0.2, # Half-saturation constant for nitrogen
    "adhesion": 0.01,  # Adhesion parameter for cells
    "volume_constraint": 0.01,  # Constraint on cell volume
    "chemotaxis_strength": 3.0,  # Strength of chemotaxis
    "target_volume": 5000,  # Target volume for cells
    "nutrient_threshold": 0.7,  # Threshold for nutrient concentration
    "P_source_loc": (2/3, 1/4),  # Location of phosphate source as a fraction of grid size
    "N_source_loc": (2/3, 3/4),  # Location of nitrogen source as a fraction of grid size
    "P_conc": 1.0,  # Initial concentration of phosphate
    "N_conc": 1.0,  # Initial concentration of nitrogen
}

def initialise_grids(grid_size):
    """
    Initialise the nutrients and biomass on the grid.
    """
    phosphate = np.zeros((grid_size, grid_size))
    nitrogen = np.zeros((grid_size, grid_size))
    root_grid = np.zeros((grid_size, grid_size), dtype=int)
    tip_map = {}

    center = grid_size // 2
    root_grid[0, center] = 1
    tip_map[1] = (0, center, 0, True)

    p_i, p_j = int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size)
    n_i, n_j = int(params["N_source_loc"][0] * grid_size), int(params["N_source_loc"][1] * grid_size)

    phosphate[p_i, p_j] = params["P_conc"]
    nitrogen[n_i, n_j] = params["N_conc"]

    # # Access species B parameters
    # species_b_params = species_params["B"]
    # V_max_P_B = species_b_params["V_max_P"]
    # V_max_N_B = species_b_params["V_max_N"]
    # branch_prob_B = species_b_params["branch_prob"]
    # max_branch_depth_B = species_b_params["max_branch_depth"]

    return phosphate, nitrogen, root_grid, tip_map

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

def steady_state_nutrient(C_init, biomass, params, species_params, species_type, nutrient_type='P', tol=1e-4, max_iter=50):
    """
    Solve the steady-state nutrient field using a finite difference method.

    C_init: Initial concentration of the nutrient.
    biomass: Current biomass grid.
    params: Simulation parameters.
    nutrient_type: 'P' for phosphate, 'N' for nitrogen.
    tol: Tolerance for convergence.
    max_iter: Maximum number of iterations.

    Returns the updated nutrient concentration grid.
    """
    grid_size = C_init.shape[0]
    C = C_init.copy()
    
    if nutrient_type == 'P':
        D = params["D_P"]
        V_max = species_params[species_type]["V_max_P"]
        K_m = params["K_m_P"]
        source = [(int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size))]
    else:
        D = params["D_N"]
        V_max = species_params[species_type]["V_max_N"]
        K_m = params["K_m_N"]
        source = [(int(params["N_source_loc"][0] * grid_size), int(params["N_source_loc"][1] * grid_size))]

    N = grid_size * grid_size
    L = build_laplacian_matrix(grid_size, D).tolil()

    source_indices = [x * grid_size + y for (x, y) in source]

    for iter in range(max_iter):
        uptake_coeff = (V_max * biomass) / (K_m + C)
        uptake_diag = diags(uptake_coeff.flatten(), 0)

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

def get_neighbors(i, j, grid_size):
    """
    Get valid neighbors for a given cell position (i, j). Moore Neighborhood with up excluded due to geotropism.
    """
    return [(i + di, j + dj) for di, dj in [(1, 0), (0, -1), (0, 1), (1,-1), (1,1)] if 0 <= i + di < grid_size and 0 <= j + dj < grid_size]

def calculate_energy(i, j, P, N, params, grid=None):
    """
    Calculate the energy for a given cell position (i, j) based on nutrient concentrations and other parameters.
    Cellular Potts model energy function with more realistic adhesion and volume terms.

    - Chemotaxis: drives growth toward higher nutrients.
    - Adhesion: uses CPM-style boundary energy (H_adhesion).
    - Volume: penalizes deviation from a target volume (total mycelium size).
    """
    # Chemotaxis term (minimize energy by moving toward nutrients) 
    chemotaxis = params["chemotaxis_strength"] * (P[i, j] + N[i, j])

    # Adhesion: CPM-style boundary energy
    adhesion = 0
    if grid is not None:
        grid_size = params.get("grid_size")
        sigma_i = 1  # Assume new cell is mycelium (label 1)
        tau = lambda sigma: 1 if sigma > 0 else 0  # 1: mycelium, 0: medium
        J = lambda tau1, tau2: params["adhesion"] if tau1 != tau2 else 0  # Adhesion energy

        neighbors = [(i+di, j+dj) for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
                     if 0 <= i+di < grid_size and 0 <= j+dj < grid_size]
        for ni, nj in neighbors:
            sigma_j = grid[ni, nj]
            delta = 1 if sigma_i == sigma_j else 0
            adhesion += J(tau(sigma_i), tau(sigma_j)) * (1 - delta) * np.random.uniform(0, 1)

    # Volume constraint: penalize deviation from target volume
    volume_penalty = 0
    if grid is not None:
        current_volume = np.sum(grid > 0)
        target_volume = min(current_volume+5, params.get("target_volume"))
        volume_penalty = params["volume_constraint"] * (current_volume - target_volume) ** 2 * np.random.uniform(0, 1)

    return -chemotaxis + adhesion + volume_penalty

def grow_tips(grid, P, N, tips, params, species_params, species_type):
    """
    Grow the tips of the mycelium based on nutrient uptake.
    """
    new_tips = {}
    cell_id = max(tips.keys()) + 1 if tips else 2
    grid_size = grid.shape[0]

    for tid, (i, j, gen, is_main) in tips.items():
        if P[i, j]>= 0.3 or N[i, j] >= 0.3:
                if species_type == "A":
                    branching_probability = 0.2
                elif species_type == "B":
                    branching_probability = 0.25
        else:
            if species_type == "A":
                branching_probability = 0.1
            elif species_type == "B":
                branching_probability = 0.15

    for tid, (i, j, gen, is_main) in tips.items():
        if i == grid_size - 1:
            break
        if (P[i, j] > params["nutrient_threshold"]) or (N[i, j] > params["nutrient_threshold"]):
            continue

        neighbors = get_neighbors(i, j, grid_size)
        candidates = [pos for pos in neighbors if grid[pos] == 0]
        if not candidates:
            continue

        if is_main:
            candidates = [pos for pos in candidates if pos[0] > i] or candidates

        # Metroplis algorithm to select the best candidate
        scored = [(pos, calculate_energy(pos[0], pos[1], P, N, params, grid)) for pos in candidates]
        scored.sort(key=lambda x: x[1])
        best = scored[0][0]

        current_energy = calculate_energy(i, j, P, N, params, grid)
        new_energy = calculate_energy(best[0], best[1], P, N, params, grid)
        delta_E = new_energy - current_energy

        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E):
            grid[best] = 1
            new_tips[cell_id] = (best[0], best[1], gen, is_main)
            cell_id += 1

        if np.random.rand() < branching_probability and gen < species_params[species_type]["max_branch_depth"]:
            branch_dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            random.shuffle(branch_dirs)
            for di, dj in branch_dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size and grid[ni, nj] == 0:
                    grid[ni, nj] = 1
                    new_tips[cell_id] = (ni, nj, gen + 1, False)
                    cell_id += 1
                    break

    return grid, new_tips

def animate_simulation(P, N, M, tips, params, species_params, species_type, num_frames=400):
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((params["grid_size"], params["grid_size"], 3)))

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # To store key frames
    snapshots = {}
    frame1, frame2, frame3 = 5, 50, num_frames - 1
    capture_frames = [frame1, frame2, frame3]

    def update(frame):
        nonlocal P, N, M, tips, snapshots
        P = steady_state_nutrient(P, M, params, species_params, species_type, nutrient_type='P')
        N = steady_state_nutrient(N, M, params, species_params, species_type, nutrient_type='N')
        M, tips = grow_tips(M, P, N, tips, params, species_params, species_type)

        rgb_image = np.ones((params["grid_size"], params["grid_size"], 3)) * [0.4, 0.26, 0.13]
        rgb_image[..., 0] += P
        rgb_image[..., 2] += N
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

    # Save animation
    now = datetime.datetime.now()
    path1 = f"results/mycelium_growth_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{species_type}_P{params['P_source_loc'][0]:.2f}_{params['P_source_loc'][1]:.2f}_{params['P_conc']:.2f}_N{params['N_source_loc'][0]:.2f}_{params['N_source_loc'][1]:.2f}_{params['N_conc']:.2f}.gif"
    ani.save(path1, writer=PillowWriter(fps=20))
    print(f"Animation saved to {path1}")

    # Create subplot of beginning, middle, end
    fig_snap, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig_snap.suptitle(f"Species={species_type}, P={params['P_conc']:.2f}, N={params['N_conc']:.2f}")
    for ax, idx in zip(axs, capture_frames):
        ax.imshow(snapshots[idx])
        ax.set_title(f"Frame {idx}")
        ax.axis('off')

    # Save subplot
    path2 = f"results/mycelium_snapshots_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{species_type}_P{params['P_source_loc'][0]:.2f}_{params['P_source_loc'][1]:.2f}_{params['P_conc']:.2f}_N{params['N_source_loc'][0]:.2f}_{params['N_source_loc'][1]:.2f}_{params['N_conc']:.2f}.png"
    fig_snap.savefig(path2)
    print(f"Snapshots saved to {path2}")
    plt.close(fig)
    plt.close(fig_snap)

if __name__ == "__main__":
    P, N, M, tips = initialise_grids(params["grid_size"])
    # species_type = "B"
    animate_simulation(P, N, M, tips, params, species_params, "A", num_frames=400)
    animate_simulation(P, N, M, tips, params, species_params, "B", num_frames=400)

