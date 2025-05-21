import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import PillowWriter
from scipy.sparse import diags, linalg
from matplotlib.animation import FuncAnimation
import random
import datetime

params = {
    "grid_size": 100, # Size of the grid
    "dt": 0.2, # Time step for the simulation
    "D_P": 0.5, # Diffusion coefficient for phosphate
    "D_N": 0.6, # Diffusion coefficient for nitrogen
    "V_max_P": 0.6, # Maximum uptake rate for phosphate
    "V_max_N": 0.8, # Maximum uptake rate for nitrogen
    "K_m_P": 0.3, # Half-saturation constant for phosphate
    "K_m_N": 0.2, # Half-saturation constant for nitrogen
    "branch_prob": 0.07, # Probability of branching
    "adhesion": 0.1, # Adhesion strength
    "volume_constraint": 0.01, # Volume constraint for the cells
    "chemotaxis_strength": 3.0, # Strength of chemotaxis
    "max_branch_depth": 5, # Maximum depth of branching, restricts length of mycelium
    "nutrient_threshold": 0.7, # Threshold for nutrient concentration to stop growth
    "grid_size": 100,  # Size of the grid
    "P_source_loc": (2/3, 1/4),  # Relative location of phosphate source as fractions of grid size
    "N_source_loc": (2/3, 3/4),  # Relative location of nitrogen source as fractions of grid size
    "P_conc": 1.0,  # Concentration of phosphate
    "N_conc": 0.75,  # Concentration of nitrogen
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

def steady_state_nutrient(C_init, biomass, params, nutrient_type='P', tol=1e-4, max_iter=50):
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
        V_max = params["V_max_P"]
        K_m = params["K_m_P"]
        source = [(int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size))]
    else:
        D = params["D_N"]
        V_max = params["V_max_N"]
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
    return [(i + di, j + dj) for di, dj in [(1, 0), (0, -1), (0, 1)] if 0 <= i + di < grid_size and 0 <= j + dj < grid_size]

def calculate_energy(i, j, P, N, params):
    """
    Calculate the energy for a given cell position (i, j) based on nutrient concentrations and other parameters.
    Cellular potts model energy function.
    """
    chemotaxis = params["chemotaxis_strength"] * (P[i, j] + N[i, j])
    adhesion = random.uniform(0, params["adhesion"])
    volume_penalty = random.uniform(0, params["volume_constraint"])
    return -chemotaxis + adhesion + volume_penalty

def grow_tips(grid, P, N, tips, params):
    """
    Grow the tips of the mycelium based on nutrient uptake.
    """
    new_tips = {}
    cell_id = max(tips.keys()) + 1 if tips else 2
    grid_size = grid.shape[0]

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

        scored = [(pos, calculate_energy(pos[0], pos[1], P, N, params)) for pos in candidates]
        scored.sort(key=lambda x: x[1])
        best = scored[0][0]
        grid[best] = 1
        new_tips[cell_id] = (best[0], best[1], gen, is_main)
        cell_id += 1

        if np.random.rand() < params["branch_prob"] and gen < params["max_branch_depth"]:
            branch_dirs = [(-1, -1), (-1, 1), (0, -1), (0, 1)]
            random.shuffle(branch_dirs)
            for di, dj in branch_dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size and grid[ni, nj] == 0:
                    grid[ni, nj] = 1
                    new_tips[cell_id] = (ni, nj, gen + 1, False)
                    cell_id += 1
                    break

    return grid, new_tips

def animate_simulation(P, N, M, tips, params, num_frames=400):
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
        P = steady_state_nutrient(P, M, params, nutrient_type='P')
        N = steady_state_nutrient(N, M, params, nutrient_type='N')
        M, tips = grow_tips(M, P, N, tips, params)

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
    # path1 = "results/mycelium_growth.gif"
    # path1 = "results/mycelium_growth2.gif" #Different nutrient spacing and concentration
    now = datetime.datetime.now()
    path1 = f"results/mycelium_growth_{now.strftime('%Y-%m-%d_%H-%M-%S')}_P{params['P_source_loc'][0]:.2f}_{params['P_source_loc'][1]:.2f}_{params['P_conc']:.2f}_N{params['N_source_loc'][0]:.2f}_{params['N_source_loc'][1]:.2f}_{params['N_conc']:.2f}.gif"
    ani.save(path1, writer=PillowWriter(fps=20))
    print(f"Animation saved to {path1}")

    # Create subplot of beginning, middle, end
    fig_snap, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig_snap.suptitle(f"P={params['P_conc']:.2f}, N={params['N_conc']:.2f}")
    for ax, idx in zip(axs, capture_frames):
        ax.imshow(snapshots[idx])
        ax.set_title(f"Frame {idx}")
        ax.axis('off')

    # Save subplot
    path2 = f"results/mycelium_snapshots_{now.strftime('%Y-%m-%d_%H-%M-%S')}_P{params['P_source_loc'][0]:.2f}_{params['P_source_loc'][1]:.2f}_{params['P_conc']:.2f}_N{params['N_source_loc'][0]:.2f}_{params['N_source_loc'][1]:.2f}_{params['N_conc']:.2f}.png"
    fig_snap.savefig(path2)
    print(f"Snapshots saved to {path2}")
    plt.close(fig)
    plt.close(fig_snap)

if __name__ == "__main__":
    P, N, M, tips = initialise_grids(params["grid_size"])
    animate_simulation(P, N, M, tips, params, num_frames=400)

