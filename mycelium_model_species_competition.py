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
        "V_max_P": 0.5,  # Maximum uptake rate of phosphate for species A
        "V_max_N": 0.7 ,  # Maximum uptake rate of nitrogen for species A
        # "V_max_P": 0.6,  # Maximum uptake rate of phosphate for species A
        # "V_max_N": 0.8,  # Maximum uptake rate of nitrogen for species A
        "branch_prob": 0.07,  # Probability of branching for species A
        "max_branch_depth": 5,  # Maximum branching depth for species A
        "chemotaxis_strength": 3.0,  # Strength of chemotaxis
    },
    "B": {
        # "V_max_P": 0.2,  # Maximum uptake rate of phosphate for species B
        # "V_max_N": 0.9,  # Maximum uptake rate of nitrogen for species B
        "V_max_P": 0.5,  # Maximum uptake rate of phosphate for species B
        "V_max_N": 0.7,  # Maximum uptake rate of nitrogen for species B
        "branch_prob": 0.1,  # Probability of branching for species B
        "max_branch_depth": 5,  # Maximum branching depth for species B
        "chemotaxis_strength": 3.0,  # Strength of chemotaxis
    },
}

params = {
    "grid_size": 100,  # Size of the simulation grid
    "dt": 0.2,  # Time step for the simulation
    "D_P": 0.5,  # Diffusion coefficient for phosphate
    "D_N": 0.6,  # Diffusion coefficient for nitrogen
    "K_m_P": 0.3, # Half-saturation constant for phosphate
    "K_m_N": 0.2, # Half-saturation constant for nitrogen
    "adhesion": 0.1,  # Adhesion parameter for cells
    "volume_constraint": 0.01,  # Constraint on cell volume
    "nutrient_threshold": 0.7,  # Threshold for nutrient concentration
    # # Same y 
    # "P_source_loc": (1/2, 1/4),  # Location of phosphate source as a fraction of grid size
    # "N_source_loc": (1/2, 3/4),  # Location of nitrogen source as a fraction of grid size
    # Same x 
    "P_source_loc": (1/4, 1/2),  # Location of phosphate source as a fraction of grid size
    "N_source_loc": (3/4, 1/2),  # Location of nitrogen source as a fraction of grid size
    "P_conc": 1.0,  # Initial concentration of phosphate
    "N_conc": 1.0,  # Initial concentration of nitrogen
}

def initialise_grids(grid_size):
    """
    Initialise the nutrients and biomass of the two species on the grid.
    """
    phosphate = np.zeros((grid_size, grid_size))
    nitrogen = np.zeros((grid_size, grid_size))
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
    # root1_grid[center, 0] = 1
    # root2_grid[center, grid_size-1] = 1 
    # tip1_map[1] = (center, 0, 0, True)
    # tip2_map[1] = (center, grid_size-1, 0, True)

    p_i, p_j = int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size)
    n_i, n_j = int(params["N_source_loc"][0] * grid_size), int(params["N_source_loc"][1] * grid_size)

    phosphate[p_i, p_j] = params["P_conc"]
    nitrogen[n_i, n_j] = params["N_conc"]

    return phosphate, nitrogen, root1_grid, root2_grid, tip1_map, tip2_map

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
    
    if nutrient_type == 'P':
        D = params["D_P"]
        V_max1 = species_params[species_type1]["V_max_P"]
        V_max2 = species_params[species_type2]["V_max_P"]
        K_m = params["K_m_P"]
        source = [(int(params["P_source_loc"][0] * grid_size), int(params["P_source_loc"][1] * grid_size))]
    else:
        D = params["D_N"]
        V_max1 = species_params[species_type1]["V_max_N"]
        V_max2 = species_params[species_type2]["V_max_N"]
        K_m = params["K_m_N"]
        source = [(int(params["N_source_loc"][0] * grid_size), int(params["N_source_loc"][1] * grid_size))]

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


def get_neighbors(i, j, grid_size, M2):
    """
    Get valid neighbors for a given cell position (i, j). Moore Neighborhood.
    M2 - the grid of the species that is not growing. you need to know what is occupied by it so that you don't go there.
    """
    # neighbors = []
    # for di, dj in [(1, 0), (0, -1), (0, 1), (-1, 0)]:
    #     ni, nj = i + di, j + dj
    #     if 0 <= ni < grid_size and 0 <= nj < grid_size:
    #         if M2[ni, nj] == 0:
    #             neighbors.append((ni, nj))
    #         else:
    #             for di2, dj2 in [(1, 0), (0, -1), (0, 1), (-1, 0)]:
    #                 ni2, nj2 = ni + di2, nj + dj2
    #                 if 0 <= ni2 < grid_size and 0 <= nj2 < grid_size and M2[ni2, nj2] == 0:
    #                     neighbors.append((ni2, nj2))
    #                     break


    # neighbors = [(i + di, j + dj) for di, dj in [(1, 0), (0, -1), (0, 1), (-1, 0)] if 0 <= i + di < grid_size and 0 <= j + dj < grid_size and M2[i + di, j + dj] == 0]
    # if not neighbors:
    #     for di, dj in [(1, 0), (0, -1), (0, 1), (-1, 0)]:
    #         ni, nj = i + di, j + dj
    #         if 0 <= ni < grid_size and 0 <= nj < grid_size and M2[ni, nj] == 1:
    #             for di2, dj2 in [(1, 0), (0, -1), (0, 1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
    #                 ni2, nj2 = ni + di2, nj + dj2
    #                 if 0 <= ni2 < grid_size and 0 <= nj2 < grid_size and M2[ni2, nj2] == 0 and (abs(di2) + abs(dj2)) > 2:
    #                     neighbors.append((ni2, nj2))
    #     neighbors = [(x, y) for x, y in neighbors if M2[x, y] == 0]

    """
    Get valid neighbors for a given cell position (i, j). Moore Neighborhood with up excluded due to geotropism.
    M2 - the grid of the species that is not growing. you need to know what is occupied by it so that you don't go there.

    """
    neighbors = [(i + di, j + dj) for di, dj in [(1, 0), (0, -1), (0, 1)] if 0 <= i + di < grid_size and 0 <= j + dj < grid_size and M2[i + di, j + dj] == 0]
    # figure out if there is M2 in the orthogonal neighbors and in the second level
    # sweep between (i-2, j-2) and (i+2, j+2)

    

    for di in range(-2, 2):
        for dj in range(-2, 2):
            if M2[i+di, j+dj] == 1:
                #find the closest two neighbors

    return neighbors

    return neighbors

def calculate_energy(i, j, P, N, params, species_params, species_type):
    """
    Calculate the energy for a given cell position (i, j) based on nutrient concentrations and other parameters.
    Cellular potts model energy function.
    """
    chemotaxis = species_params[species_type]["chemotaxis_strength"] * (P[i, j] + N[i, j]) # Maybe make the chemotaxis different for different species?
    adhesion = random.uniform(0, params["adhesion"])
    volume_penalty = random.uniform(0, params["volume_constraint"])
    return -chemotaxis + adhesion + volume_penalty

def grow_tips(grid1, grid2, P, N, tips, params, species_params, species_type):
    """
    Grow the tips of the mycelium based on nutrient uptake.

    grid1 - the grid where the tips are growing
    grid2 - the grid of the other species. this info needs to be known so you dont grow into a grid that's already occupied.
    """
    new_tips = {}
    cell_id = max(tips.keys()) + 1 if tips else 2
    grid_size = grid1.shape[0]

    for tid, (i, j, gen, is_main) in tips.items():
        if i == grid_size - 1:
            break
        if (P[i, j] > params["nutrient_threshold"]) or (N[i, j] > params["nutrient_threshold"]):
            continue

        neighbors = get_neighbors(i, j, grid_size, grid2)
        candidates = [pos for pos in neighbors if grid1[pos] == 0]
        if not candidates:
            continue

        # if is_main:
        #     candidates = [pos for pos in candidates if pos[0] > i] or candidates
        #     candidates = [pos for pos in candidates if pos[0] == i] or candidates

        scored = [(pos, calculate_energy(pos[0], pos[1], P, N, params, species_params, species_type)) for pos in candidates]
        scored.sort(key=lambda x: x[1])
        best = scored[0][0]
        grid1[best] = 1
        new_tips[cell_id] = (best[0], best[1], gen, is_main)
        cell_id += 1

        if np.random.rand() < species_params[species_type]["branch_prob"] and gen < species_params[species_type]["max_branch_depth"]:
            branch_dirs = [(-1, -1), (-1, 1), (0, -1), (0, 1)]
            random.shuffle(branch_dirs)
            for di, dj in branch_dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size and grid1[ni, nj] == 0:
                    grid1[ni, nj] = 1
                    new_tips[cell_id] = (ni, nj, gen + 1, False)
                    cell_id += 1
                    break

    return grid1, new_tips

def animate_simulation(P, N, M1, M2, tips1, tips2, params, species_params, species_type1, species_type2, num_frames=400):
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
        nonlocal P, N, M1, M2, tips1, tips2, snapshots

        P = steady_state_nutrient(P, M1, M2, params, species_params, species_type1, species_type2, nutrient_type='P')
        N = steady_state_nutrient(N, M1, M2, params, species_params, species_type1, species_type2, nutrient_type='N')
        M1, tips1 = grow_tips(M1, M2, P, N, tips1, params, species_params, species_type1)
        M2, tips2 = grow_tips(M2, M1, P, N, tips2, params, species_params, species_type2)

        rgb_image = np.ones((params["grid_size"], params["grid_size"], 3)) * [0.4, 0.26, 0.13]
        rgb_image[..., 0] += P
        rgb_image[..., 2] += N
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

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

    # Save animation
    now = datetime.datetime.now()
    path1 = f"species_results/mycelium_growth_competition_{now.strftime('%m-%d_%H-%M')}_P{params['P_source_loc'][0]:.2f}_{params['P_source_loc'][1]:.2f}_{params['P_conc']:.2f}_N{params['N_source_loc'][0]:.2f}_{params['N_source_loc'][1]:.2f}_{params['N_conc']:.2f}.gif"
    ani.save(path1, writer=PillowWriter(fps=20))
    print(f"Animation saved to {path1}")

    # Create subplot of beginning, middle, end
    fig_snap, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig_snap.suptitle(f"Species={species_type1}, {species_type2}, P={params['P_conc']:.2f}, N={params['N_conc']:.2f}")
    for ax, idx in zip(axs, capture_frames):
        ax.imshow(snapshots[idx])
        ax.set_title(f"Frame {idx}")
        ax.axis('off')

    # Save subplot
    path2 = f"species_results/mycelium_snapshots_competition_{now.strftime('%m-%d_%H-%M')}_P{params['P_source_loc'][0]:.2f}_{params['P_source_loc'][1]:.2f}_{params['P_conc']:.2f}_N{params['N_source_loc'][0]:.2f}_{params['N_source_loc'][1]:.2f}_{params['N_conc']:.2f}.png"
    fig_snap.savefig(path2)
    print(f"Snapshots saved to {path2}")
    plt.close(fig)
    plt.close(fig_snap)

if __name__ == "__main__":
    P, N, M1, M2, tips1, tips2 = initialise_grids(params["grid_size"])
    # species_type = "B"
    animate_simulation( P, N, M1, M2, tips1, tips2, params, species_params, "A", "B", num_frames=400)
    # animate_simulation( P, N, M1, M2, tips1, tips2, params, species_params, num_frames=400)

