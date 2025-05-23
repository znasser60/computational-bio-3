import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt

params = {
    "sweeps": {
        "S1": "data/05-23_13-18_singleorganism.pickle",
        "S1_S2": "data/05-23_14-25.pickle",
        "S1_S2_MUT": "data/05-23_14-30.pickle"
    }
}

def load_data(sweeps):
    dict = {}

    # import data for all simulation scenarios
    for scenario, filename in sweeps.items():

        # skip undefined scenarios
        if filename != None: 

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                dict[scenario] = data

    return dict

def get_biomass_mean(runs, specied_id = 1):
    times = np.empty([402, len(runs)]) # 402: amount of frames

    for ii in range(len(runs)):
        time = np.ndarray.flatten(np.matrix(runs[ii]['time'])[:,specied_id])
        times[:,ii] = time

    return [np.mean(times, axis=1), np.std(times, axis=1)]

def plot_biomass(data, two_species = False, filename_ext = "", title = "Title"):
    filename = f"results/biomass_vs_time_{filename_ext}.png"

    params = data['params']
    runs = len(data['runs'])

    grid_size = params["grid_size"]
    total_area = grid_size**2

    plt.figure()

    # plot biomass curves for each species for all curves
    for ii in range(runs):
        time = data['runs'][ii]['time']
        matrix = np.matrix(time)

        if not two_species:
            frame, biomass = matrix[:, 0], matrix[:, 1]
            plt.plot(frame, biomass / total_area, color="lightgrey", label="Run")
        else:
            frame, biomass_S1, biomass_S2 = matrix[:, 0], matrix[:, 1], matrix[:, 3]
            plt.plot(frame, biomass_S1 / total_area, color="red", alpha=0.1, label="S1")
            plt.plot(frame, biomass_S2 / total_area, color="blue", alpha=0.1, label="S2")

    # plot average biomass per species
    if not two_species:
        biomass_mean, biomass_sd = get_biomass_mean(data['runs'])
        plt.plot(range(len(biomass_mean)), biomass_mean / total_area, label="Average")
    else:
        biomass_S1_mean, biomass_S1_sd = get_biomass_mean(data['runs'], 1)
        biomass_S2_mean, biomass_S2_sd = get_biomass_mean(data['runs'], 3)
        plt.plot(range(len(biomass_S1_mean)), biomass_S1_mean / total_area, color="red", label="S1 average")
        plt.plot(range(len(biomass_S2_mean)), biomass_S2_mean / total_area, color="blue", label="S2 average")


    # keeps only unique legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  
    plt.legend(by_label.values(), by_label.keys())
 
    # plot labels
    plt.title(title)
    plt.xlabel("t (h)")
    plt.ylabel("cells/total area")

    plt.savefig(filename)
    plt.close()

def plot(params):
    pp = pprint.PrettyPrinter(indent=4, depth=5)

    data = load_data(params['sweeps'])
    #pp.pprint(data)

    # Plot biomass vs time for S1
    plot_biomass(data['S1'], two_species = False, filename_ext = "S1", title="Growth of Mycelium network for Single Species")
    plot_biomass(data['S1_S2'], two_species = True, filename_ext = "S1_S2", title="Growth of Mycelium network for S1 and S2 (similar)")
    plot_biomass(data['S1_S2_MUT'], two_species = True, filename_ext = "S1_S2_MUT", title="Growth of Mycelium network for S1 and dS2")


if __name__ == "__main__":
    plot(params)