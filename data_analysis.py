import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import datetime

params = {
    "sweeps": {
        # "S1": "2025-05-22_18-42-25.pickle",
        "S1": "2025-05-22_18-42-25.pickle",
        "S1_S2": "05-22_19-32.pickle",
        "S1_S2_MUT": None
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

def get_biomass_mean(runs):
    times = np.empty([402, len(runs)]) # 402: amount of frames?

    for ii in range(len(runs)):
        time = np.ndarray.flatten(np.matrix(runs[ii]['time'])[:,1])
        times[:,ii] = time

    return np.mean(times, axis=1)

def plot_biomass_S1(data):
    params = data['params']
    runs = len(data['runs'])

    grid_size = params["grid_size"]
    total_area = grid_size**2

    biomass_mean = get_biomass_mean(data['runs'])

    for ii in range(runs):
        time = data['runs'][ii]['time']
        matrix = np.matrix(time)
        frame, biomass = matrix[:, 0], matrix[:, 1]
        plt.plot(frame, biomass / total_area, color="lightgrey")

    plt.plot(range(len(biomass_mean)), biomass_mean / total_area, label="S1 average")
    plt.legend()

    plt.xlabel("t (h)")
    plt.ylabel("cells/total area")

    plt.show()


def plot(params):
    pp = pprint.PrettyPrinter(indent=4, depth=3)

    data = load_data(params['sweeps'])
    #pp.pprint(data)

    # Plot biomass vs time for S1
    this_data = data['S1']
    plot_biomass_S1(this_data)

    now = datetime.datetime.now()
    timestamp = now.strftime('%m-%d_%H-%M')
    filename = f"results/biomass_vs_time_{timestamp}.png"
    plt.savefig(filename)

    #plt.close()

if __name__ == "__main__":
    plot(params)