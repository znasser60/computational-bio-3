import pickle
import numpy as np
import pprint
import matplotlib.pyplot as pyplot
import datetime

params = {
    "sweeps": {
        # "S1": "2025-05-22_18-42-25.pickle",
        "S1": None,
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

def plot(params):
    pp = pprint.PrettyPrinter(indent=4, depth=3)

    data = load_data(params['sweeps'])
    pp.pprint(data)

    this_data = data['S1_S2']
    runs = len(this_data['runs'])

    for ii in range(runs):
        time = this_data['runs'][ii]['time']
        matrix = np.matrix(time)
        frame, biomass = matrix[:, 0], matrix[:, 1]

        pyplot.plot(frame, biomass)

    now = datetime.datetime.now()
    timestamp = now.strftime('%m-%d_%H-%M')
    filename = f"results/biomass_vs_time_{timestamp}.png"
    pyplot.savefig(filename)

    pyplot.show()

if __name__ == "__main__":
    plot(params)
