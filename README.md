# CQL for Autonomous Vehicles

## Downloading and Creating the Dataset

We recommend downloading the dataset using [this tutorial](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data). This is a large dataset (~60 GB), and downloading it manually can take a long time. Once the dataset is downloaded and uploaded to a `data_of_argo` directory.

Then, run `python get_states.py` to grab the necessary observation data from the dataset, which will be placed into the `data_for_simulator/train` directory.

Finally, to create the offline dataset, run `python make_dataset.py`, which will create an offline_dataset.pkl file that will store all of the necessary transitions for training.