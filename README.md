# CQL for Autonomous Vehicles

## Downloading the Dataset (Large)

We recommend downloading the dataset using [this tutorial](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data). This is a large dataset, and downloading it manually can take a long time. Once the dataset is downloaded and uploaded to a `data_of_argo` directory.

Then, run `python get_states.py` to grab the necessary observation data from the dataset, which will be placed into the `data_for_simulator/train` directory.