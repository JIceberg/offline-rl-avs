# CQL for Autonomous Vehicles

This is a downstream of the av2-api repository.

## Create the Environment



## Downloading and Creating the Dataset

We recommend downloading the scenario dataset using [this tutorial](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data). This is a large dataset (~60 GB), and downloading it manually can take a long time. Once the scenarios is downloaded and uploaded to a `data_of_argo` directory,
you can create your training datasets.

### Offline Training Dataset

Run `python get_states.py` to grab the necessary observation data from the scenarios, which will be placed into the `data_for_simulator/train` directory.

Then, to create the offline dataset, run `python make_dataset.py`, which will create an offline_dataset.pkl file that will store all of the necessary transitions for training.

### Create Your Own Expert Dataset

You can create your own expert dataset by running `python train_online.py`, which will train the
RL agent in an online environment and then save the replay buffer's transitions as a pickle file.

## Offline Training

Offline training for the RL agent is done by running `python train_offline.py`.