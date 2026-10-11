# Jax BC/RL Implementations for BridgeData V2

This repository provides code for training on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/).

We provide implementations for the following subset of methods described in the paper:

- Goal-conditioned BC
- Goal-conditioned BC with a diffusion policy 
- Goal-condtioned IQL
- Goal-conditioned contrastive RL 
- Language-conditioned BC

The official implementations and papers for all the methods can be found here:
- [IDQL](https://github.com/philippe-eecs/IDQL) (IQL + diffusion policy) [[Hansen-Estruch et al.](https://github.com/philippe-eecs/IDQL)] and [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) [[Chi et al.](https://diffusion-policy.cs.columbia.edu/)]
- [IQL](https://github.com/ikostrikov/implicit_q_learning) [[Kostrikov et al.](https://arxiv.org/abs/2110.06169)]
- [Contrastive RL](https://chongyi-zheng.github.io/stable_contrastive_rl/) [[Zheng et al.](https://arxiv.org/abs/2306.03346), [Eysenbach et al.](https://arxiv.org/abs/2206.07568)]
- [RT-1](https://github.com/google-research/robotics_transformer) [[Brohan et al.](https://arxiv.org/abs/2212.06817)]
- [ACT](https://github.com/tonyzhaozh/act) [[Zhao et al.](https://arxiv.org/abs/2304.13705)]

Please open a GitHub issue if you encounter problems with this code. 

## Data 
The raw dataset (comprised of JPEGs, PNGs, and pkl files) can be downloaded [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/). `demos*.zip` file contains the demonstration data, and `scripted*.zip` contains the data collected with a scripted policy. For training, the raw data needs to be converted into a format that is compatible with a data loader. We offer two options:

- A custom `tf.data` loader. This data loader is implemented in `jaxrl_m/data/bridge_dataset.py` and is used by the training script in this repo. The scripts in the `data_processing` folder convert the raw data into the format required by this data loader. First, use `bridgedata_raw_to_numpy.py` to convert the raw data into NumPy files. Then, use `bridgedata_numpy_to_tfrecord.py` to convert the NumPy files into TFRecord files. 
- A [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/overview) loader. Tensorflow Datasets is a high level wrapper around `tf.data`. We offer a pre-processed TFDS version of the dataset (downsampled to 256x256) in the `tfds` folder here [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/). In the TFDS dataset, the trajectories are structured using the [RLDS](https://github.com/google-research/rlds) format. We recommend using the [Octo](https://github.com/octo-models/octo) data loader for loading the RLDS version of BridgeData. If you would like to reprocess BridgeData into RLDS (e.g to change the resolution or add keys), you can use [this repo](https://github.com/kvablack/dlimp/tree/main/rlds_converters).

## Training

To start training run the command below. Replace `METHOD` with one of `gc_bc`, `gc_ddpm_bc`, `gc_iql`, or `contrastive_rl_td`, and replace `NAME` with a name for the run. 

```
python experiments/train.py \
    --config experiments/configs/train_config.py:METHOD \
    --bridgedata_config experiments/configs/data_config.py:all \
    --name NAME
```

Training hyperparameters can be modified in `experiments/configs/data_config.py` and data parameters (e.g. subsets to include/exclude) can be modified in `experiments/configs/train_config.py`. 

## Evaluation

Policies run on an inference machine and control the robot over the network: the robot control computer runs
the robot server from [montrealrobotics/bridge_data_robot](https://github.com/montrealrobotics/bridge_data_robot)
(branch `jazzy`, see its README), and `experiments/eval.py` here connects to it as a client. The inference
machine does not need ROS; it can also be the robot control computer itself (use the default `--ip localhost`).

Set up the inference machine as described in [Environment](#environment), start the server on the
robot, then run on the inference machine (`--ip` is the robot control computer's address):

```bash
# Specify the path to the downloaded checkpoints directory
export CHECKPOINT_DIR=/path/to/checkpoint_dir

# For GCBC
python experiments/eval.py --ip <ROBOT_IP> \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_300000 \
  --checkpoint_config_path $CHECKPOINT_DIR/gcbc_256_config.json \
  --im_size 256 --goal_type gc --show_image --blocking

# For LCBC
python experiments/eval.py --ip <ROBOT_IP> \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_145000 \
  --checkpoint_config_path $CHECKPOINT_DIR/lcbc_256_config.json \
  --im_size 256 --goal_type lc --show_image --blocking
```

You can also specify an initial position for the end effector with the flag `--initial_eep`. Similarly, use
the flag `--goal_eep` to specify the position of the end effector when taking a goal image. Cameras: by
default the client uses the cameras selected on the robot side (`cameras.yaml`) or `$WIDOWX_CAMERA_TOPICS`;
`--camera_topics` overrides both.

## Provided Checkpoints

Checkpoints for GCBC, LCBC, D-GCBC, GCIQL, and CRL are available [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/checkpoints/). Each checkpoint has an associated JSON file with its configuration information. The name of each checkpoint indicates whether it was trained with 128x128 images or 256x256 images.

We don't currently have a checkpoints for ACT or RT-1 available but may release them soon. 

## Environment

This sets up an inference / training machine: Python 3.12, jax with CUDA 12 (including RTX 50-series GPUs),
TensorFlow, and the `widowx_envs` client used to talk to the robot.

### Option 1: Docker

```bash
git clone -b ubuntu_24 https://github.com/montrealrobotics/bridge_data_v2.git
cd bridge_data_v2
docker build -t robonet-bridge-data-v2 .

# run (mount your checkpoints; --network host to reach the robot; DISPLAY/X11 for --show_image)
docker run --rm -it --gpus all --network host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/checkpoints:/checkpoints:ro \
  robonet-bridge-data-v2 bash
```

To work on the code without rebuilding, mount your checkout with `-v $PWD:/home/robonet/code/bridge_data_v2`.

### Option 2: Python environment (uv)

```bash
git clone -b ubuntu_24 https://github.com/montrealrobotics/bridge_data_v2.git
```

Then create the environment with **uv**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 ~/bridge_env
source ~/bridge_env/bin/activate
uv pip install -r bridge_data_v2/requirements.txt
```

The policy code (`jaxrl_m`) is imported from the repo itself rather than installed (`pip install -e .` does
not pick it up), so put the repo on `PYTHONPATH` in every shell you run it from:

```bash
export PYTHONPATH=$PWD/bridge_data_v2:$PYTHONPATH
```

`requirements.txt` also installs the robot client (`widowx_envs`, from
[bridge_data_robot](https://github.com/montrealrobotics/bridge_data_robot) branch `jazzy`) and `edgeml`. The
client must match the robot's version, so if the robot runs a different branch, change the `widowx_envs`
line accordingly. To edit the client code itself, install a local checkout over it:
`uv pip install -e /path/to/bridge_data_robot/widowx_envs`.

The install downloads a few GB (TensorFlow, jax and CUDA libraries); with uv it takes about a minute on a
fast connection.

### Check the setup

```bash
python -c "import jax, jaxrl_m.agents; print(jax.devices())"    # should list a CudaDevice
```

## Cite

This code is based on [jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) from Dibya Ghosh.

If you use this code and/or BridgeData V2 in your work, please cite the paper with:

```
@inproceedings{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Lee, Abraham and Kim, Moo Jin and Du, Max and Zheng, Chongyi and Zhao, Tony and Hansen-Estruch, Philippe and Vuong, Quan and He, Andre and Myers, Vivek and Fang, Kuan and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2023}
}
```
