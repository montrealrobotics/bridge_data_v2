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

## Our setup: WidowX on `block` (lab notes)

Notes for picking the robot work back up. Last updated 2026-10-03.

### Layout

- **Robot computer `block`** (Ubuntu 22.04) drives the WidowX 250s (`/dev/ttyDXL`) and a Logitech Brio 501 camera (`/dev/video0`). The robot stack runs in Docker (ROS 2 Humble) from `~/playground/bridge_data_robot`, which is the [montrealrobotics fork](https://github.com/montrealrobotics/bridge_data_robot) at commit `5554eb7` plus local fixes (see below). `block` reaches the internet over the Mila-Public Wi-Fi. **Do not pull new code on `block`.** The fork's newer commits are work in progress.
- **Laptop** runs the policy and sends actions to `block` over the server-client interface (edgeml, ports 5556/5557).
- **Link:** a direct ethernet cable with no DHCP, so both ends use fixed addresses:
  - `block` (`enp1s0`): `10.30.100.107/24`
  - laptop (NetworkManager "Wired connection 2"): `10.30.100.1/24`, with `ipv4.never-default yes` so Wi-Fi keeps the default route

```bash
ssh gberseth@10.30.100.107        # key auth
# fallback if IPv4 on the link is broken (link-local IPv6):
ssh gberseth@fe80::50f2:2091:e0e4:bdb1%enxa0cec8fb63e0
# laptop side lost its address?
nmcli connection up "Wired connection 2"
```

### Laptop Python environment

This uses a uv venv at `.venv` (Python 3.10, CPU-only JAX; the laptop has an AMD GPU). The robot client code comes from `bridge_data_robot/` in this folder, a clone of the fork checked out on branch `robot-block` at `5554eb7` so it matches the server. The older copy in `~/playground/bridge_data_robot` on the laptop is not used.

`requirements.txt` no longer installs as written. `constraints.txt` holds the pins and the full rebuild recipe. The main fixes are:
- `jaxlib==0.4.13` has been removed from PyPI, so install it from the JAX index with `--find-links https://storage.googleapis.com/jax-releases/jax_releases.html`.
- Pin `orbax-checkpoint==0.2.7`, `ml-dtypes==0.2.0` and `scipy<1.13`; newer versions break JAX 0.4.13.
- `jaxrl_m/` has no `__init__.py`, so install this repo with `--config-settings editable_mode=compat`.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv -r requirements.txt -c constraints.txt \
  --find-links https://storage.googleapis.com/jax-releases/jax_releases.html
uv pip install --python .venv --config-settings editable_mode=compat -e . -e bridge_data_robot/widowx_envs
uv pip install --python .venv git+https://github.com/youliangtan/edgeml.git
```

**Checkpoint:** a GCBC policy (ResNet-34 encoder, 128x128 images) is in `checkpoints/` (`checkpoint_145000/` and `gcbc_128_config.json`). It loads and predicts actions offline in about 10 ms per step on the CPU.

### Running

On `block`:
```bash
cd ~/playground/bridge_data_robot
# first start, or after changing code that is baked into the image:
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
# otherwise restart the existing container (keeps the docker-cp'd fixes, see below):
docker restart robonet_gberseth
# wait for "Started streamer 0 ... → topic blue", then:
docker compose exec robonet bash -lic "widowx_env_service --server"
```

On the laptop:
```bash
source .venv/bin/activate
# smoke test: the arm runs a short scripted sequence and a camera window opens
python bridge_data_robot/widowx_envs/widowx_envs/widowx_env_service.py --client --ip 10.30.100.107
# policy
cd experiments
python eval.py \
  --checkpoint_weights_path ../checkpoints/checkpoint_145000 \
  --checkpoint_config_path ../checkpoints/gcbc_128_config.json \
  --im_size 128 --goal_type gc --ip 10.30.100.107 --show_image --blocking
```

### Local fixes on `block` (uncommitted)

The robot code was partly ported from ROS 1 to ROS 2. These fixes were needed to get the server through `init`:

| File | Change | Why |
|---|---|---|
| `widowx_envs/widowx_envs/widowx_env_service.py` | `tf.transformations` → `tf_transformations` | `tf` is ROS 1 only; server crashed on `init` |
| `multicam_server/launch/streamer.launch.py` | declare args before nodes; remap `image_raw`/`camera_info` to `<camera_name>/…`; open `video_stream_provider` | crashed with `use_sim_time does not exist`; published on `/camera/*`; server timed out on `/blue/camera_info` |
| `usb_connector_chart.yml` | Brio (`usb-0000:04:00.4-1`) mapped to `blue` | key `Brio 501` gave an invalid topic; clients request `/blue/image_raw` |
| `widowx_envs/scripts/run.sh`, `docker-compose.yml` (pre-existing) | `realsense:=false`; `runtime: nvidia` commented out | no RealSense or NVIDIA GPU on `block` |

`.bak` copies of the originals sit next to the launch file and the chart. The first two fixes are mirrored in this folder's `bridge_data_robot/` clone. They were `docker cp`'d into the running container `robonet_gberseth` and are **not yet built into the image**. A recreated container loses them until you rebuild with `--build`.

### Status and open issues

- **Done:** network, SSH, laptop env, checkpoint loading, and the arm initializing on `block`. The patched camera stream was verified on its own (`/blue/image_raw` at 10 Hz plus `/blue/camera_info`).
- **Next:** restart the container, run the client smoke test, then run `eval.py` with the GCBC checkpoint.
- `widowx_rs.launch.py` ignores its `realsense` argument and always starts the RealSense node. That causes the harmless "No RealSense devices were found!" warning; fix it with an `IfCondition`.
- `image_publisher` publishes at its default 10 Hz and ignores `fps`. Pass `publish_rate` if a faster rate is needed.
- Rebuild the image so the fixes survive container recreation, and consider committing them to the fork.

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

First, set up the robot hardware according to our [guide](https://docs.google.com/document/d/1si-6cTElTWTgflwcZRPfgHU7-UwfCUkEztkH3ge5CGc/edit?usp=sharing). Install our WidowX robot controller stack from [this repo](https://github.com/rail-berkeley/bridge_data_robot).

There are two ways to interface a policy with the robot controller: the docker compose service method or the server-client method. Refer to the [bridge_data_robot](https://github.com/rail-berkeley/bridge_data_robot) docs for an explanation of how to set up each method. In general, we recommend the server-client method.

For the server-client method, start the server on the robot. Then run the following commands on the client. You can specify the IP of the remote server via the `--ip` flag. The default IP is `localhost` (i.e the server and client are the same machine). 

```bash
# Specify the path to the downloaded checkpoints directory
export CHECKPOINT_DIR=/path/to/checkpoint_dir

# For GCBC
python experiments/eval.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_300000 \
  --checkpoint_config_path $CHECKPOINT_DIR/gcbc_256_config.json \
  --im_size 256 --goal_type gc --show_image --blocking

# For LCBC
python experiments/eval.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_145000 \
  --checkpoint_config_path $CHECKPOINT_DIR/lcbc_256_config.json \
  --im_size 256 --goal_type lc --show_image --blocking
```

You can also specify an initial position for the end effector with the flag `--initial_eep`. Similarly, use the flag `--goal_eep` to specify the position of the end effector when taking a goal image.

To evaluate image-conditioned or language-conditioned methods with the docker compose service method, run `eval_gc.py` or `eval_lc.py` respectively in the `bridge_data_v2` docker container.

## Provided Checkpoints

Checkpoints for GCBC, LCBC, D-GCBC, GCIQL, and CRL are available [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/checkpoints/). Each checkpoint has an associated JSON file with its configuration information. The name of each checkpoint indicates whether it was trained with 128x128 images or 256x256 images.

We don't currently have a checkpoints for ACT or RT-1 available but may release them soon. 

## Environment

The dependencies for this codebase can be installed in a conda environment:

```bash
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e . 
pip install -r requirements.txt
```
For GPU:
```bash
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. 

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
