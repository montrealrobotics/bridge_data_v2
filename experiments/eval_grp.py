#!/usr/bin/env python3
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import time
from datetime import datetime
import traceback
from collections import deque
import json
import yaml
from pathlib import Path

from absl import app, flags, logging
import numpy as np
import cv2
from PIL import ImageShow, Image
import imageio

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus, WidowXConfigs
from utils import state_to_eep, stack_obs

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

##############################################################################

np.set_printoptions(suppress=True)
logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("checkpoint_weights_path", None, "Path to checkpoint", required=True)
flags.DEFINE_multi_string("checkpoint_config_path", None, "Path to checkpoint config JSON", required=True)
flags.DEFINE_multi_string("policy", None, "Path to policy file", required=True)
flags.DEFINE_string("goal_type", "gc", "Goal type", required=False)
flags.DEFINE_integer("im_size", 128, "Image size", required=False)
flags.DEFINE_string("video_save_path", str(PROJECT_DIR / "videos"), "Path to save video")
flags.DEFINE_string("goal_image_path", str(PROJECT_DIR / "video_images/115.jpg"),
                    "Path to a single goal image")  # not used by lc
flags.DEFINE_string("training_config", str(PROJECT_DIR / "experiments/configs/grp_config.yaml"),
                    "Path to GRP config yaml")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")  # not used by lc
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_integer("act_exec_horizon", 1, "Action sequence length")
flags.DEFINE_bool("deterministic", True, "Whether to sample action deterministically")
flags.DEFINE_string("ip", "norris", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
FIXED_STD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


def load_yaml_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_weights_path, checkpoint_config_path, cfg_path):

    config = load_yaml_config(cfg_path)

    import torch
    from einops import rearrange

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("using device:", device, torch.cuda.get_device_name(0))
    else:
        print("using device:", device)
    model_path = (FLAGS.policy[0])
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model = model.to(device).eval()
    print("model param device:", next(model.parameters()).device)

    action_mean = np.array(config["env"]["action_mean"], dtype=np.float32)
    action_std = np.array(config["env"]["action_std"], dtype=np.float32)

    resize_h = int(config["image_shape"][0])
    resize_w = int(config["image_shape"][1])

    def _encode_state(arr_f32: np.ndarray) -> np.ndarray:
        return (arr_f32 / 255.0) * 2.0 - 1.0

    def _resize_state(arr_f32: np.ndarray) -> np.ndarray:
        return cv2.resize(np.array(arr_f32, dtype=np.float32), (resize_w, resize_h))

    @torch.inference_mode()
    def get_action(obs_hist, goal_obs, text_goal=""):
        obs_frames = [o["image"] for o in obs_hist]
        obs_i = np.stack(obs_frames, axis=-1)
        obs_i = rearrange(obs_i, "h w c t -> h w (c t)")
        pose_np = np.array(obs_hist[-1]["proprio"], dtype=np.float32)
        pose_t = torch.from_numpy(pose_np[None, :]).to(device)

        goals_txt = torch.zeros((1, config['max_block_size']), dtype=torch.long, device=device)

        goal_img = goal_obs["image"] if isinstance(goal_obs, dict) else goal_obs

        obs_t = torch.from_numpy(np.array([_encode_state(_resize_state(obs_i))], dtype=np.float32)).to(device)
        goal_t = torch.from_numpy(np.array([_encode_state(_resize_state(goal_img))], dtype=np.float32)).to(device)

        action_t, loss = model.forward(obs_t, goals_txt, goal_t, pose=pose_t)

        action = action_t.detach().cpu().numpy()[0][:7]
        action = (action * action_std) + action_mean
        return action

    obs_horizon = int(config["policy"]["obs_stacking"])
    text_processor = None
    return get_action, obs_horizon, text_processor


def request_goal_image(image_goal, widowx_client):
    if image_goal is None:
        print("Taking a new goal...")
        ch = "y"
    else:
        ch = input("Taking a new goal? [y/n]")

    if ch == "y":
        goal_eep = [float(e) for e in FLAGS.goal_eep]
        widowx_client.move_gripper(1.0)

        goal_eep = state_to_eep(goal_eep, 0)
        move_status = None
        while move_status != WidowXStatus.SUCCESS:
            move_status = widowx_client.move(goal_eep, duration=1.5)

        input("Press [Enter] when ready for taking the goal image. ")

        obs = widowx_client.get_observation()
        while obs is None:
            print("WARNING retrying to get observation...")
            obs = widowx_client.get_observation()
            time.sleep(1)

        image_goal = (
            obs["image"].reshape(3, FLAGS.im_size, FLAGS.im_size).transpose(1, 2, 0) * 255
        ).astype(np.uint8)
    return image_goal


def main(_):
    assert len(FLAGS.checkpoint_weights_path) == len(FLAGS.checkpoint_config_path)

    policies = {}
    for checkpoint_weights_path, checkpoint_config_path in zip(
        FLAGS.checkpoint_weights_path, FLAGS.checkpoint_config_path
    ):
        assert Path(checkpoint_weights_path).exists(), checkpoint_weights_path

        checkpoint_num = int(str(checkpoint_weights_path).split("_")[-1])
        run_name = str(checkpoint_config_path).split("/")[-1]
        policies[f"{run_name}-{checkpoint_num}"] = load_checkpoint(
            checkpoint_weights_path, checkpoint_config_path, FLAGS.training_config
        )

    initial_eep = [float(e) for e in FLAGS.initial_eep]
    start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])

    # set up environment
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["state_state"] = list(start_state)

    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)

    # load goals
    if FLAGS.goal_type == "gc":
        image_goal = None
        if FLAGS.goal_image_path is not None:
            image_goal = np.array(Image.open(FLAGS.goal_image_path).resize((128, 128)))
            if FLAGS.show_image:
                ImageShow.show(Image.open(FLAGS.goal_image_path).resize((128, 128)))
    else:
        raise ValueError("This script currently supports goal_type == 'gc' only.")

    while True:
        if len(policies) == 1:
            policy_idx = 0
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        get_action, obs_horizon, _text_processors = policies[policy_name]

        image_goal = request_goal_image(image_goal, widowx_client)
        goal_obs = {"image": image_goal}
        input("Press [Enter] to start.")

        widowx_client.reset()
        time.sleep(2.5)

        # move to initial position
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        eep = state_to_eep(initial_eep, 0)
        widowx_client.move_gripper(1.0)

        move_status = None
        while move_status != WidowXStatus.SUCCESS:
            move_status = widowx_client.move(eep, duration=1.5)

        last_tstep = time.time()
        images = []
        image_goals = [] # only used when goal_type == "gc"
        t = 0

        obs_hist = deque(maxlen=obs_horizon) if obs_horizon is not None else None
        # keep track of our own gripper state to implement sticky gripper
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0

        try:
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                    obs = widowx_client.get_observation()
                    if obs is None:
                        print("WARNING retrying to get observation...")
                        continue

                    if FLAGS.show_image:
                        bgr_img = cv2.cvtColor(obs["full_image"], cv2.COLOR_RGB2BGR)
                        cv2.imshow("img_view", bgr_img)
                        cv2.waitKey(10)

                    image_obs = (
                        obs["image"].reshape(3, FLAGS.im_size, FLAGS.im_size).transpose(1, 2, 0) * 255
                    ).astype(np.uint8)

                    obs_d = {"image": image_obs, "proprio": obs["state"]}

                    if obs_hist is not None:
                        if len(obs_hist) == 0:
                            obs_hist.extend([obs_d] * obs_horizon)
                        else:
                            obs_hist.append(obs_d)
                        obs_in = obs_hist
                    else:
                        obs_in = [obs_d]

                    last_tstep = time.time()

                    actions = get_action(obs_in, goal_obs)

                    if actions.ndim == 1:
                        actions = actions[None, :]

                    for i in range(FLAGS.act_exec_horizon):
                        action = actions[i].copy()
                        action += np.random.normal(0, FIXED_STD)

                        if (action[-1] < 0.5) != is_gripper_closed:
                            num_consecutive_gripper_change_actions += 1
                        else:
                            num_consecutive_gripper_change_actions = 0

                        if num_consecutive_gripper_change_actions >= STICKY_GRIPPER_NUM_STEPS:
                            is_gripper_closed = not is_gripper_closed
                            num_consecutive_gripper_change_actions = 0

                        action[-1] = 0.0 if is_gripper_closed else 1.0

                        if NO_PITCH_ROLL:
                            action[3] = 0
                            action[4] = 0
                        if NO_YAW:
                            action[5] = 0

                        widowx_client.step_action(action, blocking=FLAGS.blocking)

                        images.append(image_obs)
                        image_goals.append(image_goal)
                        t += 1

        except Exception:
            print(traceback.format_exc(), file=sys.stderr)

        if len(images) == 0:
            print("No frames collected; skipping video save.")
            continue

        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.mp4",
            )

            video = np.concatenate([np.stack(image_goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)
            print("saved video:", save_path)


if __name__ == "__main__":
    app.run(main)
