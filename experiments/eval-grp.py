#!/usr/bin/env python3

import sys
import os
import time
from datetime import datetime
import traceback
from collections import deque
import json

from absl import app, flags, logging

import numpy as np
import tensorflow as tf

import cv2
# import jax
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from PIL import Image
import imageio

# from flax.training import checkpoints
# from jaxrl_m.vision import encoders
# from jaxrl_m.agents import agents
# from jaxrl_m.data.text_processing import text_processors

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus, WidowXConfigs
from utils import state_to_eep, stack_obs

##############################################################################

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint_weights_path", '/home/r2d2/bridge_ws/bridge_data_v2/dgcbc_128/checkpoint_2000000', "Path to checkpoint", required=False
)
flags.DEFINE_multi_string(
    "checkpoint_config_path", '/home/r2d2/bridge_ws/bridge_data_v2/dgcbc_128/dgcbc_128_config.json', "Path to checkpoint config JSON", required=False
)
flags.DEFINE_string("goal_type", "gc", "Goal type", required=False)
flags.DEFINE_integer("im_size", 128, "Image size", required=False)
flags.DEFINE_string("video_save_path", '/home/gberseth/playground/bridge_data_v2/videos', "Path to save video")
flags.DEFINE_string(
    "goal_image_path", '/home/gberseth/playground/bridge_data_v2/video_images/115.jpg', "Path to a single goal image"
)  # not used by lc
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist(
    "goal_eep", [0.3, 0.0, 0.15], "Goal position"
)  # not used by lc
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_integer("act_exec_horizon", 1, "Action sequence length")
flags.DEFINE_bool("deterministic", True, "Whether to sample action deterministically")
flags.DEFINE_string("ip", "norris", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/C920/image_raw"}]
FIXED_STD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################

from mini_grp2 import *

def load_checkpoint(checkpoint_weights_path, checkpoint_config_path):
    import torch
    from einops import rearrange

    model = torch.load("./miniGRP.pth")

    def get_action(obs, image_goal, text_goal=""):
        obs = [obs_["image"] for obs_ in obs] # obs is a list of dicts
        obs = np.stack(obs, axis=-1)  # stack along the last dimension
        obs = rearrange(obs, 'h w c t -> h w (c t)')  # add batch dimension
        device = "cpu"
        text_goal = np.zeros((1, 1, 512), dtype=np.float32)
        _encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        _resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (64, 64))  # resize state
        action, loss = model.forward(torch.tensor(np.array([_encode_state(_resize_state(obs))])).to(device)
                        ,torch.tensor(text_goal, dtype=torch.float).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                        # ,torch.tensor(txt_goal, dtype=torch.long).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                        ,torch.tensor(np.array([_encode_state(_resize_state(image_goal["image"]))])).to(device) ## Not the correct goal image... Should mask this.
                        )
        action = action.cpu().detach().numpy()[0][:7] ## Add in the gripper close action
        _decode_action = lambda binN: (binN * action_std) + action_mean  # Undo mapping to [-1, 1]
        action_mean = np.array([0.02461858280003071,
      0.022891279309988022,
      -0.045159097760915756,
      0.0011062328703701496,
      0.0019464065553620458,
      0.001989248674362898,
      -0.02906678058207035])
        action_std = np.array([0.2755078673362732,
      0.352868914604187,
      0.3764951825141907,
      0.07468929141759872,
      0.08207540959119797,
      0.14809995889663696,
      1.3713711500167847])
        action = _decode_action(action)
        return action

    obs_horizon = 3
    text_processor = None

    return get_action, obs_horizon, text_processor


def request_goal_image(image_goal, widowx_client):
    """
    Request a new goal image from the user.
    """
    # ask for new goal
    if image_goal is None:
        print("Taking a new goal...")
        ch = "y"
    else:
        ch = input("Taking a new goal? [y/n]")
    if ch == "y":
        assert isinstance(FLAGS.goal_eep, list)
        goal_eep = [float(e) for e in FLAGS.goal_eep]
        widowx_client.move_gripper(1.0)  # open gripper

        # retry move action until success
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
            obs["image"].reshape(3, FLAGS.im_size, FLAGS.im_size).transpose(1, 2, 0)
            * 255
        ).astype(np.uint8)
    return image_goal


def request_goal_language(instruction, text_processor):
    """
    Request a new goal language from the user.
    """
    # ask for new instruction
    if instruction is None:
        ch = "y"
    else:
        ch = input("New instruction? [y/n]")
    if ch == "y":
        instruction = text_processor.encode(input("Instruction?"))
    return instruction


##############################################################################


def main(_):
    assert len(FLAGS.checkpoint_weights_path) == len(FLAGS.checkpoint_config_path)

    # policies is a dict from run_name to get_action function
    policies = {}
    for checkpoint_weights_path, checkpoint_config_path in zip(
        FLAGS.checkpoint_weights_path, FLAGS.checkpoint_config_path
    ):
        assert tf.io.gfile.exists(checkpoint_weights_path), checkpoint_weights_path
        checkpoint_num = int(checkpoint_weights_path.split("_")[-1])
        run_name = checkpoint_config_path.split("/")[-1]
        policies[f"{run_name}-{checkpoint_num}"] = load_checkpoint(
            checkpoint_weights_path, checkpoint_config_path
        )

    assert isinstance(FLAGS.initial_eep, list)
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
            image_goal = np.array(Image.open(FLAGS.goal_image_path).resize((128,128)))
            Image._show(Image.open(FLAGS.goal_image_path).resize((128,128)))
    elif FLAGS.goal_type == "lc":
        instruction = None

    # goal sampling loop
    while True:
        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        get_action, obs_horizon, text_processors = policies[policy_name]

        # show img for monitoring
        if FLAGS.show_image:
            obs = widowx_client.get_observation()
            while obs is None:
                print("Waiting for observations...")
                obs = widowx_client.get_observation()
                time.sleep(1)
            bgr_img = cv2.cvtColor(obs["full_image"], cv2.COLOR_RGB2BGR)
            cv2.imshow("img_view", bgr_img)
            cv2.waitKey(100)

        # request goal
        if FLAGS.goal_type == "gc":
            image_goal = request_goal_image(image_goal, widowx_client)
            goal_obs = {"image": image_goal}
            input("Press [Enter] to start.")
        elif FLAGS.goal_type == "lc":
            instruction = request_goal_language(None, text_processors)
            goal_obs = {"language": instruction}
            input("Press [Enter] to start.")
        else:
            raise ValueError(f"Unknown goal type: {FLAGS.goal_type}")

        # reset env
        widowx_client.reset()
        time.sleep(2.5)

        # move to initial position
        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            eep = state_to_eep(initial_eep, 0)
            widowx_client.move_gripper(1.0)  # open gripper

            # retry move action until success
            move_status = None
            while move_status != WidowXStatus.SUCCESS:
                move_status = widowx_client.move(eep, duration=1.5)

        # do rollout
        last_tstep = time.time()
        images = []
        image_goals = []  # only used when goal_type == "gc"
        t = 0
        if obs_horizon is not None:
            obs_hist = deque(maxlen=obs_horizon)
        # keep track of our own gripper state to implement sticky gripper
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        try:
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                    obs = widowx_client.get_observation() ## array of shape (1, 49152)
                    if obs is None:
                        print("WARNING retrying to get observation...")
                        continue

                    if FLAGS.show_image:
                        bgr_img = cv2.cvtColor(obs["full_image"], cv2.COLOR_RGB2BGR)
                        cv2.imshow("img_view", bgr_img)
                        cv2.waitKey(10)

                    image_obs = (
                        obs["image"]
                        .reshape(3, FLAGS.im_size, FLAGS.im_size)
                        .transpose(1, 2, 0)
                        * 255
                    ).astype(np.uint8)
                    obs = {"image": image_obs, "proprio": obs["state"]}
                    if obs_horizon is not None:
                        if len(obs_hist) == 0:
                            obs_hist.extend([obs] * obs_horizon)
                        else:
                            obs_hist.append(obs)
                        obs = obs_hist
                    # print(f"t={t}, obs={obs.shape}")
                    last_tstep = time.time()
                    actions = get_action(obs, goal_obs)

                    # actions = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0]) 
                    if len(actions.shape) == 1:
                        actions = actions[None]
                    for i in range(FLAGS.act_exec_horizon):
                        action = actions[i]
                        action += np.random.normal(0, FIXED_STD)

                        # sticky gripper logic
                        if (action[-1] < 0.5) != is_gripper_closed:
                            num_consecutive_gripper_change_actions += 1
                        else:
                            num_consecutive_gripper_change_actions = 0

                        if (
                            num_consecutive_gripper_change_actions
                            >= STICKY_GRIPPER_NUM_STEPS
                        ):
                            is_gripper_closed = not is_gripper_closed
                            num_consecutive_gripper_change_actions = 0

                        action[-1] = 0.0 if is_gripper_closed else 1.0

                        # remove degrees of freedom
                        if NO_PITCH_ROLL:
                            action[3] = 0
                            action[4] = 0
                        if NO_YAW:
                            action[5] = 0

                        # perform environment step
                        widowx_client.step_action(action, blocking=FLAGS.blocking)

                        # save image
                        images.append(image_obs)
                        if FLAGS.goal_type == "gc":
                            image_goals.append(image_goal)

                        t += 1
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.mp4",
            )
            if FLAGS.goal_type == "gc":
                video = np.concatenate(
                    [np.stack(image_goals), np.stack(images)], axis=1
                )
                imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)
            else:
                imageio.mimsave(save_path, images, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)

# d-gcbc succeeded in 1 out of about 14 trials when goal was picking up the carrot task with no-sub goals which are
# given in eval_dgcbc.py.
