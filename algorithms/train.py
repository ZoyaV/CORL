from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict, dataclass
import os
import random
import uuid

import gym  # noqa
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from generate_pogema_trajectories import load_pogema_trajectories
from pogema_wrappers import init_vactorbased_pogema
from sequence_dataset import SequenceDataset
from dt_vector_obs import DecisionTransformer
from utils import wandb_init, wrap_env, pad_along_axis, discounted_cumsum
# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        3, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        3, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(3, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash

    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)

        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],  # noqa
            actions[:, : step + 1][:, -model.seq_len :],  # noqa
            returns[:, : step + 1][:, -model.seq_len :],  # noqa
            time_steps[:, : step + 1][:, -model.seq_len :],  # noqa
        )

        predicted_action = predicted_actions[-3:, -1]
        env_action = predicted_action.argmax(axis = 1).cpu().numpy()

        next_state, reward, done, info = env.step(env_action.tolist())
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - torch.from_numpy(np.asarray(reward)).to("cuda"))

        episode_return += np.mean(reward)
        episode_len += 1

        if np.all(done):
            break

    return episode_return, episode_len

@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))

    # data & dataloader setup
    dataset = SequenceDataset(
        config.env_name, seq_len=config.seq_len, reward_scale=config.reward_scale
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    stack_len = 3
    radius = 10
    img_size = radius * 2 + 1
    num_agents = 3

    eval_env = wrap_env(
        env=init_vactorbased_pogema(stack_len, radius, img_size, num_agents, wandb, vis = False),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale
    )
    # model & optimizer & scheduler setup
    print( eval_env.observation_space.shape)
    config.state_dim = eval_env.observation_space.shape[1]
    config.action_dim = 5

    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)
      #  actions = actions.detach()
    #    print(actions)
        actions_original = F.one_hot(actions.to(torch.int64), num_classes=5).reshape(config.batch_size, config.seq_len, 5)

        predicted_actions = model(
            states=states,
            actions=actions_original,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        actions_original = actions_original.detach()
        loss = F.cross_entropy(predicted_actions.reshape(-1,5).to(torch.float32), actions_original.reshape(-1,5).to(torch.float32))
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)

                normalized_scores = (
                    eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()
