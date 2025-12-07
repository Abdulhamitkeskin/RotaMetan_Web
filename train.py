#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BÖLÜM 9 V3 — PPO TRAINING (HATA DÜZELTİLMİŞ & TON/KM EKLENMİŞ)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from bolum8_rl_env_v3 import RotaMetanRLEnvV3
import os

# --- HİPERPARAMETRELER ---
LR = 0.0003
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
UPDATE_TIMESTEP = 2000
MAX_EPISODES = 10000 

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state, mask):
        state = torch.from_numpy(state).float()
        mask = torch.from_numpy(mask).float()
        action_logits = self.actor(state)
        masked_logits = action_logits.masked_fill(mask == 1, -1e9)
        action_probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, dist.entropy()

    def evaluate(self, state, action, mask):
        action_logits = self.actor(state)
        masked_logits = action_logits.masked_fill(mask == 1, -1e9)
        action_probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # --- DÜZELTME: SQUEEZE KALDIRILDI ---
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        old_masks = torch.stack(memory.masks).detach()
        # ------------------------------------

        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals, self.masks = [], [], [], [], [], []
    def clear(self):
        del self.actions[:], self.states[:], self.logprobs[:], self.rewards[:], self.is_terminals[:], self.masks[:]

def train():
    env = RotaMetanRLEnvV3()
    ppo = PPO(env.state_dim, env.action_size)
    memory = Memory()
    
    print("=== PPO EĞİTİMİ BAŞLIYOR (TON/KM GÖSTERGELİ) ===")
    
    time_step = 0
    
    # İstatistik değişkenleri
    running_reward = 0
    running_ton = 0
    running_km = 0
    
    log_freq = 20 # Kaç episode'da bir ekrana yazsın?

    for i_episode in range(1, MAX_EPISODES+1):
        state, mask = env.reset()
        ep_reward = 0
        ep_ton = 0
        ep_km = 0
        
        while True:
            time_step += 1
            action, log_prob, _ = ppo.policy_old.act(state, mask)
            
            next_state_mask, reward, done, info = env.step(action)
            next_state, next_mask = next_state_mask
            
            memory.states.append(torch.from_numpy(state).float())
            memory.masks.append(torch.from_numpy(mask).float())
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            mask = next_mask
            ep_reward += reward
            
            # İstatistik topla
            ep_ton += info.get("taken_ton", 0)
            ep_km += info.get("dist_km", 0)
            
            if time_step % UPDATE_TIMESTEP == 0:
                ppo.update(memory)
                memory.clear()
                time_step = 0
            
            if done:
                break
        
        # Running ortalamalar
        running_reward += ep_reward
        running_ton += ep_ton
        running_km += ep_km
        
        if i_episode % log_freq == 0:
            avg_reward = running_reward / log_freq
            avg_ton = running_ton / log_freq
            avg_km = running_km / log_freq
            
            # TON/KM HESABI (Sıfıra bölünme korumalı)
            if avg_km > 0:
                efficiency = avg_ton / avg_km
            else:
                efficiency = 0.0

            print(f"Ep {i_episode:4d} | Rew: {avg_reward:6.2f} | Ton: {avg_ton:5.1f} | KM: {avg_km:5.1f} | >> Verim: {efficiency:.3f} Ton/KM <<")
            
            running_reward = 0
            running_ton = 0
            running_km = 0
            
            if i_episode % 100 == 0:
                torch.save(ppo.policy.state_dict(), f"ppo_model_backup.pth")

if __name__ == "__main__":
    train()