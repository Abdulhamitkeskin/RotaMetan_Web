#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BÖLÜM 10 V4 — MODEL KARŞILAŞTIRMA (KARBON & ENERJİ KAYBI EKLENDİ)

Eklenen Özellikler:
1. Toplam CO2 Salınımı Raporu (kg)
2. Toplam Enerji Potansiyel Kaybı (Bekleyen atıkların çürümesi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from bolum8_rl_env_v3 import RotaMetanRLEnvV3

# --- KONFİGÜRASYON ---
MODEL_PATH = "ppo_model_backup.pth"
OUTPUT_FILE = "simulation_logs.json"
EPISODE_DAYS = 90  # 360 Günlük Simülasyon
KARBON_KATSAYISI = 0.82 
GUNLUK_CURUME_ORANI = 0.02 # Bekleyen atık günde %2 enerji kaybeder

# --- MODEL MİMARİSİ ---
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
        action = torch.argmax(action_probs).item()
        return action

# --- GÜÇLENDİRİLMİŞ KLASİK MODEL ---
def get_classical_action(env, mask):
    home_action = env.num_farms 
    kamyon_doluluk_orani = env.truck_load / env.kamyon_kapasite
    
    if kamyon_doluluk_orani >= 0.95:
        if mask[home_action] == 0.0:
            return home_action
            
    candidates = []
    current_node = env.current_node
    
    for i in range(env.num_farms):
        if mask[i] == 1.0: continue
            
        cid = env.ciftlik_ids[i]
        cap = float(env.scenario[cid]["kapasite_ton"])
        fill = env.current_fill[cid]
        ratio = fill / cap
        
        if ratio >= 0.30:
            dist = env.dist_matrix[current_node][cid]
            candidates.append((i, dist, ratio))
    
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    return home_action

# --- SİMÜLASYON FONKSİYONU ---
def run_simulation(mode="smart"):
    env = RotaMetanRLEnvV3(episode_days=EPISODE_DAYS)
    state, mask = env.reset()
    
    simulation_log = {
        "mode": mode,
        "total_km": 0.0,
        "total_ton": 0.0,
        "total_co2": 0.0,
        "total_energy_loss": 0.0, # YENİ METRİK
        "efficiency": 0.0,
        "movements": []
    }
    
    policy = None
    if mode == "smart":
        if not os.path.exists(MODEL_PATH):
            print(f"HATA: Model dosyası ({MODEL_PATH}) bulunamadı!")
            return None
        policy = ActorCritic(env.state_dim, env.action_size)
        policy.load_state_dict(torch.load(MODEL_PATH))
        policy.eval()
    
    done = False
    step_count = 0
    last_day_check = 1
    
    print(f"\n--- {mode.upper()} MODEL SİMÜLASYONU BAŞLIYOR ---")
    
    while not done:
        prev_node = env.current_node
        prev_node_name = env.tesis["name"] if prev_node == env.tesis["id"] else next((c["name"] for c in env.ciftlikler if c["id"] == prev_node), prev_node)
        
        if mode == "smart":
            action = policy.act(state, mask)
        else:
            action = get_classical_action(env, mask)
            
        next_state_mask, reward, done, info = env.step(action)
        state, mask = next_state_mask
        
        # --- ENERJİ KAYBI HESAPLAMA (GÜN SONUNDA) ---
        if env.global_day > last_day_check:
            # Gün bitti, tarlada kalan atıkların çürümesini hesaba kat
            daily_loss = 0.0
            for cid in env.ciftlik_ids:
                # O an çiftlikte ne kadar varsa %2'si değer kaybeder
                daily_loss += env.current_fill[cid] * GUNLUK_CURUME_ORANI
            
            simulation_log["total_energy_loss"] += daily_loss
            last_day_check = env.global_day
        # ----------------------------------------------

        current_node = env.current_node
        current_node_name = env.tesis["name"] if current_node == env.tesis["id"] else next((c["name"] for c in env.ciftlikler if c["id"] == current_node), current_node)
        
        km = info.get("dist_km", 0.0)
        ton = info.get("taken_ton", 0.0)
        co2 = km * KARBON_KATSAYISI
        
        simulation_log["total_km"] += km
        simulation_log["total_ton"] += ton
        simulation_log["total_co2"] += co2
        
        step_data = {
            "step": step_count,
            "day": env.global_day,
            "from_id": prev_node,
            "from_name": prev_node_name,
            "to_id": current_node,
            "to_name": current_node_name,
            "action_type": "return_home" if action == env.num_farms else "collect",
            "distance_km": round(km, 2),
            "load_ton": round(ton, 2),
            "truck_load": round(env.truck_load, 2),
            "co2_kg": round(co2, 2),
            "cumulative_ton": round(simulation_log["total_ton"], 2),
            "cumulative_km": round(simulation_log["total_km"], 2)
        }
        
        simulation_log["movements"].append(step_data)
        
        if km > 0 and step_count % 100 == 0: 
            print(f"GÜN {env.global_day} | YÜK: {ton:.1f} | CO2: {co2:.1f} kg")
        
        step_count += 1
        
    if simulation_log["total_km"] > 0:
        simulation_log["efficiency"] = simulation_log["total_ton"] / simulation_log["total_km"]
    
    print(f"--- {mode.upper()} BİTTİ. Verim: {simulation_log['efficiency']:.3f} Ton/KM ---")
    return simulation_log

if __name__ == "__main__":
    print("=== KARŞILAŞTIRMALI SİMÜLASYON (V4 - 360 GÜN - DETAYLI RAPOR) ===")
    
    classic_log = run_simulation(mode="classical")
    smart_log = run_simulation(mode="smart")
    
    full_data = {
        "classical": classic_log,
        "smart": smart_log
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Veriler '{OUTPUT_FILE}' dosyasına kaydedildi.")
    
    # --- DETAYLI SONUÇ TABLOSU ---
    print("\n" + "="*60)
    print("             SONUÇ ÖZETİ (360 GÜN)")
    print("="*60)
    
    c_eff = classic_log["efficiency"]
    s_eff = smart_log["efficiency"]
    
    print(f"{'METRİK':<25} | {'KLASİK':<12} | {'AI MODEL':<12}")
    print("-" * 60)
    print(f"{'Verim (T/KM)':<25} | {c_eff:10.3f}   | {s_eff:10.3f}")
    print(f"{'Toplam Yol (KM)':<25} | {classic_log['total_km']:10.0f}   | {smart_log['total_km']:10.0f}")
    print(f"{'Toplam Yük (Ton)':<25} | {classic_log['total_ton']:10.0f}   | {smart_log['total_ton']:10.0f}")
    print(f"{'Toplam CO2 Salınımı (kg)':<25} | {classic_log['total_co2']:10.0f}   | {smart_log['total_co2']:10.0f}")
    print(f"{'Enerji Kaybı (Atıl Potansiyel)':<25} | {classic_log['total_energy_loss']:10.0f}   | {smart_log['total_energy_loss']:10.0f}")
    print("-" * 60)
    
    diff_eff = ((s_eff - c_eff) / c_eff) * 100
    diff_co2 = ((classic_log['total_co2'] - smart_log['total_co2']) / classic_log['total_co2']) * 100
    
    print(f"🚀 VERİMLİLİK ARTIŞI: %{diff_eff:.2f}")
    print(f"🌿 CO2 AZALTIMI     : %{diff_co2:.2f}")
    print("="*60)