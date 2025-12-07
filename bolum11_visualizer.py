#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BÖLÜM 11 V11 — ROTA METAN FİNAL SUNUM ARAYÜZÜ (ULTIMATE)
(Metinler Temizlendi, CO2 Oranı Eklendi, 90 Gün)
"""

import json
import os

# --- AYARLAR ---
WORLD_FILE = "world_config.json"
LOG_FILE = "simulation_logs.json"
OUTPUT_HTML = "index_final.html" 
LOGO_FILENAME = "logo.jpg"
LANDING_PAGE = "landing.html"

# KAYNAK KOD LİNKİ
SOURCE_CODE_LINK = "https://github.com/Abdulhamitkeskin/Biogas-Route-Optimizer" 

def load_data():
    if not os.path.exists(WORLD_FILE) or not os.path.exists(LOG_FILE):
        return None, None
    with open(WORLD_FILE, "r", encoding="utf-8") as f: world = json.load(f)
    with open(LOG_FILE, "r", encoding="utf-8") as f: logs = json.load(f)
    return world, logs

def generate_html(world, logs):
    js_world = json.dumps(world)
    js_logs = json.dumps(logs)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Rota Metan - Operasyon Paneli</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">

    <style>
        :root {{ --primary: #2e7d32; --red: #d32f2f; --bg: #f4f7f6; --text: #333; }}
        body {{ margin:0; font-family:'Open Sans', sans-serif; overflow:hidden; display:flex; flex-direction:column; height:100vh; background:var(--bg); }}

        /* SPLASH SCREEN */
        #splashScreen {{
            position:fixed; top:0; left:0; width:100%; height:100%; background:linear-gradient(135deg, #fff, #e8f5e9);
            z-index:9999; display:flex; flex-direction:column; align-items:center; justify-content:center; transition:transform 0.8s;
        }}
        #splashScreen.hidden {{ transform:translateY(-100%); }}
        .start-btn {{
            background:var(--primary); color:white; padding:15px 50px; font-size:20px; font-weight:bold; margin-top:30px;
            border:none; border-radius:50px; cursor:pointer; box-shadow:0 10px 30px rgba(46,125,50,0.4);
            transition:0.3s;
        }}
        .start-btn:hover {{ transform:scale(1.05); background:#1b5e20; }}

        /* HEADER */
        header {{
            height:70px; background:white; display:flex; align-items:center; padding:0 30px;
            justify-content:space-between; border-bottom:1px solid #ddd; z-index:100;
        }}
        .logo-area {{ display:flex; align-items:center; gap:15px; }}
        .logo-area img {{ height:55px; border-radius:8px; }} 
        /* İsim yazısı kaldırıldı, sadece logo var */
        
        .header-btns {{ display:flex; gap:15px; }}
        .nav-btn {{
            padding:10px 20px; border-radius:20px; font-weight:bold; font-size:13px; text-decoration:none;
            cursor:pointer; border:1px solid #ccc; background:#f9f9f9; color:#555; transition:0.3s; display:flex; align-items:center; gap:8px;
        }}
        .nav-btn:hover {{ background:#333; color:white; border-color:#333; }}
        .btn-code {{ background:#24292e; color:white; border-color:#24292e; }} 
        .btn-code:hover {{ background:#000; }}

        /* MAIN LAYOUT */
        .main-container {{ display:flex; flex:1; }}
        .view-panel {{ flex:1; border-right:2px solid #ddd; display:flex; flex-direction:column; position:relative; }}
        .panel-header {{
            text-align:center; padding:10px; font-weight:bold; font-family:'Montserrat'; font-size:14px; background:#fcfcfc; border-bottom:1px solid #eee;
        }}
        .h-classic {{ color:var(--red); border-top:4px solid var(--red); }}
        .h-smart {{ color:var(--primary); border-top:4px solid var(--primary); }}
        .map-container {{ flex:1; background:#eef2f3; }}

        /* STATS OVERLAY */
        .map-stats {{
            position:absolute; top:15px; right:15px; width:130px; background:rgba(255,255,255,0.95);
            padding:15px; border-radius:12px; box-shadow:0 5px 15px rgba(0,0,0,0.1); z-index:900;
        }}
        .stat-row {{ display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px; color:#666; }}
        .stat-val {{ font-weight:800; color:#000; font-size:14px; }}

        /* DASHBOARD */
        .dashboard {{
            height:130px; background:white; border-top:1px solid #ccc; padding:0 30px;
            display:flex; align-items:center; justify-content:center;
        }}
        .score-board {{ display:flex; gap:20px; width:100%; justify-content:center; }}
        .score-card {{
            flex:1; max-width:220px; background:#f8f9fa; border:1px solid #e9ecef; border-radius:12px;
            padding:10px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.02); display:flex; flex-direction:column; justify-content:center;
        }}
        .score-title {{ font-size:10px; color:#888; font-weight:bold; margin-bottom:5px; text-transform:uppercase; letter-spacing:1px; }}
        .score-val {{ font-size:20px; font-weight:800; color:#333; }}
        
        .progress-line {{ position:absolute; bottom:0; left:0; height:6px; background:var(--primary); width:0%; transition:width 0.1s; z-index:200; }}

        /* MODAL */
        .modal {{
            position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9);
            z-index:10000; display:flex; align-items:center; justify-content:center; opacity:0; pointer-events:none; transition:0.5s;
        }}
        .modal.visible {{ opacity:1; pointer-events:all; }}
        .modal-content {{
            background:white; padding:40px; border-radius:20px; width:700px; text-align:center; box-shadow:0 20px 50px rgba(0,0,0,0.5);
        }}
        .result-table {{ width:100%; margin:20px 0; border-collapse:collapse; }}
        .result-table th {{ background:#f4f4f4; padding:15px; color:#555; font-size:12px; }}
        .result-table td {{ padding:15px; border-bottom:1px solid #eee; font-size:16px; font-weight:600; }}
        .win {{ color:var(--primary); font-size:18px; }} .lose {{ color:var(--red); }}
        
        .modal-btns {{ display:flex; justify-content:center; gap:15px; margin-top:30px; }}

        /* ICONS */
        .icon-plant {{ background:#0288d1; width:22px; height:22px; border-radius:50%; border:3px solid white; box-shadow:0 3px 6px rgba(0,0,0,0.2); }}
        .icon-truck {{ background:#333; color:white; border-radius:50%; display:flex; align-items:center; justify-content:center; border:2px solid white; box-shadow:0 3px 6px rgba(0,0,0,0.3); }}
        .icon-farm-s {{ background:white; border:2px solid var(--primary); color:var(--primary); font-size:10px; padding:2px 5px; border-radius:6px; font-weight:800; }}
        .icon-farm-c {{ display:flex; align-items:center; justify-content:center; border-radius:50%; color:white; font-size:10px; border:2px solid white; }}
        .bg-red {{ background:var(--red); }} .bg-grey {{ background:#90a4ae; }}

    </style>
</head>
<body>

    <div id="splashScreen">
        <img src="{LOGO_FILENAME}" style="width:250px;" onerror="this.style.display='none'">
        <p style="color:#666; font-size:16px; margin-top:10px;">Yapay Zeka Destekli Biyogaz Lojistiği (90 Günlük Pilot)</p>
        <button class="start-btn" onclick="startSim()"><i class="fas fa-rocket"></i> BAŞLAT</button>
    </div>

    <div class="modal" id="resultModal">
        <div class="modal-content">
            <h2 style="margin:0; color:var(--primary); font-family:'Montserrat';">SİMÜLASYON TAMAMLANDI</h2>
            <p style="color:#666; font-size:14px; margin-bottom:30px;">90 Günlük Performans Raporu</p>
            
            <table class="result-table">
                <thead>
                    <tr>
                        <th style="text-align:left;">METRİK</th>
                        <th>KLASİK MODEL</th>
                        <th>ROTA METAN (AI)</th>
                        <th>SONUÇ</th>
                    </tr>
                </thead>
                <tbody id="resBody"></tbody>
            </table>

            <div class="modal-btns">
                <a href="{SOURCE_CODE_LINK}" target="_blank" class="nav-btn btn-code"><i class="fas fa-code"></i> KAYNAK KODLARI</a>
                <button class="nav-btn" onclick="closeResults()"><i class="fas fa-eye"></i> HARİTAYI İNCELE</button>
            </div>
        </div>
    </div>

    <header>
        <div class="logo-area">
            <img src="{LOGO_FILENAME}" onerror="this.src='https://via.placeholder.com/50'">
            </div>
        
        <div class="header-btns">
            <a href="{SOURCE_CODE_LINK}" target="_blank" class="nav-btn btn-code"><i class="fas fa-code"></i> KAYNAK KODLARI</a>
            <a href="{LANDING_PAGE}" class="nav-btn btn-home"><i class="fas fa-home"></i> ANA MENÜ</a>
        </div>
    </header>

    <div class="main-container">
        <div class="view-panel">
            <div class="panel-header h-classic">KLASİK YÖNTEM</div>
            <div id="mapC" class="map-container"></div>
            <div class="map-stats">
                <div class="stat-row"><span>GÜN</span><span class="stat-val" id="dayC">1</span></div>
                <div class="stat-row"><span>CO2 SALINIMI</span><span class="stat-val" style="color:var(--red)" id="co2C">0 kg</span></div>
            </div>
        </div>
        <div class="view-panel">
            <div class="panel-header h-smart">ROTA METAN (AI)</div>
            <div id="mapS" class="map-container"></div>
            <div class="map-stats">
                <div class="stat-row"><span>GÜN</span><span class="stat-val" id="dayS">1</span></div>
                <div class="stat-row"><span>CO2 SALINIMI</span><span class="stat-val" style="color:var(--primary)" id="co2S">0 kg</span></div>
            </div>
        </div>
    </div>

    <div class="progress-line" id="pBar"></div>

    <div class="dashboard">
        <div class="score-board">
            <div class="score-card">
                <div class="score-title">TOPLAM YOL (KM)</div>
                <div class="score-val">
                    <span style="color:var(--red)" id="tKmC">0</span> 
                    <span style="color:#ddd; margin:0 5px;">|</span> 
                    <span style="color:var(--primary)" id="tKmS">0</span>
                </div>
            </div>
            <div class="score-card">
                <div class="score-title">TOPLANAN ATIK (TON)</div>
                <div class="score-val">
                    <span style="color:#555" id="tTonC">0</span> 
                    <span style="color:#ddd; margin:0 5px;">|</span> 
                    <span style="color:#555" id="tTonS">0</span>
                </div>
            </div>
            
            <div class="score-card">
                <div class="score-title">ANLIK VERİMLİLİK (T/KM)</div>
                <div class="score-val">
                    <span style="color:var(--red)" id="effC">0.00</span> 
                    <span style="color:#ddd; margin:0 5px;">|</span> 
                    <span style="color:var(--primary)" id="effS">0.00</span>
                </div>
            </div>

            <div class="score-card" style="border-color:var(--primary); background:#e8f5e9;">
                <div class="score-title" style="color:var(--primary)">ARTIŞ ORANI</div>
                <div class="score-val" style="color:var(--primary); font-size:26px;" id="effDiff">%0.0</div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const data={js_world}, logs={js_logs};
        let mapC, mapS, truckC, truckS, polyC, polyS, markersC={{}}, markersS={{}};
        let logC=logs.classical.movements, logS=logs.smart.movements;
        let maxIdx=Math.max(logC.length, logS.length)-1, curIdx=0, timer=null;
        
        const SPEED = 40; 

        function init() {{
            const center=[data.tesis.lat, data.tesis.lon];
            const tiles='https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png';
            
            mapC=L.map('mapC', {{zoomControl:false}}).setView(center, 10);
            mapS=L.map('mapS', {{zoomControl:false}}).setView(center, 10);
            L.tileLayer(tiles, {{maxZoom:19}}).addTo(mapC); L.tileLayer(tiles, {{maxZoom:19}}).addTo(mapS);

            const pIcon = L.divIcon({{className:'icon-plant', iconSize:[20,20]}});
            L.marker(center, {{icon:pIcon}}).addTo(mapC).bindPopup("<b>BİYOGAZ TESİSİ</b>");
            L.marker(center, {{icon:pIcon}}).addTo(mapS).bindPopup("<b>BİYOGAZ TESİSİ</b>");

            data.ciftlikler.forEach(f=>{{
                markersC[f.id]=L.marker([f.lat, f.lon], {{icon:L.divIcon({{className:'icon-farm-c bg-red', html:'<i class="fas fa-warehouse"></i>', iconSize:[24,24]}})}}).addTo(mapC);
                markersS[f.id]=L.marker([f.lat, f.lon], {{icon:L.divIcon({{className:'icon-farm-s', html:'%80', iconSize:[40,20]}})}}).addTo(mapS);
            }});

            const tIcon=L.divIcon({{className:'icon-truck', html:'<i class="fas fa-truck"></i>', iconSize:[30,30]}});
            truckC=L.marker(center, {{icon:tIcon}}).addTo(mapC); truckS=L.marker(center, {{icon:tIcon}}).addTo(mapS);
            polyC=L.polyline([], {{color:'#d32f2f', weight:3, opacity:0.6}}).addTo(mapC);
            polyS=L.polyline([], {{color:'#2e7d32', weight:3, opacity:0.6}}).addTo(mapS);
        }}

        function update() {{
            let dC=logC[Math.min(curIdx, logC.length-1)], dS=logS[Math.min(curIdx, logS.length-1)];
            
            move(truckC, polyC, dC); move(truckS, polyS, dS);
            
            txt('dayC', dC.day); txt('dayS', dS.day);
            txt('co2C', dC.co2_kg.toFixed(0)+" kg"); txt('co2S', dS.co2_kg.toFixed(0)+" kg");
            
            txt('tKmC', dC.cumulative_km.toFixed(0)); txt('tKmS', dS.cumulative_km.toFixed(0));
            txt('tTonC', dC.cumulative_ton.toFixed(0)); txt('tTonS', dS.cumulative_ton.toFixed(0));

            // CANLI VERİMLİLİK HESABI
            let valC = dC.cumulative_km > 0 ? (dC.cumulative_ton / dC.cumulative_km) : 0;
            let valS = dS.cumulative_km > 0 ? (dS.cumulative_ton / dS.cumulative_km) : 0;
            txt('effC', valC.toFixed(2));
            txt('effS', valS.toFixed(2));

            if(valC > 0) txt('effDiff', "%" + (((valS-valC)/valC)*100).toFixed(1));

            if(dC.to_id!==data.tesis.id && markersC[dC.to_id]) markersC[dC.to_id].setIcon(L.divIcon({{className:'icon-farm-c bg-grey', html:'<i class="fas fa-check"></i>', iconSize:[24,24]}}));
            if(dS.to_id!==data.tesis.id && markersS[dS.to_id]) markersS[dS.to_id].setIcon(L.divIcon({{className:'icon-farm-s', html:'%0', iconSize:[40,20]}}));
            if(curIdx%10===0) Object.keys(markersS).forEach(k=>{{ if(k!==dS.to_id) markersS[k].setIcon(L.divIcon({{className:'icon-farm-s', html:'%'+(Math.floor(Math.random()*40)+50), iconSize:[40,20]}})) }});

            document.getElementById('pBar').style.width = ((curIdx/maxIdx)*100) + "%";
        }}

        function move(mk, pl, d) {{
            let lat, lon;
            if(d.to_id===data.tesis.id) {{ lat=data.tesis.lat; lon=data.tesis.lon; }}
            else {{ let f=data.ciftlikler.find(x=>x.id===d.to_id); if(f){{lat=f.lat; lon=f.lon;}} }}
            if(lat) {{ let p=[lat,lon]; mk.setLatLng(p); pl.addLatLng(p); }}
        }}

        function txt(id,v) {{ document.getElementById(id).innerText=v; }}

        function startSim() {{
            document.getElementById('splashScreen').classList.add('hidden');
            setTimeout(()=>{{
                timer = setInterval(()=>{{
                    if(curIdx<maxIdx) {{ curIdx++; update(); }}
                    else {{ clearInterval(timer); showFinal(); }}
                }}, SPEED);
            }}, 500);
        }}

        function showFinal() {{
            const c=logs.classical, s=logs.smart;
            const diff = (((s.efficiency - c.efficiency)/c.efficiency)*100).toFixed(1);
            
            // KARBON ORANI HESABI
            const co2Diff = (((c.total_co2 - s.total_co2) / c.total_co2) * 100).toFixed(1);

            const html = `
                <tr><td style="text-align:left">Toplam Yol</td><td>${{c.total_km.toFixed(0)}} KM</td><td class="win">${{s.total_km.toFixed(0)}} KM</td><td>-${{(c.total_km - s.total_km).toFixed(0)}} KM</td></tr>
                
                <tr>
                    <td style="text-align:left">Karbon (CO2)</td>
                    <td class="lose">${{c.total_co2.toFixed(0)}} kg</td>
                    <td class="win">${{s.total_co2.toFixed(0)}} kg</td>
                    <td class="win">-%${{co2Diff}}</td> </tr>
                
                <tr style="background:#e8f5e9; border-top:2px solid var(--primary);"><td style="text-align:left; font-weight:bold;">VERİMLİLİK</td><td>${{c.efficiency.toFixed(3)}}</td><td class="win" style="font-size:20px;">${{s.efficiency.toFixed(3)}}</td><td class="win">+%${{diff}}</td></tr>
            `;
            document.getElementById('resBody').innerHTML = html;
            document.getElementById('resultModal').classList.add('visible');
        }}

        function closeResults() {{ document.getElementById('resultModal').classList.remove('visible'); }}

        init(); update();
    </script>
</body>
</html>
    """
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(html_content)
    print(f"✅ FİNAL SUNUM SİTESİ HAZIR (CO2 Oranı Eklendi): {OUTPUT_HTML}")

if __name__ == "__main__":
    world, logs = load_data()
    if world and logs: generate_html(world, logs)