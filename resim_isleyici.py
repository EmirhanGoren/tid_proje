import cv2
import mediapipe as mp
import json
import os
import glob
import re
import math
import numpy as np  # YENİ: Türkçe karakterli yolları okumak için eklendi

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def parmak_durumu_al(landmarks):
    parmak_uclari = [8, 12, 16, 20]; parmak_kokleri = [6, 10, 14, 18]
    durumlar = []
    if landmarks[4].x > landmarks[3].x: durumlar.append("acik")
    else: durumlar.append("kapali")
    for uc, kok in zip(parmak_uclari, parmak_kokleri):
        if landmarks[uc].y < landmarks[kok].y: durumlar.append("acik")
        else: durumlar.append("kapali")
    return durumlar

def dogal_sirala(liste):
    return sorted(liste, key=lambda s: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)])

veri_seti = {}
dataset_klasoru = "dataset"

print("\n--- EL VE MESAFE ODAKLI VERİ SETİ OLUŞTURULUYOR ---")

for unite in os.listdir(dataset_klasoru):
    unite_yolu = os.path.join(dataset_klasoru, unite)
    if os.path.isdir(unite_yolu):
        veri_seti[unite] = {}
        print(f"\n>>> ÜNİTE: {unite}")
        
        for kelime in os.listdir(unite_yolu):
            kelime_yolu = os.path.join(unite_yolu, kelime)
            if os.path.isdir(kelime_yolu):
                resimler = dogal_sirala(glob.glob(os.path.join(kelime_yolu, "*.*")))
                kelime_asamalari = []
                
                for resim_yolu in resimler:
                    # KRİTİK DÜZELTME: Türkçe karakter içeren dosya yollarını okumak için Numpy kullanıyoruz
                    img_array = np.fromfile(resim_yolu, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if img is None: 
                        print(f"    [HATA] Dosya okunamadı (Bozuk olabilir): {os.path.basename(resim_yolu)}")
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        eller = []
                        for lms in results.multi_hand_landmarks:
                            eller.append(lms)
                        
                        # Elleri X koordinatına göre sırala
                        eller.sort(key=lambda x: x.landmark[0].x)
                        
                        asama_verisi = []
                        eller_arasi_mesafe = -1
                        
                        # Eğer iki el varsa aralarındaki mesafeyi hesapla
                        if len(eller) == 2:
                            x1, y1 = eller[0].landmark[0].x, eller[0].landmark[0].y
                            x2, y2 = eller[1].landmark[0].x, eller[1].landmark[0].y
                            eller_arasi_mesafe = round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2), 3)

                        for hand_landmarks in eller:
                            parmaklar = parmak_durumu_al(hand_landmarks.landmark)
                            konum = {"x": round(hand_landmarks.landmark[0].x, 3), "y": round(hand_landmarks.landmark[0].y, 3)}
                            
                            asama_verisi.append({
                                "parmaklar": parmaklar,
                                "konum": konum,
                                "eller_arasi_mesafe": eller_arasi_mesafe
                            })
                        
                        kelime_asamalari.append(asama_verisi)
                        print(f"    [OK] {os.path.basename(resim_yolu)} -> {len(eller)} el. Mesafe: {eller_arasi_mesafe}")
                    else:
                        print(f"    [!!] {os.path.basename(resim_yolu)} -> EL ALGILANAMADI!")
                
                if kelime_asamalari:
                    veri_seti[unite][kelime] = kelime_asamalari

with open("veri_seti.json", "w", encoding='utf-8') as f:
    json.dump(veri_seti, f, indent=4, ensure_ascii=False)

print("\nİŞLEM TAMAM! 'veri_seti.json' güncellendi.")