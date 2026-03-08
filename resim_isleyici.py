import cv2
import mediapipe as mp
import json
import math
import os
import glob

mp_hands = mp.solutions.hands
# İki el desteği burada aktif
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def normalize_et_advanced(landmarks):
    # Bilek (0) ve Orta Parmak Kökü (9) arasındaki mesafeye göre ölçekleme
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    ref_x, ref_y, ref_z = landmarks[9].x, landmarks[9].y, landmarks[9].z
    distance = math.sqrt((ref_x - base_x)**2 + (ref_y - base_y)**2 + (ref_z - base_z)**2) or 1
    
    return [{"x": (lm.x - base_x) / distance, "y": (lm.y - base_y) / distance, "z": (lm.z - base_z) / distance} for lm in landmarks]

veri_seti = {}
dataset_klasoru = "dataset"

print("Sıralı ve Çift El destekli işleme başlıyor (Vücut Takibi Kaldırıldı)...")

for kelime in os.listdir(dataset_klasoru):
    kelime_yolu = os.path.join(dataset_klasoru, kelime)
    if os.path.isdir(kelime_yolu):
        # Resimleri isimlerine göre (1.jpg, 2.jpg...) sıralı oku
        resimler = sorted(glob.glob(os.path.join(kelime_yolu, "*.*")))
        asamalar = []
        
        for resim_yolu in resimler:
            img = cv2.imread(resim_yolu)
            if img is None: continue
            
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                # Elleri X koordinatına göre (soldan sağa) hizala
                sirali_eller = sorted(results.multi_hand_landmarks, key=lambda el: el.landmark[0].x)
                eller_verisi = [normalize_et_advanced(hand.landmark) for hand in sirali_eller]
                asamalar.append(eller_verisi)
                print(f"BİLGİ: '{kelime}' - {os.path.basename(resim_yolu)} eklendi.")
        
        if asamalar:
            veri_seti[kelime] = asamalar

with open("veri_seti.json", "w") as f:
    json.dump(veri_seti, f, indent=4)

print("\nİşlem tamam! 'veri_seti.json' hazır.")