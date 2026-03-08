import cv2
import mediapipe as mp
import json
import math
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

def normalize_et_advanced(landmarks):
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    ref_x, ref_y, ref_z = landmarks[9].x, landmarks[9].y, landmarks[9].z
    distance = math.sqrt((ref_x - base_x)**2 + (ref_y - base_y)**2 + (ref_z - base_z)**2) or 1
    return [{"x": (lm.x - base_x) / distance, "y": (lm.y - base_y) / distance, "z": (lm.z - base_z) / distance} for lm in landmarks]

def mesafe_hesapla(anlik, kayitli):
    return sum(math.sqrt((anlik[i]['x']-kayitli[i]['x'])**2 + (anlik[i]['y']-kayitli[i]['y'])**2 + (anlik[i]['z']-kayitli[i]['z'])**2) for i in range(21)) / 21

try:
    with open("veri_seti.json", "r") as f:
        veri_seti = json.load(f)
except:
    print("HATA: veri_seti.json bulunamadı."); exit()

sorulan_kelime = random.choice(list(veri_seti.keys()))
referans_asamalar = veri_seti[sorulan_kelime]
mevcut_asama = 0
dogru_yapildi_mi = False

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break
    
    # Perspektif karışmaması için ayna efekti (flip) kapalı!
    # img = cv2.flip(img, 1) 
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if dogru_yapildi_mi:
        cv2.rectangle(img, (0, 0), (640, 80), (0, 255, 0), -1)
        cv2.putText(img, f"TEBRIKLER! '{sorulan_kelime}' TAMAM.", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(img, f"Soru: {sorulan_kelime} | Asama: {mevcut_asama+1}/{len(referans_asamalar)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if results.multi_hand_landmarks and not dogru_yapildi_mi:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        sirali_anlik = sorted(results.multi_hand_landmarks, key=lambda el: el.landmark[0].x)
        hedef_eller = referans_asamalar[mevcut_asama]
        
        if len(sirali_anlik) == len(hedef_eller):
            hata_toplam = 0
            for i in range(len(hedef_eller)):
                anlik_norm = normalize_et_advanced(sirali_anlik[i].landmark)
                hata_toplam += mesafe_hesapla(anlik_norm, hedef_eller[i])
            
            ortalama_hata = hata_toplam / len(hedef_eller)
            
            if ortalama_hata < 0.40:
                if mevcut_asama < len(referans_asamalar) - 1:
                    mevcut_asama += 1 # Sonraki adıma geç
                else:
                    dogru_yapildi_mi = True
            else:
                cv2.putText(img, f"Hata: {ortalama_hata:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(img, f"Beklenen El: {len(hedef_eller)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("TID Duolingo (Eski Saglam Versiyon)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()