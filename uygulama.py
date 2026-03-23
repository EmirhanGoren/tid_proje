import cv2
import mediapipe as mp
import json
import random
import time
import math

# ==========================================
# --- 1. AYARLAR VE HASSASİYET ---
# ==========================================
ZORLUK_KATSAYISI = 0.20      # Elin konumu için tolerans (Artırılırsa kolaylaşır)
POZ_SABITLEME_SURESI = 0.2   # 0.2 saniye sabit tutunca onaylar

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def parmak_durumu_al(landmarks):
    """Parmakların açık/kapalı durumunu hesaplar"""
    parmak_uclari = [8, 12, 16, 20]; parmak_kokleri = [6, 10, 14, 18]
    durumlar = []
    # Baş Parmak (X ekseni)
    if landmarks[4].x > landmarks[3].x: durumlar.append("acik")
    else: durumlar.append("kapali")
    # Diğerleri (Y ekseni)
    for uc, kok in zip(parmak_uclari, parmak_kokleri):
        if landmarks[uc].y < landmarks[kok].y: durumlar.append("acik")
        else: durumlar.append("kapali")
    return durumlar

# ==========================================
# --- 2. VERİ YÜKLEME VE ÜNİTE SEÇİMİ ---
# ==========================================
try:
    with open("veri_seti.json", "r", encoding='utf-8') as f:
        tum_veri = json.load(f)
except FileNotFoundError:
    print("HATA: veri_seti.json bulunamadı. Önce resim_isleyici çalıştırın."); exit()

uniteler = list(tum_veri.keys())
print("\n--- ÜNİTE SEÇİN ---")
for i, u in enumerate(uniteler): print(f"{i+1}. {u}")

while True:
    try:
        secim = int(input(f"\nUnite No (1-{len(uniteler)}): ")) - 1
        if 0 <= secim < len(uniteler):
            secilen_unite_adi = uniteler[secim]
            unite_kelimeleri = tum_veri[secilen_unite_adi]
            break
        else: print("Geçersiz numara.")
    except ValueError: print("Sayı girin.")

# İlk kelimeyi başlat
kelime_listesi = list(unite_kelimeleri.keys())
sorulan_kelime = random.choice(kelime_listesi)
referans_asamalar = unite_kelimeleri[sorulan_kelime]

mevcut_asama = 0
dogru_yapildi_mi = False
baslangic_zamani = None

cap = cv2.VideoCapture(0)

# ==========================================
# --- 3. ANA DÖNGÜ ---
# ==========================================
while True:
    success, img = cap.read()
    if not success: break
    
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Üst Panel
    panel_renk = (0, 255, 0) if dogru_yapildi_mi else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (w, 85), panel_renk, -1)
    
    if dogru_yapildi_mi:
        cv2.putText(img, f"TEBRIKLER! {sorulan_kelime.upper()} TAMAM.", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.putText(img, f"Soru: {sorulan_kelime} | Asama: {mevcut_asama+1}/{len(referans_asamalar)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results.multi_hand_landmarks and not dogru_yapildi_mi:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        sirali_anlik = sorted(results.multi_hand_landmarks, key=lambda el: el.landmark[0].x)
        hedef_verisi = referans_asamalar[mevcut_asama]
        
        if len(sirali_anlik) == len(hedef_verisi):
            asama_onayi = True
            
            for i in range(len(hedef_verisi)):
                anlik_parmak = parmak_durumu_al(sirali_anlik[i].landmark)
                hedef_parmak = hedef_verisi[i]["parmaklar"]
                
                ax, ay = sirali_anlik[i].landmark[0].x, sirali_anlik[i].landmark[0].y
                hx, hy = hedef_verisi[i]["konum"]["x"], hedef_verisi[i]["konum"]["y"]
                mesafe_hata = math.sqrt((ax - hx)**2 + (ay - hy)**2)
                
                # --- EKRANDA DURUM GÖSTERGESİ ---
                # Hata Puanı (Sarı)
                cv2.putText(img, f"Konum Hatasi: {mesafe_hata:.2f} / <{ZORLUK_KATSAYISI}", (int(ax*w), int(ay*h)-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Parmak Durumu (Eğer yanlışsa kırmızı yazar)
                if anlik_parmak != hedef_parmak:
                    cv2.putText(img, "PARMAK SEKLI YANLIS!", (int(ax*w), int(ay*h)-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    asama_onayi = False
                
                if mesafe_hata > ZORLUK_KATSAYISI:
                    asama_onayi = False
            
            if asama_onayi:
                if baslangic_zamani is None: baslangic_zamani = time.time()
                if (time.time() - baslangic_zamani) >= POZ_SABITLEME_SURESI:
                    if mevcut_asama < len(referans_asamalar) - 1:
                        mevcut_asama += 1; baslangic_zamani = None
                    else: dogru_yapildi_mi = True
            else:
                baslangic_zamani = None

    cv2.imshow("TID Egitim Paneli", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('n') and dogru_yapildi_mi:
        sorulan_kelime = random.choice(kelime_listesi); referans_asamalar = unite_kelimeleri[sorulan_kelime]
        mevcut_asama = 0; dogru_yapildi_mi = False; baslangic_zamani = None

cap.release()
cv2.destroyAllWindows()