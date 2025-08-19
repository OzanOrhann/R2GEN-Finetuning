import json
import os
import base64
from PIL import Image
from io import BytesIO

# Dosya yolları
json_path = "data/iu_xray/annotation.json"
image_base_path = "data/iu_xray/images"
output_tsv_path = "data/iu_xray/iu_xray_test.tsv"  # test için değiştir

# JSON'u oku
with open(json_path, "r") as f:
    data = json.load(f)

# Test verilerini al
test_samples = data.get("test", [])  # varsa "test" anahtarı
if not test_samples:
    print("⚠️ Uyarı: JSON'da 'test' bölmesi bulunamadı. 'train' verileri kullanılacak.")
    test_samples = data.get("train", [])  # fallback olarak train kullan

# .tsv yazmaya başla
with open(output_tsv_path, "w") as out:
    for i, entry in enumerate(test_samples):
        image_id = entry["id"]
        caption = entry["report"].replace('\t', ' ')  # tab temizle
        images = entry["image_path"]

        for j, rel_path in enumerate(images):
            image_path = os.path.join(image_base_path, rel_path)

            # base64 encode image
            with Image.open(image_path) as img:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            uniq_id = f"200_{i+1}_{j+1}"  # örnek: 200_1_1, 200_1_2
            out.write(f"{uniq_id}\t{image_id}-{j+1}\t{caption}\t\t{base64_str}\n")

print(f"{len(test_samples)} örnekten toplam {len(test_samples)*2} satır yazıldı → {output_tsv_path}")
