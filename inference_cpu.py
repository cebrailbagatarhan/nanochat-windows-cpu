"""
CPU'da eğitilmiş nanochat modelini kullanarak metin üretme
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from pathlib import Path
from nanochat.gpt_cpu import CPUCompatibleGPT
from nanochat.tokenizer import get_tokenizer

print("=" * 80)
print("🎭 nanochat CPU Inference")
print("=" * 80)

# Model yolu
data_dir = Path.home() / ".cache" / "nanochat" / "shakespeare"
checkpoint_path = data_dir / "best_model.pt"

if not checkpoint_path.exists():
    print(f"\n❌ Model bulunamadı: {checkpoint_path}")
    print("   Önce modeli eğitin: python train_cpu_shakespeare.py")
    exit(1)

# Model yükle
print(f"\n📂 Model yükleniyor: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

config = checkpoint['config']
model = CPUCompatibleGPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   ✅ Model yüklendi")
print(f"   📊 Val loss: {checkpoint['val_loss']:.4f}")
print(f"   📈 Eğitim step: {checkpoint['step']}")

# Tokenizer yükle
tokenizer = get_tokenizer()

def generate_text(prompt, max_tokens=200, temperature=0.8, top_k=40):
    """Metin üret"""
    print(f"\n🎯 Prompt: '{prompt}'")
    print(f"🔧 Ayarlar: max_tokens={max_tokens}, temperature={temperature}, top_k={top_k}")
    print(f"\n📝 Üretilen metin:")
    print("-" * 80)
    
    # Encode
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    with torch.no_grad():
        for i in range(max_tokens):
            # Context kırp
            context_crop = context if context.size(1) <= config.sequence_len else context[:, -config.sequence_len:]
            
            # Forward pass
            logits, _ = model(context_crop)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
    
    # Decode
    generated = tokenizer.decode(context[0].tolist())
    print(generated)
    print("-" * 80)
    return generated

# İnteraktif mod
print("\n" + "=" * 80)
print("🎮 İNTERAKTİF MOD")
print("=" * 80)
print("Komutlar:")
print("  - Metin girin ve Enter'a basın")
print("  - 'q' veya 'quit' yazarak çıkın")
print("  - 'örnek' yazarak örnek promptları görün")
print("=" * 80)

examples = [
    "ROMEO:",
    "JULIET:",
    "To be or not to be,",
    "Friends, Romans, countrymen,",
    "What light through yonder window",
]

while True:
    prompt = input("\n💬 Prompt (veya 'q' çıkış): ").strip()
    
    if prompt.lower() in ['q', 'quit', 'exit']:
        print("👋 Görüşürüz!")
        break
    
    if prompt.lower() in ['örnek', 'example', 'examples']:
        print("\n📚 Örnek promptlar:")
        for i, ex in enumerate(examples, 1):
            print(f"   {i}. {ex}")
        continue
    
    if not prompt:
        print("⚠️  Boş prompt! Lütfen bir şeyler yazın.")
        continue
    
    try:
        generate_text(prompt, max_tokens=150, temperature=0.8, top_k=40)
    except Exception as e:
        print(f"❌ Hata: {e}")
        continue

print("\n" + "=" * 80)
