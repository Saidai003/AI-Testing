import json
import time
import os
from QuantizedSplatLM import QuantizedSplatLM

# CONFIGURACIÓN DEL VOLUMEN MASIVO
LIMITE_EJEMPLOS = 1100000  # Más de 1 Millón de filas
USAR_DATASET_LOCAL = True  # Cambiar a True si tienes un archivo .jsonl local
RUTA_LOCAL_JSONL = "databricks-dolly-15k.jsonl"

def stream_training_corpus(max_rows):
    """Generador eficiente que hace streaming de datos línea por línea."""
    count = 0
    
    if USAR_DATASET_LOCAL and os.path.exists(RUTA_LOCAL_JSONL):
        print(f"[*] Leyendo stream local desde {RUTA_LOCAL_JSONL}...")
        with open(RUTA_LOCAL_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= max_rows: 
                    break
                try:
                    data = json.loads(line)
                    prompt = data.get("instruction" or "prompt", "").strip()
                    res = data.get("response", "").strip()
                    if prompt and res and len(prompt) < 200 and len(res) < 400:
                        yield prompt, res
                        count += 1
                except Exception:
                    continue
    else:
        # Modo Demostración de Escala: Genera datos sintéticos masivos dinámicamente si no hay archivo
        print("[!] No se detectó archivo local masivo. Ejecutando simulación de escala masiva en RAM...")
        conocimiento_base = [
            ("why can camels survive for long without water", "camels store fat in their humps which allows them to survive without water for long periods"),
            ("what is photosynthesis", "it is the process by which plants use sunlight to synthesize foods from carbon dioxide and water"),
            ("how do computers store data", "computers store data using binary code composed of ones and zeros written onto physical storage drives")
        ]
        for i in range(max_rows):
            prompt_base, res_base = conocimiento_base[i % len(conocimiento_base)]
            # Añadimos ligeras variaciones sintácticas para simular variabilidad de un corpus gigante
            yield f"{prompt_base} id {i}", f"{res_base} ref {i}"

if __name__ == "__main__":
    print("=== SYSTEM: QUANTIZED GAUSSIAN SPLAT LM (HIGH-SCALE OPTIMIZED) ===")
    
    # Instanciamos el modelo con 512 dimensiones para evitar colisiones en alta densidad
    model = QuantizedSplatLM(dimensions=512, default_variance=2.0)
    
    print(f"[*] Inicializando flujo de datos para {LIMITE_EJEMPLOS} ejemplos...")
    corpus_stream = stream_training_corpus(max_rows=LIMITE_EJEMPLOS)
    
    start_time = time.time()
    model.train_batch(corpus_stream)
    
    print("-" * 60)
    print(f"[✓] Entrenamiento masivo completado en {time.time() - start_time:.2f} segundos.")
    print("-" * 60)
    
    # Test rápido de inferencia interna con el nuevo heatmap matricial
    print("[*] Ejecutando test de sanidad matricial...")
    test_query = "why can camels survive without water"
    print(f"Usuario > {test_query}")
    print(f"IA Splat > {model.generate(test_query)}")
    print("-" * 60)
    
    model.guardar("splat_memory_space.pkl")