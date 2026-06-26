import os
import glob
import pandas as pd

# 1. CORRECCIÓN DE IMPORTACIÓN: Llamamos directo al archivo local y a la nueva clase
from LinearAI import LinearAI

def stream_parquet_oasst(directory="."):
    patrones = [f"{directory}/*.parquet", f"{directory}/data/*.parquet"]
    archivos_parquet = []
    for p in patrones:
        archivos_parquet.extend(glob.glob(p))
        
    if not archivos_parquet:
        return

    for archivo in archivos_parquet:
        try:
            df = pd.read_parquet(archivo)
            mensajes_humanos = df[df['role'] == 'prompter'].set_index('message_id')['text'].to_dict()
            df_asistente = df[(df['role'] == 'assistant') & (df['lang'].isin(['es', 'en']))]
            
            for _, row in df_asistente.iterrows():
                parent_id = row.get('parent_id')
                respuesta = str(row.get('text', '')).strip()
                prompt = str(mensajes_humanos.get(parent_id, '')).strip()
                
                if prompt and respuesta:
                    yield prompt, respuesta
        except Exception:
            continue

def main():
    print("==========================================================")
    print("    SPLAT LM - ENTRENAMIENTO VECTORIAL SEMÁNTICO N/2       ")
    print("==========================================================")
    
    model_path = "splat_memory_space.pkl"
    
    # 2. CORRECCIÓN DE INSTANCIA: Usamos el nuevo nombre
    modelo = LinearAI()
    
    print("[*] Extrayendo dataset...")
    generador = stream_parquet_oasst(".")
    
    # Extraemos 80,000 pares para el entrenamiento
    pares = []
    try:
        for _ in range(80000):
            pares.append(next(generador))
    except StopIteration:
        pass

    modelo.train_batch(pares)
    modelo.guardar(model_path)
    print("\n[*] ¡Base de datos semántica indexada y guardada con éxito!")

if __name__ == "__main__":
    main()