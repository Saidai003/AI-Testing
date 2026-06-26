import os
import sys

# Apagamos las advertencias de Hugging Face para mantener la consola limpia
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from LinearAI import LinearAI

def main():
    print("==========================================================")
    print("      LINEARLM - INTERFAZ DE RAZONAMIENTO SEMÁNTICO       ")
    print("==========================================================")
    print("[*] Cargando motor vectorial y base de datos... ", end="")
    sys.stdout.flush()
    
    ruta_modelo = "splat_memory_space.pkl"
    
    if not os.path.exists(ruta_modelo):
        print("\n[!] No se encontró la memoria. Ejecuta run_trainer.py primero.")
        return
        
    try:
        # Cargar el nuevo modelo con FAISS
        modelo = LinearAI.cargar(ruta_modelo)
        
        # Extraemos la información de la nueva arquitectura FAISS HNSW
        cantidad_pares = len(modelo.memory_inputs)
        resolucion = modelo.dim  # Ahora leemos la dimensión directamente de la clase
        
        print("[OK]")
        print(f"[*] Base de datos activa: {cantidad_pares} memorias indexadas (FAISS 8-bit).")
        print(f"[*] Resolución semántica: {resolucion} dimensiones por concepto.")
        print("[*] Escribe 'salir' para finalizar la conversación.")
        print("----------------------------------------------------------\n")
    except Exception as e:
        print(f"\n[!] Error al cargar el modelo: {e}")
        return

    while True:
        try:
            usuario = input("Usuario > ").strip()
            if usuario.lower() in ['salir', 'exit', 'quit']:
                print("[*] Desconectando motor semántico...")
                break
                
            if not usuario:
                continue

            # Llamada al nuevo motor semántico precalculado
            respuesta = modelo.generate(usuario)
            print(f"IA Splat > {respuesta}\n")
                
        except KeyboardInterrupt:
            print("\n[*] Interrupción detectada. Saliendo...")
            break
        except Exception as e:
            print(f"IA Splat > [Error de procesamiento: {e}]\n")

if __name__ == "__main__":
    main()