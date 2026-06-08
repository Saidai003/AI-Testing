import sys
from QuantizedSplatLM import QuantizedSplatLM  

print("==================================================")
print("     SPLAT LM - MATRICIAL & VECINDARIOS EN VIVO   ")
print("==================================================")

print("[*] Cargando tensores de memoria... ", end="")
try:
    model = QuantizedSplatLM.cargar("splat_memory_space.pkl")
    print("[OK]")
except FileNotFoundError:
    print("[ERROR]\nNo se encontró 'splat_memory_space.pkl'. Ejecuta primero 'run_trainer.py'.")
    sys.exit(1)

print(f"[*] Dimensiones activas del espacio latente: {model.dimensions}")
print("[*] Escribe 'salir' para cerrar la terminal.\n")

while True:
    try:
        prompt_original = input("\nUsuario > ")
        
        if prompt_original.lower().strip() in ['salir', 'exit', 'quit']:
            print("Cerrando la terminal del espacio gaussiano. ¡Hasta luego!")
            break
            
        if not prompt_original.strip():
            continue

        # Inferencia de alta velocidad basada en el mapa de calor
        respuesta = model.generate(prompt_original)
        print(f"IA Splat > {respuesta}")

    except KeyboardInterrupt:
        print("\nSesión cerrada por el usuario.")
        break
    except Exception as e:
        print(f"\n[!] Error en el bucle de inferencia: {e}")