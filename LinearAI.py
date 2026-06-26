import numpy as np
import pickle
import random
import time
import re
import os
import faiss
from sentence_transformers import SentenceTransformer

class LinearAI:
    def __init__(self, dim=384):
        print("[*] Desplegando IA Lineal: HNSW de 8-bits y GA Vectorial...")
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.dim = dim
        
        # 1. INDICE FAISS HNSW + CUANTIZACIÓN ESCALAR A 8 BITS
        # Usamos Producto Interno (METRIC_INNER_PRODUCT) que equivale a Similitud Coseno si los vectores están normalizados
        self.index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 32, faiss.METRIC_INNER_PRODUCT)
        
        self.memory_inputs = []       
        self.memory_outputs = []      

    def train_batch(self, dataset_generator):
        raw_inputs = []
        raw_outputs = []
        textos_vistos = set()
        
        for prompt, response in dataset_generator:
            p_limpio = prompt.strip()
            r_limpio = response.strip()
            if len(p_limpio) > 5 and len(r_limpio) > 5:
                if p_limpio.lower() not in textos_vistos:
                    textos_vistos.add(p_limpio.lower())
                    raw_inputs.append(p_limpio)
                    raw_outputs.append(r_limpio)
        
        print(f"[*] Transformando y Cuantizando {len(raw_inputs)} memorias a 8-bits...")
        embeddings = self.encoder.encode(raw_inputs, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalizamos a L2 para que FAISS Inner Product = Similitud Coseno
        faiss.normalize_L2(embeddings)
        
        # [NUEVO] Calibramos el cuantizador de 8-bits con los datos actuales
        if not self.index.is_trained:
            self.index.train(embeddings)
            
        self.index.add(embeddings)
        
        self.memory_inputs.extend(raw_inputs)
        self.memory_outputs.extend(raw_outputs)

    def generate(self, prompt, umbral_similitud=0.40, max_vecinos=4):
        if self.index.ntotal == 0:
            return "[Error: Memoria vacía]"

        prompt_embedding = self.encoder.encode([prompt], convert_to_numpy=True)
        faiss.normalize_L2(prompt_embedding)
        
        # 2. BÚSQUEDA HNSW ULTRARRÁPIDA (Sin escanear toda la matriz)
        similitudes, indices = self.index.search(prompt_embedding, max_vecinos)
        similitudes = similitudes[0]
        indices = indices[0]
        
        objetivos = []
        print("\n[FAISS] Extraídos Barrios Semánticos:")
        for idx, sim in zip(indices, similitudes):
            if sim > umbral_similitud and idx != -1:
                print(f"  -> {sim*100:.1f}% Match (ID: {idx})")
                objetivos.append({"target_sim": sim, "output": self.memory_outputs[idx]})

        if not objetivos:
            return "[Sin coincidencias semánticas en el vecindario HNSW]"

        if objetivos[0]["target_sim"] > 0.95:
            return objetivos[0]["output"]

        return self._evolucion_semantica_vectorial(objetivos)

    def _extraer_fragmentos(self, texto):
        fragmentos = [f.strip() for f in re.split(r'(?<=[.?!,;:\n])\s+', texto) if f.strip()]
        return fragmentos if fragmentos else [texto.strip()]

    def _evolucion_semantica_vectorial(self, objetivos, tamano_poblacion=100, max_generaciones=50, timeout=5.0):
        """
        3. GA ACELERADO: Ya no codifica texto en cada bucle. 
        Opera matemáticamente sobre índices y vectores pre-cacheados.
        """
        start_time = time.time()
        
        textos_objetivo = [obj["output"] for obj in objetivos]
        embeddings_objetivo = self.encoder.encode(textos_objetivo, convert_to_numpy=True)
        faiss.normalize_L2(embeddings_objetivo)
        
        fragmentos_pool = []
        for texto in textos_objetivo:
            fragmentos_pool.extend(self._extraer_fragmentos(texto))
            
        # Caché de mutación: codificamos las piezas de lego una sola vez
        pool_embeddings = self.encoder.encode(fragmentos_pool, convert_to_numpy=True)
        faiss.normalize_L2(pool_embeddings)
        
        # Población inicial: El cromosoma ahora es una simple lista de números (índices)
        poblacion = []
        for _ in range(tamano_poblacion):
            cantidad = random.randint(2, min(5, len(fragmentos_pool)))
            cromosoma = random.choices(range(len(fragmentos_pool)), k=cantidad)
            poblacion.append(cromosoma)

        mejor_cromosoma = []
        mejor_fitness = float('inf')

        for generacion in range(max_generaciones):
            if time.time() - start_time > timeout:
                break
            
            evaluaciones = []
            for crom in poblacion:
                # Matemática pura: Combinamos los vectores de los fragmentos
                if not crom:
                    continue
                vec_sum = np.sum([pool_embeddings[i] for i in crom], axis=0)
                norm = np.linalg.norm(vec_sum)
                if norm > 0:
                    vec_sum = vec_sum / norm
                
                error_total = 0
                for j, obj in enumerate(objetivos):
                    sim_lograda = np.dot(vec_sum, embeddings_objetivo[j])
                    error_total += abs(obj["target_sim"] - sim_lograda)
                
                evaluaciones.append((error_total, crom))

            evaluaciones.sort(key=lambda x: x[0])
            
            if evaluaciones[0][0] < mejor_fitness:
                mejor_fitness = evaluaciones[0][0]
                mejor_cromosoma = evaluaciones[0][1]

            # Reproducción basada en índices
            nueva_poblacion = [evaluaciones[0][1], evaluaciones[1][1]]
            
            while len(nueva_poblacion) < tamano_poblacion:
                padre1 = random.choice(evaluaciones[:tamano_poblacion//2])[1]
                padre2 = random.choice(evaluaciones[:tamano_poblacion//2])[1]
                
                min_len = min(len(padre1), len(padre2))
                if min_len > 1:
                    punto = random.randint(1, min_len - 1)
                    hijo = padre1[:punto] + padre2[punto:]
                else:
                    hijo = padre1 + padre2
                
                # Mutación de índices
                if random.random() < 0.30 and hijo:
                    idx = random.randint(0, len(hijo)-1)
                    opcion = random.random()
                    if opcion < 0.4:
                        hijo[idx] = random.randint(0, len(fragmentos_pool)-1) 
                    elif opcion < 0.7:
                        hijo.pop(idx) 
                    else:
                        hijo.insert(idx, random.randint(0, len(fragmentos_pool)-1)) 
                
                nueva_poblacion.append(hijo)
                
            poblacion = nueva_poblacion

        print(f"[GA] Matemática convergida. Fitness Error: {mejor_fitness:.4f}")
        
        # Traducción inversa final: convertir los números ganadores en texto
        respuesta_final = " ".join([fragmentos_pool[i] for i in mejor_cromosoma])
        return respuesta_final

    def guardar(self, path):
        # FAISS debe guardarse aparte por estar en C++
        faiss_path = path + ".faiss"
        faiss.write_index(self.index, faiss_path)
        
        # Guardamos el resto del cerebro en Pickle
        temp_index = self.index
        temp_encoder = self.encoder
        self.index = None
        self.encoder = None
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
        self.index = temp_index
        self.encoder = temp_encoder

    @staticmethod
    def cargar(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            
        print("[*] Reconectando subrutinas FAISS y Transformer...")
        obj.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        faiss_path = path + ".faiss"
        if os.path.exists(faiss_path):
            obj.index = faiss.read_index(faiss_path)
        else:
            print("[!] Advertencia: Archivo .faiss no encontrado.")
            
        return obj