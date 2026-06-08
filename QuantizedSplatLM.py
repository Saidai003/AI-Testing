import numpy as np
import re
import pickle
import difflib
import collections

class QuantizedSplatLM:
    def __init__(self, dimensions=512, default_variance=2.0):
        self.dimensions = dimensions
        self.default_variance = default_variance
        
        # CODEBOOK: Compresión uniforme en 1 byte (uint8)
        self.codebook = np.linspace(-15.0, 15.0, 256, dtype=np.float32)
        
        # Matrices optimizadas por palabra
        self.space_coords = {}
        self.space_order = {}
        self.space_seq_id = {}  # <-- NUEVO: Rastreador de identidad de la frase
        self.word_anchors = {}

    def _tokenize(self, text):
        texto_limpio = re.sub(r'[^\w\s]', '', text.lower())
        return texto_limpio.split()

    def _quantize(self, float_vector):
        clipped = np.clip(float_vector, -15.0, 15.0)
        indices = np.round((clipped + 15.0) * (255.0 / 30.0))
        return indices.astype(np.uint8)

    def _dequantize(self, uint8_indices):
        return self.codebook[uint8_indices]

    def _get_or_create_anchor(self, word):
        if word not in self.word_anchors:
            raw_vector = np.random.uniform(-10, 10, self.dimensions).astype(np.float32)
            self.word_anchors[word] = self._quantize(raw_vector)
        return self._dequantize(self.word_anchors[word])

    def _calculate_prompt_centroid(self, words, is_training=False):
        if not words: 
            return np.zeros(self.dimensions, dtype=np.float32)
            
        anchors = []
        vocabulario_existente = list(self.word_anchors.keys()) if not is_training else []
        
        for w in words:
            if w in self.word_anchors:
                anchors.append(self._dequantize(self.word_anchors[w]))
            elif is_training:
                anchors.append(self._get_or_create_anchor(w))
            else:
                coincidencias = difflib.get_close_matches(w, vocabulario_existente, n=1, cutoff=0.75)
                if coincidencias:
                    anchors.append(self._dequantize(self.word_anchors[coincidencias[0]]))
                        
        if not anchors:
            return np.zeros(self.dimensions, dtype=np.float32)
            
        return np.mean(anchors, axis=0)

    def train_batch(self, dataset_generator):
        """Entrenamiento masivo vinculando cada splat a su secuencia original."""
        temp_coords = collections.defaultdict(list)
        temp_orders = collections.defaultdict(list)
        temp_seq_ids = collections.defaultdict(list)  # <-- Temporal para IDs
        
        count = 0
        for seq_id, (prompt, response) in enumerate(dataset_generator):
            p_words = self._tokenize(prompt)
            r_words = self._tokenize(response)
            if not p_words or not r_words:
                continue
                
            prompt_centroid = self._calculate_prompt_centroid(p_words, is_training=True)
            
            num_words = len(r_words)
            noise = np.random.normal(0, 0.2, (num_words, self.dimensions)).astype(np.float32)
            splat_coords_batch = prompt_centroid + noise
            quantized_coords_batch = self._quantize(splat_coords_batch)
            
            for index, word in enumerate(r_words):
                temp_coords[word].append(quantized_coords_batch[index])
                temp_orders[word].append(index)
                temp_seq_ids[word].append(seq_id)  # Guardamos qué fila generó esta palabra
                
                old_anchor = self._get_or_create_anchor(word)
                new_anchor = old_anchor + 0.02 * (splat_coords_batch[index] - old_anchor)
                self.word_anchors[word] = self._quantize(new_anchor)
            
            count += 1
            if count % 100000 == 0:
                print(f"[*] Procesados {count} ejemplos en el espacio latente...")

        print("[*] Compactando y serializando espacio latente en matrices NumPy de alta velocidad...")
        for word in list(temp_coords.keys()):
            # Conservamos un máximo de 500 instancias de contexto por palabra para proteger el almacenamiento
            self.space_coords[word] = np.array(temp_coords[word][:500], dtype=np.uint8)
            self.space_order[word] = np.array(temp_orders[word][:500], dtype=np.uint32)
            self.space_seq_id[word] = np.array(temp_seq_ids[word][:500], dtype=np.uint32)

    def generate(self, prompt, threshold=0.01):
        """Generación por aislamiento del patrón de mayor densidad global."""
        prompt_words = self._tokenize(prompt)
        if not prompt_words: 
            return "..."
        
        input_anchors = []
        for w in prompt_words:
            if w in self.word_anchors:
                vec = self._dequantize(self.word_anchors[w])
                norm = np.linalg.norm(vec)
                input_anchors.append(vec / (norm + 1e-9) if norm > 0 else vec)
                
        if not input_anchors:
            return "[Sin coincidencias semánticas en la memoria]"

        input_anchors = np.array(input_anchors, dtype=np.float32)
        
        # Estructuras para evaluar qué oración (secuencia) gana el mapa de calor
        seq_scores = collections.defaultdict(float)
        seq_words = collections.defaultdict(list)
        
        for word, coords_batch in self.space_coords.items():
            orders_batch = self.space_order[word]
            seq_ids_batch = self.space_seq_id[word]
            
            real_coords = self._dequantize(coords_batch)
            norm_coords = np.linalg.norm(real_coords, axis=1, keepdims=True)
            real_coords_norm = real_coords / (norm_coords + 1e-9)
            
            # Interferencia por producto punto matricial
            dot_product = np.dot(real_coords_norm, input_anchors.T)
            dist_sq = 2.0 - 2.0 * dot_product
            densities = np.exp(-dist_sq / 1.2)
            
            coincidencias_vecinos = np.sum(densities > 0.3, axis=1)
            factor_importancia = np.where(coincidencias_vecinos > 0, coincidencias_vecinos ** 2, 0.1)
            
            scores = np.sum(densities, axis=1) * factor_importancia
            
            valid_idxs = np.where(scores > threshold)[0]
            for idx in valid_idxs:
                s_id = seq_ids_batch[idx]
                ord_val = orders_batch[idx]
                
                # Acumulamos la energía en la secuencia correspondiente
                seq_scores[s_id] += scores[idx]
                seq_words[s_id].append((ord_val, word))
                    
        if not seq_scores: 
            return "[El mapa de calor no detectó colisiones semánticas estructurales]"
        
        # ¡AQUÍ ESTÁ LA MAGIA! Encontramos la secuencia exacta que ganó la colisión de energía
        best_seq_id = max(seq_scores, key=seq_scores.get)
        
        # Reconstruimos única y exclusivamente el patrón ganador ordenado por su posición
        winning_pattern = sorted(seq_words[best_seq_id], key=lambda x: x[0])
        
        return " ".join([w[1] for w in winning_pattern])

    def guardar(self, ruta="splat_memory_space.pkl"):
        with open(ruta, "wb") as f:
            pickle.dump(self, f)
        print(f"[✓] Estructura matricial guardada con éxito en: {ruta}")

    @staticmethod
    def cargar(ruta="splat_memory_space.pkl"):
        with open(ruta, "rb") as f:
            return pickle.load(f)