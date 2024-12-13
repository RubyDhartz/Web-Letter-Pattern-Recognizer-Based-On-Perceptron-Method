import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import json

class PerceptronLetterRecognition:
    def __init__(self, alpha=0.1, threshold=0.9, max_epochs=100):
        """
        Inisialisasi kelas Perceptron untuk pengenalan pola huruf
        """
        self.weights = None
        self.alpha = alpha
        self.threshold = threshold
        self.max_epochs = max_epochs
        self.database_file = 'letter_database.json'
        self.model_file = 'perceptron_model.json'
        self.letter_database = self.load_database()
        self.trained_letters = []
        self.load_model()

    def load_database(self):
        """
        Memuat database pola huruf dari file JSON
        """
        if os.path.exists(self.database_file):
            with open(self.database_file, 'r') as f:
                return json.load(f)
        return {}

    def save_database(self):
        """
        Menyimpan database pola huruf ke file JSON
        """
        with open(self.database_file, 'w') as f:
            json.dump(self.letter_database, f)

    def load_model(self):
        """
        Memuat bobot model yang tersimpan
        """
        if os.path.exists(self.model_file):
            with open(self.model_file, 'r') as f:
                model_data = json.load(f)
                self.weights = np.array(model_data['weights'])
                self.alpha = model_data['alpha']
                self.threshold = model_data['threshold']

    def save_model(self):
        """
        Menyimpan bobot model ke file JSON
        """
        model_data = {
            'weights': self.weights.tolist(),
            'alpha': self.alpha,
            'threshold': self.threshold
        }
        with open(self.model_file, 'w') as f:
            json.dump(model_data, f)

    def add_letter_pattern(self, letter, pattern):
        """
        Menambahkan pola huruf baru ke database
        """
        if len(pattern) != 25:
            raise ValueError("Pola huruf harus berukuran 5x5 (25 elemen)")
        
        pattern = [1 if x > 0 else 0 for x in pattern]
        
        if letter not in self.letter_database:
            self.letter_database[letter] = []
        
        self.letter_database[letter].append(pattern)
        self.save_database()
        return f"Pola huruf {letter} berhasil ditambahkan."

    def delete_letter_pattern(self, letter, pattern_index=None):
        """
        Menghapus pola huruf dari database
        """
        if letter not in self.letter_database:
            return {'error': f'Huruf {letter} tidak ditemukan dalam database'}
        
        if pattern_index is None:
            # Hapus semua pola untuk huruf tersebut
            del self.letter_database[letter]
        else:
            # Hapus pola spesifik berdasarkan index
            try:
                del self.letter_database[letter][pattern_index]
                
                # Hapus kunci huruf jika tidak ada pola lagi
                if len(self.letter_database[letter]) == 0:
                    del self.letter_database[letter]
            except IndexError:
                return {'error': 'Index pola tidak valid'}
        
        self.save_database()
        return {'message': f'Pola huruf {letter} berhasil dihapus'}

    def clear_database(self):
        """
        Menghapus seluruh database pola huruf
        """
        self.letter_database = {}
        self.save_database()
        return {'message': 'Database berhasil dikosongkan'}

    def train(self):
        """
        Melatih model Perceptron dengan pola huruf yang tersimpan
        """
        X = []
        y = []
        self.trained_letters = list(self.letter_database.keys())

        for letter, patterns in self.letter_database.items():
            for pattern in patterns:
                X.append(pattern)
                y.append(letter)
        
        X = np.array(X)
        
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
        
        training_log = []
        for epoch in range(self.max_epochs):
            total_error = 0
            for i in range(len(X)):
                y_pred = np.dot(X[i], self.weights)
                activation = 1 if y_pred >= self.threshold else 0
                
                expected = 1
                error = expected - activation
                total_error += abs(error)
                
                self.weights += self.alpha * error * X[i]
            
            training_log.append({
                'epoch': epoch + 1,
                'total_error': total_error
            })
            
            if total_error == 0:
                break
        
        self.save_model()
        return training_log

    def predict(self, pattern):
        """
        Memprediksi huruf dari pola input dengan validasi yang lebih ketat
        """
        # Validasi pola input
        if len(pattern) != 25:
            raise ValueError("Pola huruf harus berukuran 5x5 (25 elemen)")

        # Hitung persentase sel yang terisi
        filled_cells = sum(1 for x in pattern if x > 0)
        fill_percentage = filled_cells / 25

        # Tolak pola yang terlalu kosong atau terlalu penuh
        if fill_percentage < 0.2 or fill_percentage > 0.8:
            return {
                'prediction': "Huruf Tidak Dikenali",
                'letter': None,
                'confidence': 0
            }
    
        # Perhitungan similarity dengan metode yang lebih ketat
        similarities = {}
        for letter in self.trained_letters:
            letter_similarities = []
            for trained_pattern in self.letter_database[letter]:
                # Hitung similarity dengan metode pixel-perfect
                pixel_match = sum(p1 == p2 for p1, p2 in zip(pattern, trained_pattern))
                pixel_similarity = pixel_match / 25  # Persentase piksel yang cocok persis
                
                # Tambahkan penalti untuk setiap perbedaan piksel
                similarity_score = pixel_similarity * (1 - (sum(p1 != p2 for p1, p2 in zip(pattern, trained_pattern)) / 25))
                
                letter_similarities.append(similarity_score)
            
            # Gunakan similarity terburuk sebagai representasi letter
            similarities[letter] = min(letter_similarities) if letter_similarities else 0
    
        # Naikkan ambang batas confidence untuk validasi yang lebih ketat
        MIN_CONFIDENCE = 0.9  # Ambang batas confidence yang lebih tinggi
    
        # Cek similaritas dengan kriteria yang lebih ketat
        if similarities:
            detected_letter = max(similarities, key=similarities.get)
            confidence = similarities[detected_letter]
        
            # Hanya terima jika confidence sangat tinggi
            if confidence >= MIN_CONFIDENCE:
                return {
                    'prediction': f"Huruf Terdeteksi: {detected_letter}",
                    'letter': detected_letter,
                    'confidence': float(confidence)
                }
    
        # Jika tidak ada huruf yang memenuhi kriteria
        return {
            'prediction': "Huruf Tidak Dikenali",
            'letter': None,
            'confidence': 0
        }

    def add_letter_variations(self, letter, base_pattern):
        """
        Membuat variasi pola huruf untuk meningkatkan robustness
        """
        variations = [
            base_pattern,  # Pola asli
            # Tambahkan variasi dengan sedikit perubahan
            [p if np.random.random() > 0.1 else (1-p) for p in base_pattern]
            # Tambahkan lebih banyak variasi sesuai kebutuhan
        ]
    
        for variation in variations:
            self.add_letter_pattern(letter, variation)

# Inisialisasi Flask
app = Flask(__name__)

# Inisialisasi model global
model = PerceptronLetterRecognition()

@app.route('/')
def index():
    """
    Halaman utama aplikasi
    """
    return render_template('index.html')

@app.route('/add_pattern', methods=['POST'])
def add_pattern():
    """
    Endpoint untuk menambahkan pola huruf baru
    """
    try:
        data = request.json
        letter = data.get('letter')
        pattern = data.get('pattern')
        
        if not letter or not pattern:
            return jsonify({'error': 'Letter dan pattern harus disediakan'}), 400
        
        result = model.add_letter_pattern(letter, pattern)
        return jsonify({'message': result})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train_model():
    """
    Endpoint untuk melatih model
    """
    try:
        training_log = model.train()
        return jsonify({
            'message': 'Pelatihan model selesai',
            'training_log': training_log
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_pattern', methods=['POST'])
def delete_pattern():
    """
    Endpoint untuk menghapus pola huruf
    """
    try:
        data = request.json
        letter = data.get('letter')
        pattern_index = data.get('pattern_index')
        
        if not letter:
            return jsonify({'error': 'Huruf harus disediakan'}), 400
        
        result = model.delete_letter_pattern(letter, pattern_index)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_database', methods=['POST'])
def clear_database():
    """
    Endpoint untuk menghapus seluruh database
    """
    try:
        result = model.clear_database()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_letter():
    """
    Endpoint untuk memprediksi huruf
    """
    try:
        data = request.json
        pattern = data.get('pattern')
        
        if not pattern:
            return jsonify({'error': 'Pattern harus disediakan'}), 400
        
        result = model.predict(pattern)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_database', methods=['GET'])
def get_database():
    """
    Endpoint untuk mendapatkan database pola huruf
    """
    return jsonify(model.letter_database)

if __name__ == '__main__':
    app.run(debug=True)