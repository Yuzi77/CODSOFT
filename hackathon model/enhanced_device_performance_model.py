import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedDevicePerformanceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'cpu_cores', 'cpu_frequency_mhz', 'ram_gb', 'storage_gb', 
            'android_version', 'gpu_score', 'architecture_score', 
            'thermal_score', 'battery_capacity', 'screen_resolution_score'
        ]
        
    def create_enhanced_dataset(self, n_samples=10000):
        """Create enhanced synthetic dataset with more complex scenarios"""
        np.random.seed(42)
        
        # Generate device specifications with more realistic distributions
        cpu_cores = np.random.choice([2, 4, 6, 8, 10, 12], n_samples, 
                                   p=[0.05, 0.35, 0.25, 0.25, 0.08, 0.02])
        
        # CPU frequency with correlation to cores
        cpu_frequency = np.zeros(n_samples)
        for i in range(n_samples):
            if cpu_cores[i] >= 8:
                cpu_frequency[i] = np.random.uniform(2000, 3200)
            elif cpu_cores[i] >= 4:
                cpu_frequency[i] = np.random.uniform(1500, 2800)
            else:
                cpu_frequency[i] = np.random.uniform(1000, 2000)
        
        # RAM with correlation to device tier
        ram_gb = np.random.choice([1, 2, 3, 4, 6, 8, 12, 16, 18], n_samples, 
                                p=[0.02, 0.08, 0.15, 0.25, 0.2, 0.15, 0.1, 0.04, 0.01])
        
        # Storage with correlation to RAM
        storage_gb = np.zeros(n_samples)
        for i in range(n_samples):
            if ram_gb[i] >= 8:
                storage_gb[i] = np.random.choice([128, 256, 512, 1024], p=[0.3, 0.4, 0.25, 0.05])
            elif ram_gb[i] >= 4:
                storage_gb[i] = np.random.choice([64, 128, 256, 512], p=[0.2, 0.5, 0.25, 0.05])
            else:
                storage_gb[i] = np.random.choice([16, 32, 64, 128], p=[0.1, 0.3, 0.5, 0.1])
        
        # Android version with realistic distribution
        android_version = np.random.choice([7, 8, 9, 10, 11, 12, 13, 14], n_samples, 
                                         p=[0.02, 0.05, 0.08, 0.15, 0.2, 0.25, 0.2, 0.05])
        
        # GPU score correlated with device tier
        gpu_score = np.zeros(n_samples)
        for i in range(n_samples):
            if cpu_cores[i] >= 8 and ram_gb[i] >= 8:
                gpu_score[i] = np.random.uniform(600, 1200)  # High-end
            elif cpu_cores[i] >= 4 and ram_gb[i] >= 4:
                gpu_score[i] = np.random.uniform(300, 700)   # Mid-range
            else:
                gpu_score[i] = np.random.uniform(100, 400)   # Low-end
        
        # Architecture score (ARM64 dominance in newer devices)
        architecture_score = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
        
        # Thermal management score (affects sustained performance)
        thermal_score = np.random.uniform(0.5, 1.0, n_samples)
        
        # Battery capacity (affects overall device capability)
        battery_capacity = np.random.uniform(2000, 6000, n_samples)
        
        # Screen resolution score (affects GPU load)
        screen_resolution_score = np.random.choice([1, 2, 3, 4], n_samples, 
                                                 p=[0.1, 0.3, 0.5, 0.1])  # 1=HD, 2=FHD, 3=QHD, 4=4K
        
        # Calculate enhanced performance score and category
        performance_categories = []
        concurrent_tasks = []
        performance_scores = []
        
        for i in range(n_samples):
            # Enhanced scoring system
            perf_score = self.calculate_enhanced_performance_score({
                'cpu_cores': cpu_cores[i],
                'cpu_frequency_mhz': cpu_frequency[i],
                'ram_gb': ram_gb[i],
                'storage_gb': storage_gb[i],
                'android_version': android_version[i],
                'gpu_score': gpu_score[i],
                'architecture_score': architecture_score[i],
                'thermal_score': thermal_score[i],
                'battery_capacity': battery_capacity[i],
                'screen_resolution_score': screen_resolution_score[i]
            })
            
            # Categorize based on enhanced score
            if perf_score >= 80:
                category = 'HIGH'
                tasks = min(15, 8 + int(cpu_cores[i] * 0.6) + int(ram_gb[i] * 0.3))
            elif perf_score >= 60:
                category = 'MID'
                tasks = min(10, 4 + int(cpu_cores[i] * 0.4) + int(ram_gb[i] * 0.2))
            else:
                category = 'LOW'
                tasks = min(6, 2 + int(cpu_cores[i] * 0.2) + int(ram_gb[i] * 0.1))
            
            performance_categories.append(category)
            concurrent_tasks.append(tasks)
            performance_scores.append(perf_score)
        
        # Create enhanced DataFrame
        data = {
            'cpu_cores': cpu_cores,
            'cpu_frequency_mhz': cpu_frequency,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'android_version': android_version,
            'gpu_score': gpu_score,
            'architecture_score': architecture_score,
            'thermal_score': thermal_score,
            'battery_capacity': battery_capacity,
            'screen_resolution_score': screen_resolution_score,
            'performance_category': performance_categories,
            'concurrent_tasks': concurrent_tasks,
            'performance_score': performance_scores
        }
        
        return pd.DataFrame(data)
    
    def calculate_enhanced_performance_score(self, device_specs):
        """Enhanced performance scoring system"""
        score = 0
        
        # CPU Performance (35% weight)
        cpu_score = 0
        cores = device_specs['cpu_cores']
        freq = device_specs['cpu_frequency_mhz']
        
        # Core count scoring with diminishing returns
        if cores >= 12:
            cpu_score += 20
        elif cores >= 8:
            cpu_score += 18
        elif cores >= 6:
            cpu_score += 15
        elif cores >= 4:
            cpu_score += 12
        else:
            cpu_score += 8
        
        # Frequency scoring
        if freq >= 3000:
            cpu_score += 15
        elif freq >= 2500:
            cpu_score += 13
        elif freq >= 2000:
            cpu_score += 11
        elif freq >= 1500:
            cpu_score += 9
        else:
            cpu_score += 6
        
        # RAM Performance (25% weight)
        ram_score = 0
        ram = device_specs['ram_gb']
        
        if ram >= 16:
            ram_score += 25
        elif ram >= 12:
            ram_score += 23
        elif ram >= 8:
            ram_score += 20
        elif ram >= 6:
            ram_score += 16
        elif ram >= 4:
            ram_score += 12
        elif ram >= 2:
            ram_score += 8
        else:
            ram_score += 4
        
        # Storage Performance (10% weight)
        storage_score = 0
        storage = device_specs['storage_gb']
        
        if storage >= 512:
            storage_score += 10
        elif storage >= 256:
            storage_score += 9
        elif storage >= 128:
            storage_score += 8
        elif storage >= 64:
            storage_score += 6
        elif storage >= 32:
            storage_score += 4
        else:
            storage_score += 2
        
        # GPU Performance (15% weight)
        gpu_score = min(15, device_specs['gpu_score'] / 80)
        
        # Android Version (5% weight)
        android_score = 0
        android_ver = device_specs['android_version']
        
        if android_ver >= 13:
            android_score += 5
        elif android_ver >= 11:
            android_score += 4
        elif android_ver >= 9:
            android_score += 3
        elif android_ver >= 8:
            android_score += 2
        else:
            android_score += 1
        
        # Architecture bonus (3% weight)
        arch_score = device_specs['architecture_score'] * 3
        
        # Thermal management (4% weight)
        thermal_score = device_specs['thermal_score'] * 4
        
        # Battery impact (2% weight)
        battery_score = min(2, device_specs['battery_capacity'] / 3000)
        
        # Screen resolution penalty (1% weight)
        resolution_penalty = device_specs['screen_resolution_score'] * 0.5
        
        # Calculate total score
        total_score = (cpu_score + ram_score + storage_score + gpu_score + 
                      android_score + arch_score + thermal_score + 
                      battery_score - resolution_penalty)
        
        return min(100, max(0, total_score))
    
    def create_advanced_tensorflow_model(self, data):
        """Create advanced TensorFlow model with better architecture"""
        # Prepare data
        X = data[self.feature_names]
        y = data['performance_category']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=3)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, 
                                                           test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create advanced model with batch normalization and regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=200,
            batch_size=64,
            validation_data=(X_test_scaled, y_test),
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        self.tf_model = model
        return model, history
    
    def predict_device_performance(self, device_specs):
        """Enhanced prediction with confidence scoring"""
        # Convert input to numpy array
        features = np.array([
            device_specs['cpu_cores'],
            device_specs['cpu_frequency_mhz'],
            device_specs['ram_gb'],
            device_specs['storage_gb'],
            device_specs['android_version'],
            device_specs['gpu_score'],
            device_specs['architecture_score'],
            device_specs.get('thermal_score', 0.8),
            device_specs.get('battery_capacity', 4000),
            device_specs.get('screen_resolution_score', 2)
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict using Random Forest
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        # Calculate enhanced metrics
        concurrent_tasks = self.calculate_enhanced_concurrent_tasks(device_specs)
        performance_score = self.calculate_enhanced_performance_score(device_specs)
        
        # Calculate confidence
        confidence = np.max(probability)
        
        return {
            'category': category,
            'probability': {
                'HIGH': probability[np.where(self.label_encoder.classes_ == 'HIGH')[0][0]],
                'MID': probability[np.where(self.label_encoder.classes_ == 'MID')[0][0]],
                'LOW': probability[np.where(self.label_encoder.classes_ == 'LOW')[0][0]]
            },
            'confidence': confidence,
            'concurrent_tasks': concurrent_tasks,
            'performance_score': performance_score,
            'thermal_throttling_risk': self.assess_thermal_risk(device_specs),
            'gaming_suitability': self.assess_gaming_suitability(device_specs),
            'multitasking_efficiency': self.assess_multitasking_efficiency(device_specs)
        }
    
    def calculate_enhanced_concurrent_tasks(self, device_specs):
        """Enhanced concurrent task calculation"""
        base_tasks = 2
        
        # CPU contribution
        base_tasks += device_specs['cpu_cores'] * 0.8
        
        # RAM contribution with efficiency curve
        ram_factor = min(1.0, device_specs['ram_gb'] / 8.0)
        base_tasks += ram_factor * 6
        
        # Frequency contribution
        freq_factor = min(1.0, device_specs['cpu_frequency_mhz'] / 2500)
        base_tasks += freq_factor * 3
        
        # Thermal and battery factors
        thermal_factor = device_specs.get('thermal_score', 0.8)
        base_tasks *= thermal_factor
        
        return int(min(base_tasks, 20))
    
    def assess_thermal_risk(self, device_specs):
        """Assess thermal throttling risk"""
        thermal_score = device_specs.get('thermal_score', 0.8)
        cpu_intensity = (device_specs['cpu_cores'] * device_specs['cpu_frequency_mhz']) / 20000
        
        risk_score = 1 - (thermal_score * 0.7 + (1 - min(cpu_intensity, 1.0)) * 0.3)
        
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def assess_gaming_suitability(self, device_specs):
        """Assess gaming performance suitability"""
        gpu_score = device_specs['gpu_score']
        ram_score = device_specs['ram_gb']
        cpu_score = device_specs['cpu_cores'] * device_specs['cpu_frequency_mhz'] / 1000
        
        gaming_score = (gpu_score * 0.5 + ram_score * 50 + cpu_score * 0.3) / 3
        
        if gaming_score > 400:
            return "EXCELLENT"
        elif gaming_score > 250:
            return "GOOD"
        elif gaming_score > 150:
            return "MODERATE"
        else:
            return "POOR"
    
    def assess_multitasking_efficiency(self, device_specs):
        """Assess multitasking efficiency"""
        ram_efficiency = min(1.0, device_specs['ram_gb'] / 12.0)
        cpu_efficiency = min(1.0, device_specs['cpu_cores'] / 8.0)
        android_efficiency = min(1.0, device_specs['android_version'] / 14.0)
        
        efficiency_score = (ram_efficiency * 0.5 + cpu_efficiency * 0.3 + android_efficiency * 0.2)
        
        if efficiency_score > 0.8:
            return "EXCELLENT"
        elif efficiency_score > 0.6:
            return "GOOD"
        elif efficiency_score > 0.4:
            return "MODERATE"
        else:
            return "POOR"
    
    def train_enhanced_model(self, data):
        """Train enhanced model with comprehensive evaluation"""
        # Prepare features
        X = data[self.feature_names]
        y = data['performance_category']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                           test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train enhanced Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        # Get predictions for detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Testing Accuracy: {test_accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return train_accuracy, test_accuracy
    
    def save_enhanced_model(self, model_path='enhanced_device_performance_model'):
        """Save enhanced model"""
        # Save Random Forest model
        joblib.dump(self.model, f'{model_path}_rf.pkl')
        joblib.dump(self.scaler, f'{model_path}_scaler.pkl')
        joblib.dump(self.label_encoder, f'{model_path}_encoder.pkl')
        
        # Save TensorFlow model
        if hasattr(self, 'tf_model'):
            self.tf_model.save(f'{model_path}_tf.keras')
        
        # Save enhanced configuration
        with open(f'{model_path}_config.json', 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'label_classes': self.label_encoder.classes_.tolist(),
                'model_version': '2.0',
                'features_count': len(self.feature_names)
            }, f, indent=2)
        
        print(f"Enhanced model saved to {model_path}")
    
    def convert_to_optimized_tflite(self, model_path='enhanced_device_performance_model'):
        """Convert to optimized TensorFlow Lite model"""
        if not hasattr(self, 'tf_model'):
            print("TensorFlow model not found. Please train the model first.")
            return
        
        # Convert with optimizations
        converter = tf.lite.TFLiteConverter.from_keras_model(self.tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable post-training quantization
        converter.representative_dataset = self._representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Save optimized TFLite model
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"Optimized TensorFlow Lite model saved to {model_path}_optimized.tflite")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model
    
    def _representative_dataset(self):
        """Generate representative dataset for quantization"""
        # Create sample data for quantization
        sample_data = np.random.random((100, len(self.feature_names))).astype(np.float32)
        sample_data_scaled = self.scaler.transform(sample_data)
        
        for data in sample_data_scaled:
            yield [data.reshape(1, -1)]

def test_real_devices():
    """Test the model with real device specifications"""
    test_devices = [
        {
            'name': 'Samsung Galaxy S23 Ultra',
            'specs': {
                'cpu_cores': 8,
                'cpu_frequency_mhz': 3200,
                'ram_gb': 12,
                'storage_gb': 256,
                'android_version': 13,
                'gpu_score': 950,
                'architecture_score': 1,
                'thermal_score': 0.9,
                'battery_capacity': 5000,
                'screen_resolution_score': 3
            }
        },
        {
            'name': 'Google Pixel 7',
            'specs': {
                'cpu_cores': 8,
                'cpu_frequency_mhz': 2800,
                'ram_gb': 8,
                'storage_gb': 128,
                'android_version': 13,
                'gpu_score': 750,
                'architecture_score': 1,
                'thermal_score': 0.85,
                'battery_capacity': 4355,
                'screen_resolution_score': 2
            }
        },
        {
            'name': 'Samsung Galaxy A54',
            'specs': {
                'cpu_cores': 8,
                'cpu_frequency_mhz': 2400,
                'ram_gb': 6,
                'storage_gb': 128,
                'android_version': 13,
                'gpu_score': 450,
                'architecture_score': 1,
                'thermal_score': 0.8,
                'battery_capacity': 5000,
                'screen_resolution_score': 2
            }
        },
        {
            'name': 'Xiaomi Redmi 10',
            'specs': {
                'cpu_cores': 8,
                'cpu_frequency_mhz': 2000,
                'ram_gb': 4,
                'storage_gb': 64,
                'android_version': 11,
                'gpu_score': 280,
                'architecture_score': 1,
                'thermal_score': 0.7,
                'battery_capacity': 5000,
                'screen_resolution_score': 2
            }
        }
    ]
    
    return test_devices

def main():
    """Main function to train enhanced model and test"""
    print("Initializing Enhanced Device Performance Model...")
    model = EnhancedDevicePerformanceModel()
    
    # Create enhanced dataset
    print("Creating enhanced synthetic dataset...")
    data = model.create_enhanced_dataset(n_samples=10000)
    
    # Train enhanced Random Forest model
    print("Training enhanced Random Forest model...")
    model.train_enhanced_model(data)
    
    # Train advanced TensorFlow model
    print("Training advanced TensorFlow model...")
    tf_model, history = model.create_advanced_tensorflow_model(data)
    
    # Save enhanced models
    print("Saving enhanced models...")
    model.save_enhanced_model()
    
    # Convert to optimized TensorFlow Lite
    print("Converting to optimized TensorFlow Lite...")
    model.convert_to_optimized_tflite()
    
    # Test with real devices
    print("\n" + "="*50)
    print("TESTING WITH REAL DEVICE SPECIFICATIONS")
    print("="*50)
    
    test_devices = test_real_devices()
    
    for device in test_devices:
        print(f"\nTesting: {device['name']}")
        print("-" * 30)
        
        result = model.predict_device_performance(device['specs'])
        
        print(f"Performance Category: {result['category']}")
        print(f"Performance Score: {result['performance_score']:.1f}/100")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Concurrent Tasks: {result['concurrent_tasks']}")
        print(f"Gaming Suitability: {result['gaming_suitability']}")
        print(f"Multitasking Efficiency: {result['multitasking_efficiency']}")
        print(f"Thermal Risk: {result['thermal_throttling_risk']}")
        
        print("Category Probabilities:")
        for cat, prob in result['probability'].items():
            print(f"  {cat}: {prob:.3f}")

if __name__ == "__main__":
    main()
