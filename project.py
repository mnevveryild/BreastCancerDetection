
from sklearn.datasets import load_breast_cancer  # Veri setini yüklemek için
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak bölmek için
from sklearn.preprocessing import StandardScaler  # Veriyi ölçeklendirmek için
from sklearn.tree import DecisionTreeClassifier  # Karar ağaçları sınıflandırıcı modelini kullanmak için
from sklearn.svm import SVC  # Destek vektör makineleri sınıflandırıcı modelini kullanmak için
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score  # Model değerlendirme için gerekli metrikler
import numpy as np  # NumPy, sayısal işlemler için kullanılır


data = load_breast_cancer()  # Meme kanseri verisetini yükler
X = data.data 
y = data.target  

# Veriyi %80 eğitim, %20 test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Verinin ölçeklenmesi, modellerin daha iyi performans göstermesi için önemlidir
scaler = StandardScaler()  # S
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  


# Karar ağacı sınıflandırma modelini kuruyoruz
dt_model = DecisionTreeClassifier(random_state=42)  
dt_model.fit(X_train_scaled, y_train)  
dt_preds = dt_model.predict(X_test_scaled)  

# Karar ağacı modelinin performansını yazdırıyoruz
print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, dt_preds))  
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_preds))  
print("Classification Report:\n", classification_report(y_test, dt_preds))  
print("ROC-AUC:", roc_auc_score(y_test, dt_preds)) 
print("-" * 50)


# SVM (Support Vector Machine) modelini kuruyoruz
svm_model = SVC(kernel='linear', probability=True) 
svm_model.fit(X_train_scaled, y_train)  
svm_preds = svm_model.predict(X_test_scaled)  

# SVM modelinin performansını yazdırıyoruz
print("SVM")
print("Accuracy:", accuracy_score(y_test, svm_preds)) 
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_preds))  
print("Classification Report:\n", classification_report(y_test, svm_preds))  
print("ROC-AUC:", roc_auc_score(y_test, svm_preds)) 
print("-" * 50)


# Kullanıcıdan yeni veri girmesini ister
print("\n🧪 You can enter a new sample to get a prediction.")  
print("Please enter 30 feature values, separated by commas.")  
print("Example: 14.5,20.0,95.0,...")  

try:
    # Kullanıcıdan veri al
    user_input = input("Enter the data: ") 
    user_input_list = [float(x.strip()) for x in user_input.split(",")]  

    if len(user_input_list) != 30: 
        raise ValueError("Invalid input: Please enter 30 numbers.") 

 
    new_sample = np.array([user_input_list])
    new_sample_scaled = scaler.transform(new_sample)  # veriyi dönüştür

    # Model ile tahmin yap
    prediction = svm_model.predict(new_sample_scaled)

    # Tahmin sonucunu yazdır
    print("\n🔍 Prediction Result:")
    if prediction[0] == 0:
        print("🔴 Malignant Tumor") 
    else:
        print("🟢 Benign Tumor")  
except Exception as e:
 
    print("⚠️ Error:", e)
