import tkinter as tk
from tkinter import ttk
import save_read_model_scaler
import numpy as np
from PIL import Image, ImageTk


loaded_model, loaded_scaler = save_read_model_scaler.load_model_scaler()


def create_widgets(root):
    global age_entry, sex_entry, bp_entry, bs_entry, hr_entry, angina_entry, peak_entry, chest_entry, ecg_entry, slope_entry

    
    welcome_label = ttk.Label(root, text=(
    "Welcome to the Heart Disease Prediction App!\n"
    "Please enter the following data to assess your risk "
    "of developing heart disease. Your information is valuable "
    "and will help us provide better healthcare services."),
        font=('Helvetica', 16), wraplength=750, justify="center")
    welcome_label.grid(row=0, column=0, columnspan=2, pady=(20, 40))

    
    age_entry = ttk.Entry(root)
    age_entry.grid(row=2, column=1, padx=10, pady=5)
    sex_entry = ttk.Entry(root)
    sex_entry.grid(row=3, column=1, padx=10, pady=5)
    bp_entry = ttk.Entry(root)
    bp_entry.grid(row=4, column=1, padx=10, pady=5)
    bs_entry = ttk.Entry(root)
    bs_entry.grid(row=5, column=1, padx=10, pady=5)
    hr_entry = ttk.Entry(root)
    hr_entry.grid(row=6, column=1, padx=10, pady=5)
    angina_entry = ttk.Entry(root)
    angina_entry.grid(row=7, column=1, padx=10, pady=5)
    peak_entry = ttk.Entry(root)
    peak_entry.grid(row=8, column=1, padx=10, pady=5)
    chest_entry = ttk.Entry(root)
    chest_entry.grid(row=9, column=1, padx=10, pady=5)
    ecg_entry = ttk.Entry(root)
    ecg_entry.grid(row=10, column=1, padx=10, pady=5)
    slope_entry = ttk.Entry(root)
    slope_entry.grid(row=11, column=1, padx=10, pady=5)

    
    labels = ['Age', 'Sex (0 - Male, 1 - Female)', 'Resting Blood Pressure',
              'Fasting Blood Sugar (0 - sugar level less than 120 mg/dl, 1 - more than 120mg/dl)',
              'Maximum Heart Rate Achieved (between 60-202)',
              'Exercise Angina (0 - No, 1 - Yes)',
              'Oldpeak (numerical value of ST section)',
              'Chest Pain Type ASY - asymptomatic (0 - No, 1 - Yes)',
              'Resting ECG Normal (0 - No, 1 - Yes)',
              'ST Slope Up (0 - No, 1 - Yes)']
    for i, label_text in enumerate(labels):
        label = ttk.Label(root, text=label_text)
        label.grid(row=i+2, column=0, sticky='w', padx=10, pady=5)

    
    predict_button = ttk.Button(root, text="Predict", command=predict_disease)
    predict_button.grid(row=12, column=0, columnspan=2, padx=5, pady=5)

    
    global result_label
    result_label = ttk.Label(root, text="Prediction: None")
    result_label.grid(row=13, column=0, columnspan=2, padx=5, pady=5)
    
    
    image = Image.open('heart.png')
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.image = photo  
    image_label.grid(row=14, column=0, columnspan=2, sticky='ew')



def predict_disease():
    try:
        # Przygotowanie danych wejściowych
        inputs = np.array([[int(age_entry.get()), int(sex_entry.get()), int(bp_entry.get()),
                            int(bs_entry.get()), int(hr_entry.get()), int(angina_entry.get()),
                            float(peak_entry.get()), int(chest_entry.get()), int(ecg_entry.get()),
                            int(slope_entry.get())]])
        inputs_scaled = loaded_scaler.transform(inputs)
        
        # Przewidywanie
        prediction = loaded_model.predict(inputs_scaled)
        probability = loaded_model.predict_proba(inputs_scaled)[0]
        result_label.config(text=(f"Prediction of heart disease: "
                                  f"{prediction[0]},Probability for classes 0 "
                                  f"(absence of disease) or 1 "
                                  f"(presence of disease): {probability}") )
    except Exception as e:
        result_label.config(text=f'Error: {str(e)}')

def main():
    root = tk.Tk()
    root.title("Heart Disease Prediction")
    root.geometry('760x1000')
    create_widgets(root)
    root.mainloop()


if __name__ == "__main__":
    main()