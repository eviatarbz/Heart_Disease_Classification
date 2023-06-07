import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
class GUI:
    def __init__(self,best_model,columns,convert_age,standardize_data,old_df,nominal_cols):
        
        self.chosen_model = best_model # המודל הנבחר
        self.columns = columns
        self.convert_age =convert_age
        self.standardize_data = standardize_data
        self.old_df = old_df
        self.nominal_cols = nominal_cols
        
        self.root = tk.Tk()#פתיחת ממשק
        self.root.title("Excel Predictor")
        
        
        self.data = None
        self.current_row = None
        self.patient_ID = None

        self.row_label = tk.Label(self.root, text="No file selected") # הצגת התז עבור השורה הנוכחית
        self.row_label.pack(padx=10, pady=10)

        self.root.geometry("400x250+200+200")

        self.file_button = tk.Button(self.root, text="Select File", command=self.select_file) # כפתור לבחירת קובץ
        self.file_button.pack(padx=10, pady=10)
        
        self.id_label = tk.Label(self.root, text="Enter ID:")
        self.id_label.pack(padx=10, pady=6)
        self.id_entry = tk.Entry(self.root) # הכנסת תז
        self.id_entry.pack(padx=10, pady=6)

        self.predict_button = tk.Button(self.root, text="Get Prediction", command=self.get_prediction)
        self.predict_button.pack(padx=10, pady=10)

        
        self.prediction_label = tk.Label(self.root, text="Here you will see the prediction") #תוצאות חיזוי
        self.prediction_label.pack(padx=10, pady=10)

    def preprocess_data(self):#עיבוד המידע הנקלט
        if self.data is not None:
            dataframe = self.convert_age(self.data)
            self.ID = dataframe['ID']
            dataframe = self.standardize_data(dataframe, self.old_df)
            one_hot = pd.get_dummies(dataframe[self.nominal_cols], dtype=float)
            dataframe = pd.concat([dataframe.select_dtypes(include=["float64", "int64"]), one_hot], axis=1)
            dataframe = dataframe.reindex(columns=self.columns, fill_value=0)
            return dataframe
        return None

    def select_file(self):#בחירת קובץ
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_excel(file_path)
            self.data = self.preprocess_data()
            self.row_label.config(text="file selected successfully!")
        else:
            tk.messagebox.showwarning("No file selected", "Please select an Excel file.")

    def get_prediction(self): #קריאה להצגת החיזוי עבור תז מסויים
        if self.data is not None and len(self.data) > 0:
            input_id = self.id_entry.get().strip()
            
            if input_id != "":
                if (input_id.isnumeric() == False):
                    tk.messagebox.showwarning("Invalid ID", "ID must contain numbers only.")
                input_id = int(input_id)
                if input_id in self.ID.values:
                    row_index = self.ID[self.ID == input_id].index[0]
                    self.current_row = row_index
                    self.row_label.config(text="Patient's ID: {}".format(input_id))
                    self.show_prediction()
                else:
                    tk.messagebox.showwarning("Invalid ID", "The entered ID is not found in the file.")
            else:
                tk.messagebox.showwarning("Missing ID", "Please enter an ID.")
        else:
            tk.messagebox.showwarning("No data available", "Please select an Excel file.")

    def show_prediction(self):
        if self.data is not None and len(self.data) > 0:
            values = self.data.iloc[[self.current_row]].values
            result = self.chosen_model.predict(values)
            prediction_probabilities = self.chosen_model.predict_proba(self.data.iloc[[self.current_row]])
            if result[0] == 1:
                prediction = 'Yes'
            else:
                prediction = 'No'

            confidence = prediction_probabilities[0][result[0]]
            self.prediction_label.config(text=f"Prediction: {prediction}\nConfidence: {confidence}")
        else:
            self.prediction_label.config(text="No data available")

    def run(self):
        self.root.mainloop()