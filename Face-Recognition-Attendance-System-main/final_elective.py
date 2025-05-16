import tkinter as tk
from tkinter import *
import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from tkinter import messagebox

window = tk.Tk()
window.title("GROUP 2 : FACE RECOGNITION-BASED ATTENDANCE SYSTEM")

window.geometry('1280x720')
window.configure(background='black')

user_type = tk.StringVar()
user_type.set("student") 
current_subject = tk.StringVar()  

name_id_mapping = {}

def load_existing_ids():
    """Load existing IDs and names from CSV file"""
    global name_id_mapping
    name_id_mapping = {}
    
    if not os.path.exists('StudentDetails/StudentDetails.csv'):
        return
        
    try:
        with open('StudentDetails/StudentDetails.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'ID' in row and 'Name' in row:
                    name_id_mapping[row['ID'].strip()] = row['Name'].strip()
    except Exception as e:
        print(f"Error loading existing IDs: {e}")

def show_registered_students():
    import csv
    from tkinter import ttk
    from datetime import datetime, timedelta
    
    root = tk.Toplevel()
    root.title("Registered Users")
    root.configure(background='grey80')
    root.geometry('1140x600')
    
    filter_frame = tk.Frame(root, bg='grey80')
    filter_frame.pack(pady=10)
    
    tk.Label(root, text="Registered Users Details", 
            font=('times', 20, 'bold'), bg='grey80').pack(pady=10)
    
    # Filter controls
    tk.Label(filter_frame, text="Filter by:", font=('times', 15, 'bold'), bg='grey80').pack(side=tk.LEFT, padx=5)
    
    filter_var = tk.StringVar()
    filter_dropdown = ttk.Combobox(filter_frame, textvariable=filter_var,
                                 values=["All Users", "Students Only", "Professors Only"],
                                 state="readonly", font=('times', 14), width=15)
    filter_dropdown.current(0)
    filter_dropdown.pack(side=tk.LEFT, padx=5)

    tk.Label(filter_frame, text="Subject:", font=('times', 15, 'bold'),
             bg='grey80').pack(side=tk.LEFT, padx=5)
    
    subject_var = tk.StringVar()
    subject_entry = tk.Entry(filter_frame, textvariable=subject_var, font=('times', 14), width=15)
    subject_entry.pack(side=tk.LEFT, padx=5)

    tk.Label(filter_frame, text="Sort by Name:", font=('times', 15, 'bold'), bg='grey80').pack(side=tk.LEFT, padx=5)
    
    sort_var = tk.StringVar()
    sort_dropdown = ttk.Combobox(filter_frame, textvariable=sort_var,
                                 values=["Ascending", "Descending"],
                                 state="readonly", font=('times', 14), width=10)
    sort_dropdown.current(0)
    sort_dropdown.pack(side=tk.LEFT, padx=5)
    
    # Date filter controls
    tk.Label(filter_frame, text="Date Filter:", font=('times', 15, 'bold'), bg='grey80').pack(side=tk.LEFT, padx=5)
    
    date_filter_var = tk.StringVar()
    date_filter_dropdown = ttk.Combobox(filter_frame, textvariable=date_filter_var,
                                      values=["All Dates", "Today", "This Week", "This Month"],
                                      state="readonly", font=('times', 14), width=10)
    date_filter_dropdown.current(0)
    date_filter_dropdown.pack(side=tk.LEFT, padx=5)
    
    table_frame = tk.Frame(root)
    table_frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(table_frame, bg='grey80')
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg='grey80')
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    def get_date_range(filter_type):
        today = datetime.now().date()
        if filter_type == "Today":
            return today, today
        elif filter_type == "This Week":
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=6)
            return start, end
        elif filter_type == "This Month":
            start = today.replace(day=1)
            if today.month == 12:
                end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
            return start, end
        return None, None
    
    def display_data(filter_type="All Users", subject_filter="", sort_order="Ascending", date_filter="All Dates"):
        for widget in scrollable_frame.winfo_children():
            widget.destroy()
        
        headers = ["ID No.", "Name", "User Type", "Subject", "Date", "Time"]
        for col, header in enumerate(headers):
            tk.Label(scrollable_frame, width=15, height=2, fg="black", 
                    font=('times', 15, 'bold'), bg="lightgray", 
                    text=header, relief=tk.RIDGE).grid(row=0, column=col, sticky="nsew")
        
        if not os.path.exists('StudentDetails/StudentDetails.csv'):
            tk.Label(scrollable_frame, text="No user data found!", 
                    fg="red", font=('times', 15)).grid(row=1, column=0, columnspan=6)
            return
            
        try:
            with open('StudentDetails/StudentDetails.csv', newline="") as file:
                if os.stat('StudentDetails/StudentDetails.csv').st_size == 0:
                    tk.Label(scrollable_frame, text="User data file is empty!", 
                            fg="red", font=('times', 15)).grid(row=1, column=0, columnspan=6)
                    return
                
                reader = csv.reader(file)
                try:
                    next(reader) 
                except StopIteration:
                    tk.Label(scrollable_frame, text="User data file is empty!", 
                            fg="red", font=('times', 15)).grid(row=1, column=0, columnspan=6)
                    return
                
                data_rows = []
                date_start, date_end = get_date_range(date_filter)
                
                for row in reader:
                    if len(row) < 6:
                        continue
                    
                    user_type_value = row[2].strip().lower()
                    subject_value = row[3].strip().lower()
                    date_value = row[4].strip()
                    
                    show_row = False
                    
                    if filter_type == "All Users":
                        show_row = True
                    elif filter_type == "Students Only" and user_type_value == "student":
                        show_row = True
                    elif filter_type == "Professors Only" and user_type_value == "professor":
                        show_row = True
                    
                    if subject_filter:
                        if subject_filter.lower() not in subject_value:
                            show_row = False

                    if date_filter == "Today" and show_row:
                        try:
                            row_date = datetime.strptime(date_value, '%Y-%m-%d').date()
                            if date_start == row_date == date_end:
                                show_row = True
                            else:
                                show_row = False
                        except Exception as exe:
                            try:
                                temp = date_value.split('/')
                                date_value = temp[2] + "-" + temp[0] + "-" + temp[1]
                                row_date = datetime.strptime(date_value, '%Y-%m-%d').date()
                                if date_start == row_date == date_end:
                                    show_row = True
                                else:
                                    show_row = False
                            except Exception as exe:
                                show_row = False

                    if date_filter == "This Week" or date_filter == "This Month" and show_row:
                        try:
                            row_date = datetime.strptime(date_value, '%Y-%m-%d').date()
                            if date_end >= row_date >= date_start:
                                show_row = True
                            else:
                                show_row = False
                        except Exception as exe:
                            try:
                                temp = date_value.split('/')
                                date_value = temp[2] + "-" + temp[0] + "-" + temp[1]
                                row_date = datetime.strptime(date_value, '%Y-%m-%d').date()
                                if date_end >= row_date >= date_start:
                                    show_row = True
                                else:
                                    show_row = False
                            except Exception as exe:
                                show_row = False
                    
                    if show_row:
                        data_rows.append(row)
                
                if sort_order == "Ascending":
                    data_rows.sort(key=lambda x: x[1].lower())
                elif sort_order == "Descending":
                    data_rows.sort(key=lambda x: x[1].lower(), reverse=True)
                
                if not data_rows:
                    tk.Label(scrollable_frame, text="No matching records found!", 
                            fg="red", font=('times', 15)).grid(row=1, column=0, columnspan=6)
                else:
                    row_num = 1
                    for row in data_rows:
                        for col, value in enumerate(row):
                            tk.Label(scrollable_frame, width=15, height=1, fg="black", 
                                     font=('times', 13), bg="white", 
                                     text=value, relief=tk.RIDGE).grid(row=row_num, column=col, sticky="nsew")
                        row_num += 1
                
        except Exception as e:
            tk.Label(scrollable_frame, text=f"Error reading data: {str(e)}", 
                    fg="red", font=('times', 15)).grid(row=1, column=0, columnspan=6)
    
    def on_filter_change(event=None):
        display_data(filter_var.get(), subject_var.get(), sort_var.get(), date_filter_var.get())
    
    filter_dropdown.bind("<<ComboboxSelected>>", on_filter_change)
    subject_entry.bind("<KeyRelease>", on_filter_change)
    sort_dropdown.bind("<<ComboboxSelected>>", on_filter_change)
    date_filter_dropdown.bind("<<ComboboxSelected>>", on_filter_change)
    
    display_data()  
    
    root.mainloop()

def clear():
    txt.delete(first=0, last=22)
    txt2.delete(first=0, last=22)
    txt3.delete(first=0, last=22)

def del_sc1():
    sc1.destroy()

def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry('300x100')
    sc1.title('Warning!!')
    sc1.configure(background='grey80')
    Label(sc1, text='All fields are required!!!', fg='black',
          bg='white', font=('times', 16)).pack()
    Button(sc1, text='OK', command=del_sc1, fg="black", bg="lawn green", width=9,
           height=1, activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

def validate_id_name(id_num, name):
    """
    Validate that:
    1. ID must be 4 digits
    2. ID must be unique
    """
    if not (id_num.isdigit() and len(id_num) == 4):
        return False, "ID must be a 4-digit number"
    
    if id_num in name_id_mapping:
        if name_id_mapping[id_num].lower() != name.lower():
            return False, f"ID {id_num} already registered to {name_id_mapping[id_num]}"
        else:
            return True, "Validation passed"
    
    return True, "Validation passed"


def take_img():
    id_num = txt.get().strip()  
    name = txt2.get().strip()  
    subject = txt3.get().strip()  
    
    if not id_num or not name or not subject:
        Notification.configure(
            text="All fields are required!",
            bg="red", width=40, font=('times', 18, 'bold')
        )
        Notification.place(x=300, y=600)
        return
    
    is_valid, validation_msg = validate_id_name(id_num, name)
    if not is_valid:
        Notification.configure(
            text=validation_msg,
            bg="red", width=60, font=('times', 18, 'bold')
        )
        Notification.place(x=200, y=600)
        return
    
    try:
        os.makedirs("TrainingImage", exist_ok=True)
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("Could not access camera")
        
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        sampleNum = 0
        existing_images = len([f for f in os.listdir("TrainingImage") if f.startswith(f"{name}_{id_num}_")])
        
        instructions = [
            "Please move your head slowly:",
            "- Left to right",
            "- Up and down",
            "- Change lighting if possible",
            "We'll capture 200   images for better accuracy"
        ]

        
        while sampleNum < 200:
            ret, img = cam.read()
            if not ret:
                raise Exception("Failed to capture image")
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                img_num = existing_images + sampleNum
                img_name = f"TrainingImage/{name}_{id_num}_{img_num}.jpg"
                cv2.imwrite(img_name, gray[y:y+h, x:x+w])
                
                cv2.putText(img, f"Captured: {img_num}/200", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                for i, instruction in enumerate(instructions):
                    cv2.putText(img, instruction, (10, 70 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f"Registering {name} (ID: {id_num}) - Press Q to cancel", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        if sampleNum > 0:
            if id_num not in name_id_mapping:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                os.makedirs("StudentDetails", exist_ok=True)
                with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    if csvFile.tell() == 0:  
                        writer.writerow(["ID", "Name", "UserType", "Subject", "Date", "Time"])
                    writer.writerow([id_num, name, user_type.get(), subject, date, time_stamp])
                
                name_id_mapping[id_num] = name
            
            Notification.configure(
                text=f"Success! Captured {sampleNum} images for {name} (ID: {id_num})",
                bg="green", width=60, font=('times', 18, 'bold')
            )
        else:
            Notification.configure(
                text="No faces detected during capture!",
                bg="red", width=50, font=('times', 18, 'bold')
            )
            
    except Exception as e:
        Notification.configure(
            text=f"Error: {str(e)}",
            bg="red", width=50, font=('times', 18, 'bold')
        )
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()
    
    Notification.place(x=250, y=600)

def trainimg():
    if txt.get().strip() == '' or txt2.get().strip() == '' or txt3.get().strip() == '':
        Notification.configure(
            text="Error: Please fill all input fields (ID, Name, Subject) before training!",
            bg="red",
            width=60,
            font=('times', 18, 'bold')
        )
        Notification.place(x=100, y=600)
        return

    try:
        if not os.path.exists("TrainingImage") or len(os.listdir("TrainingImage")) == 0:
            raise Exception("No training images found. Please capture images first.")
        
        faces, Ids = getImagesAndLabels("TrainingImage")
        if len(faces) == 0:
            raise Exception("No valid faces detected in training images")
        
        unique_ids = set(Ids)
        print(f"[DEBUG] Unique IDs in training set: {unique_ids}")
        if len(unique_ids) < 2:
            raise Exception(f"Need at least 2 different people for training. Found only {len(unique_ids)}")
        
        face_size = (100, 100) 
        processed_faces = [cv2.resize(face, face_size) for face in faces]
        faces_array = np.array(processed_faces)

        n_samples, h, w = faces_array.shape
        X = faces_array.reshape((n_samples, h * w))
        y = np.array(Ids)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        n_components = min(100, min(X_train.shape[0], X_train.shape[1]))
        pca = PCA(n_components=n_components, whiten=True, random_state=42)
        
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
        svm_classifier.fit(X_train_pca, y_train)
        
        accuracy = svm_classifier.score(X_test_pca, y_test)
        print(f"[INFO] SVM Accuracy: {accuracy:.2%}")
        
        os.makedirs("TrainingImageLabel", exist_ok=True)
        model_path = r"TrainingImageLabel\svm_model.pkl"
        pca_path = r"TrainingImageLabel\pca_model.pkl"
        
        joblib.dump(svm_classifier, model_path)
        joblib.dump(pca, pca_path)
        
        success_msg = (f"Training Successful — Accuracy: {accuracy:.2%}")
        Notification.configure(
            text=success_msg,
            bg="green",
            width=60,
            font=('times', 18, 'bold')
        )
        
    except Exception as e:
        error_msg = f"Training Failed: {str(e)}"
        Notification.configure(
            text=error_msg,
            bg="red",
            width=60,
            font=('times', 18, 'bold')
        )
    
    Notification.place(x=100, y=600)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []

    print(f"Found {len(imagePaths)} images in '{path}'")

    for imagePath in imagePaths:
        try:
            filename = os.path.split(imagePath)[-1]
            parts = filename.split('_')

            if len(parts) < 3 or not parts[1].isdigit():
                print(f"Skipping file with invalid format: {filename}")
                continue

            Id = int(parts[1])  

            pilImage = Image.open(imagePath).convert('L')  
            imageNp = np.array(pilImage, 'uint8')

            faces = detector.detectMultiScale(imageNp)
            if len(faces) == 0:
                print(f"No face detected in image: {filename}")
                continue

            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)

            print(f"Processed: {filename} | ID: {Id}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    print(f"\n[INFO] Total images processed: {len(imagePaths)}")
    print(f"[INFO] Total valid face samples: {len(faceSamples)}")
    print(f"[INFO] Unique IDs in training set: {set(Ids)}\n")

    return faceSamples, Ids


def take_attendance():
    subject = current_subject.get().strip()
    if not subject:
        messagebox.showerror("Error", "Please enter a subject first!")
        return
    
    try:
        if not os.path.exists("TrainingImageLabel/svm_model.pkl") or not os.path.exists("TrainingImageLabel/pca_model.pkl"):
            messagebox.showerror("Error", "Models not found. Please train the system first!")
            return
        
        svm_model = joblib.load("TrainingImageLabel/svm_model.pkl")
        pca = joblib.load("TrainingImageLabel/pca_model.pkl")

        load_existing_ids()
        
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        attendance_file = f"Attendance/Attendance_{subject}.csv"
        os.makedirs("Attendance", exist_ok=True)
        
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            if today in df['Date'].values:
                if messagebox.askyesno("Confirm", f"Attendance for {subject} already taken today. Overwrite?"):
                    df = df[df['Date'] != today]  
                else:
                    return
        
        recognized_ids = set()
        attendance_data = []
        
        while True:
            ret, img = cam.read()
            if not ret:
                raise Exception("Failed to capture image")
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (100, 100))
                
                face_flat = face_resized.reshape(1, -1)
                face_pca = pca.transform(face_flat)
                id_pred = svm_model.predict(face_pca)[0]
                prob = np.max(svm_model.predict_proba(face_pca))
                
                if prob > 0.6: 
                    name = name_id_mapping.get(str(id_pred), "Unknown")
                    
                    cv2.putText(img, f"ID: {id_pred}", (x, y-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(img, f"Name: {name}", (x, y-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    if id_pred not in recognized_ids:
                        recognized_ids.add(id_pred)
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                        user_type_val = "Unknown"
                        if os.path.exists('StudentDetails/StudentDetails.csv'):
                            with open('StudentDetails/StudentDetails.csv', 'r') as csvfile:
                                reader = csv.DictReader(csvfile)
                                for row in reader:
                                    if row['ID'].strip() == str(id_pred):
                                        user_type_val = row.get('UserType', 'Unknown')
                                        break

                        row_data = [id_pred, name, user_type_val, subject, date, time_stamp]
                        attendance_data.append(row_data)

                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if os.stat(attendance_file).st_size == 0:
                                writer.writerow(["ID", "Name", "UserType", "Subject", "Date", "Time"])
                            writer.writerow(row_data)

                        exists_in_registered = False
                        if os.path.exists('StudentDetails/StudentDetails.csv'):
                            with open('StudentDetails/StudentDetails.csv', 'r') as f:
                                reader = csv.reader(f)
                                next(reader, None)
                                for row in reader:
                                    if row[:6] == list(map(str, row_data)):
                                        exists_in_registered = True
                                        break

                        if not exists_in_registered:
                            with open('StudentDetails/StudentDetails.csv', 'a', newline='') as f:
                                writer = csv.writer(f)
                                if os.stat('StudentDetails/StudentDetails.csv').st_size == 0:
                                    writer.writerow(["ID", "Name", "UserType", "Subject", "Date", "Time"])
                                writer.writerow(row_data)

                        cv2.putText(img, "ATTENDANCE SAVED", (x, y+h+30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        cv2.putText(img, "ATTENDANCE MARKED", (x, y+h+30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(img, f"Taking attendance for: {subject}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press 'Q' to save and exit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Taking Attendance - Press Q to save", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        if attendance_data:
            header = ["ID", "Name", "UserType", "Subject", "Date", "Time"]
        
            file_exists = os.path.exists(attendance_file)
        
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow(header)
                
                writer.writerows(attendance_data)

            messagebox.showinfo("Success", f"Attendance recorded for {len(attendance_data)} students in {subject}")
        else:
            messagebox.showwarning("Warning", "No faces recognized for attendance!")

            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to take attendance: {str(e)}")
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

message = tk.Label(window, text="REAL-TIME FACE RECOGNITION BASED", bg="black", fg="white", 
                  width=50, font=('times', 30, ' bold '))
message.place(x=50, y=30)

message = tk.Label(window, text="ATTENDANCE MONITORING SYSTEM", bg="black", fg="white", 
                  width=50, font=('times', 30, ' bold '))
message.place(x=50, y=100)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,
                       height=3, font=('times', 17))

reg_frame = tk.LabelFrame(window, text="Registration", bg="black", fg="white", 
                         font=('times', 15, 'bold'))
reg_frame.place(x=100, y=200, width=500, height=350)

user_type_label = tk.Label(reg_frame, text="User Type:", width=10, height=2,
                          fg="white", bg="black", font=('times', 12, 'bold'))
user_type_label.grid(row=0, column=0, padx=5, pady=5)

student_rb = tk.Radiobutton(reg_frame, text="Student", variable=user_type, 
                           value="student", bg="black", fg="white", font=('times', 12))
student_rb.grid(row=0, column=1, padx=5, pady=5)

professor_rb = tk.Radiobutton(reg_frame, text="Professor", variable=user_type, 
                             value="professor", bg="black", fg="white", font=('times', 12))
professor_rb.grid(row=0, column=2, padx=5, pady=5)

tk.Label(reg_frame, text="ID No. (4 digits):", width=15, height=2,
        fg="white", bg="black", font=('times', 12, 'bold')).grid(row=1, column=0, padx=5, pady=5)

def testVal(inStr, acttyp):
    if acttyp == '1':   
        if not inStr.isdigit() or len(inStr) > 4:
            return False
    return True

txt = tk.Entry(reg_frame, validate="key", width=15, bg="white",
               fg="black", font=('times', 12))
txt['validatecommand'] = (txt.register(testVal), '%P', '%d')
txt.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

tk.Label(reg_frame, text="Name:", width=15, height=2,
        fg="white", bg="black", font=('times', 12, 'bold')).grid(row=2, column=0, padx=5, pady=5)

txt2 = tk.Entry(reg_frame, width=15, bg="white",
                fg="black", font=('times', 12))
txt2.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

tk.Label(reg_frame, text="Subject:", width=15, height=2,
        fg="white", bg="black", font=('times', 12, 'bold')).grid(row=3, column=0, padx=5, pady=5)

txt3 = tk.Entry(reg_frame, width=15, bg="white",
                fg="black", font=('times', 12))
txt3.grid(row=3, column=1, columnspan=2, padx=5, pady=5)

takeImg = tk.Button(reg_frame, text="Take Images", command=take_img, fg="black", bg="SkyBlue1",
                    width=12, height=1, activebackground="white", font=('times', 12, ' bold '))
takeImg.grid(row=4, column=0, padx=5, pady=10)

trainImg = tk.Button(reg_frame, text="Train Images", fg="black", command=trainimg, bg="SkyBlue1",
                     width=12, height=1, activebackground="white", font=('times', 12, ' bold '))
trainImg.grid(row=4, column=1, padx=5, pady=10)

clearButton = tk.Button(reg_frame, text="Clear", command=clear, fg="black", bg="SkyBlue1",
                        width=12, height=1, activebackground="white", font=('times', 12, ' bold '))
clearButton.grid(row=4, column=2, padx=5, pady=10)

att_frame = tk.LabelFrame(window, text="Attendance", bg="black", fg="white", 
                         font=('times', 15, 'bold'))
att_frame.place(x=650, y=200, width=500, height=350)

tk.Label(att_frame, text="Subject:", width=15, height=2,
        fg="white", bg="black", font=('times', 12, 'bold')).grid(row=0, column=0, padx=5, pady=5)

subject_entry = tk.Entry(att_frame, textvariable=current_subject, width=15, bg="white",
                        fg="black", font=('times', 12))
subject_entry.grid(row=0, column=1, padx=5, pady=5)

takeAtt = tk.Button(att_frame, text="Take Attendance", command=take_attendance, fg="black", bg="lawn green",
                    width=20, height=2, activebackground="white", font=('times', 12, ' bold '))
takeAtt.grid(row=1, column=0, columnspan=2, padx=5, pady=20)

viewUsers = tk.Button(att_frame, text="View Registered Users", command=show_registered_students, fg="black", bg="SkyBlue1",
                      width=20, height=2, activebackground="white", font=('times', 12, ' bold '))
viewUsers.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

quitWindow = tk.Button(window, text="Quit", command=on_closing, fg="black",
                       bg="red", width=15, height=2, activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=970, y=600)

window.mainloop()
