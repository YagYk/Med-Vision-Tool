import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
import cv2
import numpy as np

from models.classifier import ImageClassifier
from models.genai_helper import GeminiHelper
from utils.image_processing import preprocess_image, draw_anomalies
from utils.report_generator import generate_report

class ApplicationUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        
        # Initialize models
        self.classifier = ImageClassifier()
        self.genai = GeminiHelper()
        
        # Track current image and analysis
        self.current_image_path = None
        self.current_image = None
        self.current_results = None
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        # Top frame for buttons
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(fill="x", pady=10)
        
        # Load image button
        self.load_button = tk.Button(
            self.top_frame,
            text="Load Image",
            command=self.load_image
        )
        self.load_button.pack(side="left", padx=10)
        
        # Analyze button
        self.analyze_button = tk.Button(
            self.top_frame,
            text="Analyze",
            command=self.analyze_image,
            state="disabled"
        )
        self.analyze_button.pack(side="left", padx=10)
        
        # Save report button
        self.save_button = tk.Button(
            self.top_frame,
            text="Save Report",
            command=self.save_report,
            state="disabled"
        )
        self.save_button.pack(side="left", padx=10)
        
        # Middle frame for image display
        self.image_frame = tk.Frame(self)
        self.image_frame.pack(fill="both", expand=True, pady=10)
        
        # Image display
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill="both", expand=True)
        
        # Bottom frame for results
        self.results_frame = tk.Frame(self)
        self.results_frame.pack(fill="both", expand=True, pady=10)
        
        # Results text box
        self.results_text = tk.Text(self.results_frame, height=15)
        self.results_text.pack(fill="both", expand=True, padx=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side="bottom", fill="x")
    
    def load_image(self):
        """Open a file dialog to load a medical image."""
        file_path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load and display the image
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_button.config(state="normal")
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.save_button.config(state="disabled")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def display_image(self, image_path):
        """Display the image in the UI."""
        # Load image with PIL
        img = Image.open(image_path)
        
        # Resize image to fit display area while maintaining aspect ratio
        display_width = 600
        width, height = img.size
        ratio = display_width / width
        new_size = (display_width, int(height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        
        # Convert to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(img)
        
        # Update image display
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def analyze_image(self):
        """Analyze the loaded image with CV and GenAI."""
        if not self.current_image_path:
            return
        
        # Disable analyze button during processing
        self.analyze_button.config(state="disabled")
        self.status_var.set("Analyzing image...")
        
        # Use a thread to avoid freezing the UI
        def analysis_task():
            try:
                # Preprocess the image
                preprocessed = preprocess_image(self.current_image_path)
                
                # Run computer vision analysis
                cv_results = self.classifier.detect_anomalies(preprocessed)
                
                # Run GenAI analysis
                genai_results = self.genai.analyze_medical_image(
                    self.current_image_path,
                    cv_results
                )
                
                # Generate report
                report = generate_report(
                    self.current_image_path,
                    cv_results,
                    genai_results
                )
                
                # Store results
                self.current_results = {
                    "cv_results": cv_results,
                    "genai_results": genai_results,
                    "report": report
                }
                
                # Update UI with results
                self.display_results(preprocessed, cv_results, report)
            
            except Exception as e:
                # Show error in UI
                self.master.after(0, lambda: messagebox.showerror(
                    "Analysis Error",
                    f"Error during analysis: {str(e)}"
                ))
                self.status_var.set("Analysis failed")
            
            finally:
                # Re-enable analyze button
                self.master.after(0, lambda: self.analyze_button.config(state="normal"))
        
        # Start analysis thread
        threading.Thread(target=analysis_task).start()
    
    def display_results(self, image, cv_results, report):
        """Display analysis results in the UI."""
        # Draw anomalies on image
        result_image = draw_anomalies(image, cv_results)
        
        # Convert to PIL format for display
        result_pil = Image.fromarray(result_image)
        
        # Resize for display
        display_width = 600
        width, height = result_pil.size
        ratio = display_width / width
        new_size = (display_width, int(height * ratio))
        result_pil = result_pil.resize(new_size, Image.LANCZOS)
        
        # Update image display
        photo = ImageTk.PhotoImage(result_pil)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, report)
        
        # Enable save button
        self.save_button.config(state="normal")
        
        # Update status
        anomaly_status = "Anomalies detected" if cv_results.get("has_anomaly", False) else "No anomalies detected"
        self.status_var.set(f"Analysis complete: {anomaly_status}")
    
    def save_report(self):
        """Save the analysis report to a file."""
        if not self.current_results:
            return
        
        # Get file path from user
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Write report to file
            with open(file_path, 'w') as f:
                f.write(self.current_results["report"])
            
            self.status_var.set(f"Report saved to {os.path.basename(file_path)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not save report: {str(e)}") 