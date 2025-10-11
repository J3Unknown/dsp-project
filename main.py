import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys

class SignalVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Visualizer")
        self.root.geometry("900x700")
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables to store signal data
        self.signals = []  # List to store multiple signals
        self.current_signal_index = -1
        
        # Navigation state
        self.pan_start = None
        self.zoom_rect = None
        self.original_xlim = None
        self.original_ylim = None
        self.is_panning = False
        
        # Zoom mode
        self.zoom_mode = "both"  # "both", "x", "y"
        
        # Plot style (continuous vs discrete)
        self.plot_style = "continuous"  # "continuous" or "discrete"
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side controls
        left_controls = tk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Load file button
        load_btn = tk.Button(left_controls, text="Load Signal File", command=self.load_file)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Generate signal button
        generate_btn = tk.Button(left_controls, text="Generate Signal", command=self.open_generate_dialog)
        generate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Signal selection
        self.signal_var = tk.StringVar()
        self.signal_dropdown = ttk.Combobox(left_controls, textvariable=self.signal_var, state="readonly")
        self.signal_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.signal_dropdown.bind('<<ComboboxSelected>>', self.on_signal_select)
        
        # Remove signal button
        remove_btn = tk.Button(left_controls, text="Remove Signal", command=self.remove_signal)
        remove_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear all button
        clear_btn = tk.Button(left_controls, text="Clear All", command=self.clear_all)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Right side controls
        right_controls = tk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT)
        
        # Reset view button
        reset_btn = tk.Button(right_controls, text="Reset View", command=self.reset_view)
        reset_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Normalize button (now opens dialog)
        normalize_btn = tk.Button(right_controls, text="Normalize", command=self.open_normalize_dialog)
        normalize_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Operation frame
        operation_frame = tk.Frame(main_frame)
        operation_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1 operations
        op_row1 = tk.Frame(operation_frame)
        op_row1.pack(fill=tk.X, pady=(0, 5))
        
        # Multiply operation
        multiply_label = tk.Label(op_row1, text="Multiply by:")
        multiply_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.multiply_var = tk.StringVar(value="1.0")
        multiply_entry = tk.Entry(op_row1, textvariable=self.multiply_var, width=8)
        multiply_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        multiply_btn = tk.Button(op_row1, text="Apply", command=self.multiply_signal)
        multiply_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Square signal button
        square_btn = tk.Button(op_row1, text="Square Signal", command=self.square_signal)
        square_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Accumulate signal button (now opens dialog)
        accumulate_btn = tk.Button(op_row1, text="Accumulate Signal", command=self.open_accumulate_dialog)
        accumulate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Row 2 operations
        op_row2 = tk.Frame(operation_frame)
        op_row2.pack(fill=tk.X)
        
        # Add signals button
        add_btn = tk.Button(op_row2, text="Add Selected Signals", command=self.open_add_dialog)
        add_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Subtract signals button
        subtract_btn = tk.Button(op_row2, text="Subtract Selected Signals", command=self.open_subtract_dialog)
        subtract_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Plot style frame
        style_frame = tk.Frame(main_frame)
        style_frame.pack(fill=tk.X, pady=(0, 10))
        
        style_label = tk.Label(style_frame, text="Plot Style:")
        style_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.style_var = tk.StringVar(value="continuous")
        continuous_radio = tk.Radiobutton(style_frame, text="Continuous", variable=self.style_var, value="continuous", command=self.toggle_plot_style)
        continuous_radio.pack(side=tk.LEFT, padx=(0, 5))
        discrete_radio = tk.Radiobutton(style_frame, text="Discrete", variable=self.style_var, value="discrete", command=self.toggle_plot_style)
        discrete_radio.pack(side=tk.LEFT)
        
        # Zoom mode frame
        zoom_frame = tk.Frame(main_frame)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        zoom_label = tk.Label(zoom_frame, text="Zoom Mode:")
        zoom_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.zoom_var = tk.StringVar(value="both")
        both_radio = tk.Radiobutton(zoom_frame, text="Both", variable=self.zoom_var, value="both")
        both_radio.pack(side=tk.LEFT, padx=(0, 5))
        x_radio = tk.Radiobutton(zoom_frame, text="X only", variable=self.zoom_var, value="x")
        x_radio.pack(side=tk.LEFT, padx=(0, 5))
        y_radio = tk.Radiobutton(zoom_frame, text="Y only", variable=self.zoom_var, value="y")
        y_radio.pack(side=tk.LEFT)
        
        # Plot frame
        self.plot_frame = tk.Frame(main_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Information frame
        self.info_frame = tk.Frame(main_frame)
        self.info_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Information labels
        self.info_label = tk.Label(self.info_frame, text="No signal loaded", justify=tk.LEFT)
        self.info_label.pack(fill=tk.X)
        
        # Navigation instructions
        nav_label = tk.Label(self.info_frame, 
                            text="Mouse Controls: Left drag to pan | Right drag to zoom | Scroll to zoom in/out | Use radio buttons for X/Y zoom", 
                            font=("Arial", 8), fg="gray")
        nav_label.pack(fill=tk.X)
        
        # Initialize plot
        self.setup_plot()
        
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Signal Visualization")
        self.ax.set_xlabel("Time / Frequency")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        
        # Embed the plot in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect mouse events for navigation
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
    def toggle_plot_style(self):
        """Toggle between continuous and discrete plot styles"""
        self.plot_style = self.style_var.get()
        if self.current_signal_index >= 0:
            self.plot_signal()
    
    def open_normalize_dialog(self):
        """Open dialog to select normalization type"""
        if self.current_signal_index < 0 or not self.signals:
            messagebox.showwarning("Normalize", "No signal selected")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Normalization Type")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Select normalization type:", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Normalization type selection
        norm_var = tk.StringVar(value="-1 to 1")
        
        frame = tk.Frame(dialog)
        frame.pack(pady=10)
        
        tk.Radiobutton(frame, text="Normalize to [-1, 1] (Amplitude)", 
                      variable=norm_var, value="-1 to 1").pack(anchor="w", pady=5)
        tk.Radiobutton(frame, text="Normalize to [0, 1] (Min-Max)", 
                      variable=norm_var, value="0 to 1").pack(anchor="w", pady=5)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def apply_normalization():
            norm_type = norm_var.get()
            self.normalize_signal(norm_type)
            dialog.destroy()
        
        apply_btn = tk.Button(button_frame, text="Apply", command=apply_normalization)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def open_accumulate_dialog(self):
        """Open dialog to input accumulation parameters"""
        if self.current_signal_index < 0 or not self.signals:
            messagebox.showwarning("Accumulate", "No signal selected")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Accumulation Parameters")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Accumulation Parameters", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Input for initial condition
        ic_frame = tk.Frame(dialog)
        ic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(ic_frame, text="Initial condition (y(-1)):").pack(side=tk.LEFT)
        ic_var = tk.StringVar(value="0.0")
        ic_entry = tk.Entry(ic_frame, textvariable=ic_var, width=10)
        ic_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(ic_frame, text="(sum from k=-∞ to -1)").pack(side=tk.LEFT)
        
        # Explanation
        explanation = tk.Label(dialog, text="Accumulation formula: γ(n) = Σ x(k) from k=-∞ to n\n\n" +
                                           "Practical implementation: y(n) = y(n-1) + x(n)", 
                              font=("Arial", 9), fg="gray", justify=tk.LEFT)
        explanation.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def apply_accumulation():
            try:
                initial_condition = float(ic_var.get())
                self.accumulate_signal(initial_condition)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for initial condition")
        
        apply_btn = tk.Button(button_frame, text="Apply", command=apply_accumulation)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
    def on_closing(self):
        """Properly close the application"""
        self.root.quit()
        self.root.destroy()
        sys.exit()
        
    def open_generate_dialog(self):
        """Open dialog to generate signals from parameters"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Signal")
        dialog.geometry("400x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Signal type
        tk.Label(dialog, text="Signal Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        type_var = tk.StringVar(value="sin")
        type_menu = ttk.Combobox(dialog, textvariable=type_var, values=["sin", "cos", "accumulation"], state="readonly")
        type_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Amplitude
        tk.Label(dialog, text="Amplitude (A):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        amp_var = tk.StringVar(value="1.0")
        amp_entry = tk.Entry(dialog, textvariable=amp_var)
        amp_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Analog Frequency
        tk.Label(dialog, text="Analog Frequency:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        freq_var = tk.StringVar(value="100")
        freq_entry = tk.Entry(dialog, textvariable=freq_var)
        freq_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Sampling Frequency
        tk.Label(dialog, text="Sampling Frequency:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        samp_var = tk.StringVar(value="1000")
        samp_entry = tk.Entry(dialog, textvariable=samp_var)
        samp_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        # Phase Shift
        tk.Label(dialog, text="Phase Shift:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        phase_var = tk.StringVar(value="0.0")
        phase_entry = tk.Entry(dialog, textvariable=phase_var)
        phase_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        
        # Duration
        tk.Label(dialog, text="Duration (seconds):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        duration_var = tk.StringVar(value="1.0")
        duration_entry = tk.Entry(dialog, textvariable=duration_var)
        duration_entry.grid(row=5, column=1, sticky="ew", padx=5, pady=5)
        
        # Smoothness - Default to 10000
        tk.Label(dialog, text="Smoothness (points):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        smooth_var = tk.StringVar(value="10000")
        smooth_entry = tk.Entry(dialog, textvariable=smooth_var)
        smooth_entry.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
        tk.Label(dialog, text="Higher = smoother").grid(row=7, column=1, sticky="w", padx=5)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.grid(row=8, column=0, columnspan=2, pady=10)
        
        def generate():
            try:
                signal_type = type_var.get()
                A = float(amp_var.get())
                analog_freq = float(freq_var.get())
                sampling_freq = float(samp_var.get())
                phase_shift = float(phase_var.get())
                duration = float(duration_var.get())
                smoothness = int(smooth_var.get())
                
                # Generate time vector with high resolution for smoothness
                t = np.linspace(0, duration, smoothness, endpoint=False)
                
                # Generate signal based on type
                if signal_type == "sin":
                    y = A * np.sin(2 * np.pi * analog_freq * t + phase_shift)
                    filename = f"Generated {signal_type} (A={A}, f={analog_freq})"
                elif signal_type == "cos":
                    y = A * np.cos(2 * np.pi * analog_freq * t + phase_shift)
                    filename = f"Generated {signal_type} (A={A}, f={analog_freq})"
                elif signal_type == "accumulation":
                    # Generate a simple signal and then accumulate it
                    # For accumulation, we'll generate a random signal and accumulate it
                    np.random.seed(42)  # For reproducible results
                    x = np.random.randn(len(t)) * A  # Random input signal
                    
                    # Apply accumulation using the exact equation: y(n) = sum_{k=-∞}^{n} x(k)
                    # With initial condition y(-1) = 0
                    y = np.zeros_like(x)
                    y[0] = x[0]  # y(0) = x(0) when initial condition is 0
                    for i in range(1, len(x)):
                        y[i] = y[i-1] + x[i]  # y(n) = y(n-1) + x(n)
                    
                    filename = f"Accumulated signal (A={A})"
                else:
                    messagebox.showerror("Error", "Unknown signal type")
                    return
                
                # Create signal data
                signal_data = {
                    'signal_type': 0,  # Time domain
                    'is_periodic': 0,  # Accumulated signals are not periodic
                    'x': t,
                    'y': y,
                    'filename': filename
                }
                
                self.signals.append(signal_data)
                self.update_signal_dropdown()
                self.current_signal_index = len(self.signals) - 1
                self.signal_var.set(signal_data['filename'])
                self.plot_signal()
                dialog.destroy()
                messagebox.showinfo("Success", "Signal generated successfully")
                
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {str(e)}")
        
        generate_btn = tk.Button(button_frame, text="Generate", command=generate)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        dialog.columnconfigure(1, weight=1)
        
    def open_add_dialog(self):
        """Open dialog to select which signals to add"""
        if len(self.signals) < 2:
            messagebox.showwarning("Add Signals", "Need at least 2 signals to add")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Signals to Add")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main container
        main_container = tk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        tk.Label(main_container, text="Select signals to add:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Create scrollable frame for checkboxes
        canvas = tk.Canvas(main_container, borderwidth=0)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create checkboxes for each signal
        signal_vars = []
        for i, signal in enumerate(self.signals):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(scrollable_frame, text=signal['filename'], variable=var, 
                               font=("Arial", 9), anchor="w")
            cb.pack(fill="x", pady=2)
            signal_vars.append((var, signal))
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Button frame
        button_frame = tk.Frame(main_container)
        button_frame.pack(fill="x", pady=(15, 5))
        
        def add_selected():
            selected_signals = [signal for var, signal in signal_vars if var.get()]
            
            if len(selected_signals) < 2:
                messagebox.showwarning("Add Signals", "Please select at least 2 signals")
                return
                
            # Perform the addition with the selected signals
            self.add_signals(selected_signals)
            dialog.destroy()
        
        # Apply button
        apply_btn = tk.Button(button_frame, text="Apply Addition", command=add_selected, 
                             bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                             padx=20, pady=5)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                              padx=20, pady=5)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Make sure the dialog is properly sized and buttons are visible
        dialog.update()
        
    def open_subtract_dialog(self):
        """Open dialog to select which signals to subtract"""
        if len(self.signals) < 2:
            messagebox.showwarning("Subtract Signals", "Need at least 2 signals to subtract")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Signals to Subtract")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main container
        main_container = tk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        tk.Label(main_container, text="Select signals to subtract (first - second - ...):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Create scrollable frame for checkboxes
        canvas = tk.Canvas(main_container, borderwidth=0)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create checkboxes for each signal
        signal_vars = []
        for i, signal in enumerate(self.signals):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(scrollable_frame, text=signal['filename'], variable=var, 
                               font=("Arial", 9), anchor="w")
            cb.pack(fill="x", pady=2)
            signal_vars.append((var, signal))
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Button frame
        button_frame = tk.Frame(main_container)
        button_frame.pack(fill="x", pady=(15, 5))
        
        def subtract_selected():
            selected_signals = [signal for var, signal in signal_vars if var.get()]
            
            if len(selected_signals) < 2:
                messagebox.showwarning("Subtract Signals", "Please select at least 2 signals")
                return
                
            # Perform the subtraction with the selected signals
            self.subtract_signals(selected_signals)
            dialog.destroy()
        
        # Apply button
        apply_btn = tk.Button(button_frame, text="Apply Subtraction", command=subtract_selected, 
                             bg="#FF9800", fg="white", font=("Arial", 10, "bold"),
                             padx=20, pady=5)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                              padx=20, pady=5)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Make sure the dialog is properly sized and buttons are visible
        dialog.update()
        
    def square_signal(self):
        """Square the currently selected signal (element-wise)"""
        if self.current_signal_index >= 0 and self.signals:
            signal = self.signals[self.current_signal_index]
            signal['y'] = signal['y'] ** 2
            signal['filename'] = f"Squared {signal['filename']}"
            self.update_signal_dropdown()
            self.plot_signal()
            messagebox.showinfo("Square Signal", "Signal squared successfully")
        else:
            messagebox.showwarning("Square Signal", "No signal selected")
            
    def accumulate_signal(self, initial_condition=0.0):
        """
        Apply accumulation to the currently selected signal using the exact equation:
        γ(n) = Σ x(k) from k=-∞ to n
        
        Practical implementation: y(n) = y(n-1) + x(n) with y(-1) = initial_condition
        """
        if self.current_signal_index >= 0 and self.signals:
            signal = self.signals[self.current_signal_index]
            
            # Apply accumulation using the exact equation
            original_y = signal['y'].copy()
            accumulated_y = np.zeros_like(original_y)
            
            # Initialize with initial condition (sum from k=-∞ to -1)
            if len(original_y) > 0:
                accumulated_y[0] = initial_condition + original_y[0]
                
                # Recursive implementation: y(n) = y(n-1) + x(n)
                for i in range(1, len(original_y)):
                    accumulated_y[i] = accumulated_y[i-1] + original_y[i]
            
            signal['y'] = accumulated_y
            signal['filename'] = f"Accumulated (IC={initial_condition}) {signal['filename']}"
            signal['is_periodic'] = 0  # Accumulated signals are not periodic
            self.update_signal_dropdown()
            self.plot_signal()
            
            # Show detailed information about the operation
            info_text = f"Applied accumulation: γ(n) = Σ x(k) from k=-∞ to n\n"
            info_text += f"Initial condition y(-1) = {initial_condition}\n"
            info_text += f"Final value y({len(original_y)-1}) = {accumulated_y[-1]:.6f}"
            messagebox.showinfo("Accumulate Signal", info_text)
        else:
            messagebox.showwarning("Accumulate Signal", "No signal selected")
    
    def normalize_signal(self, norm_type="-1 to 1"):
        """Normalize the signal using the specified method"""
        if self.current_signal_index >= 0 and self.signals:
            signal = self.signals[self.current_signal_index]
            
            if norm_type == "-1 to 1":
                # Amplitude normalization: [-1, 1]
                max_amp = np.max(np.abs(signal['y']))
                if max_amp > 0:
                    signal['y'] = signal['y'] / max_amp
                    messagebox.showinfo("Normalization", "Signal normalized to range [-1, 1]")
                else:
                    messagebox.showwarning("Normalization", "Cannot normalize signal with zero amplitude")
                    return
                    
            elif norm_type == "0 to 1":
                # Min-Max normalization: [0, 1]
                min_val = np.min(signal['y'])
                max_val = np.max(signal['y'])
                if max_val != min_val:
                    signal['y'] = (signal['y'] - min_val) / (max_val - min_val)
                    messagebox.showinfo("Normalization", "Signal normalized to range [0, 1]")
                else:
                    messagebox.showwarning("Normalization", "Cannot normalize constant signal")
                    return
            
            self.plot_signal()
        else:
            messagebox.showwarning("Normalization", "No signal selected")
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left mouse button - pan
            self.pan_start = (event.xdata, event.ydata, event.x, event.y)
            self.original_xlim = self.ax.get_xlim()
            self.original_ylim = self.ax.get_ylim()
            self.is_panning = True
            self.canvas.get_tk_widget().configure(cursor="fleur")
            
        elif event.button == 3:  # Right mouse button - zoom
            self.pan_start = (event.xdata, event.ydata)
            self.zoom_rect = plt.Rectangle((event.xdata, event.ydata), 0, 0, 
                                          fill=False, linestyle='--', color='red', linewidth=1.5)
            self.ax.add_patch(self.zoom_rect)
            self.canvas.draw()
            
    def on_release(self, event):
        if event.button == 1:  # Left mouse button - pan
            self.is_panning = False
            self.canvas.get_tk_widget().configure(cursor="")
            
        elif event.button == 3:  # Right mouse button - zoom
            if self.pan_start and self.zoom_rect:
                x_start, y_start = self.pan_start
                x_end, y_end = event.xdata, event.ydata
                
                # Remove the zoom rectangle
                self.zoom_rect.remove()
                self.zoom_rect = None
                
                # Apply zoom to the selected region
                if x_start is not None and y_start is not None and x_end is not None and y_end is not None:
                    if abs(x_end - x_start) > 0 and abs(y_end - y_start) > 0:
                        self.ax.set_xlim(min(x_start, x_end), max(x_start, x_end))
                        self.ax.set_ylim(min(y_start, y_end), max(y_start, y_end))
                        self.canvas.draw()
                    
            self.pan_start = None
            self.canvas.get_tk_widget().configure(cursor="")
            
    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
            
        if self.is_panning and self.pan_start and len(self.pan_start) == 4:  # Continuous panning
            x_start, y_start, pixel_x, pixel_y = self.pan_start
            
            # Calculate pixel distance moved
            dx_pixels = event.x - pixel_x
            dy_pixels = event.y - pixel_y
            
            # Convert pixel distance to data distance
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Get figure dimensions in pixels
            bbox = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width_pixels = width * self.fig.dpi
            height_pixels = height * self.fig.dpi
            
            # Calculate data distance per pixel
            x_per_pixel = (xlim[1] - xlim[0]) / width_pixels
            y_per_pixel = (ylim[1] - ylim[0]) / height_pixels
            
            # Calculate new limits - FIXED Y PANNING: Now natural Y panning direction
            new_xlim = (xlim[0] - dx_pixels * x_per_pixel, xlim[1] - dx_pixels * x_per_pixel)
            new_ylim = (ylim[0] - dy_pixels * y_per_pixel, ylim[1] - dy_pixels * y_per_pixel)  # Fixed Y direction
            
            # Apply new limits
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw()
            
            # Update pan start for continuous panning
            self.pan_start = (x_start, y_start, event.x, event.y)
            
        elif self.pan_start and event.button == 3 and self.zoom_rect:  # Right drag - update zoom rectangle
            x_start, y_start = self.pan_start
            x_end, y_end = event.xdata, event.ydata
            
            # Update the zoom rectangle
            if x_end is not None and y_end is not None:
                self.zoom_rect.set_width(x_end - x_start)
                self.zoom_rect.set_height(y_end - y_start)
                self.canvas.draw()
            
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
            
        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate zoom factor
        zoom_factor = 0.9 if event.button == 'up' else 1.1
        
        # Get mouse position in data coordinates
        x_mouse = event.xdata
        y_mouse = event.ydata
        
        # Apply zoom based on selected mode - IMPROVED: Zoom relative to mouse position
        zoom_mode = self.zoom_var.get()
        
        if zoom_mode == "both" or zoom_mode == "x":
            # Zoom X relative to mouse position
            x_range = xlim[1] - xlim[0]
            new_x_range = x_range * zoom_factor
            
            # Calculate new limits keeping mouse position fixed
            left_ratio = (x_mouse - xlim[0]) / x_range
            right_ratio = (xlim[1] - x_mouse) / x_range
            
            new_xlim = (
                x_mouse - left_ratio * new_x_range,
                x_mouse + right_ratio * new_x_range
            )
            self.ax.set_xlim(new_xlim)
        
        if zoom_mode == "both" or zoom_mode == "y":
            # Zoom Y relative to mouse position
            y_range = ylim[1] - ylim[0]
            new_y_range = y_range * zoom_factor
            
            # Calculate new limits keeping mouse position fixed
            bottom_ratio = (y_mouse - ylim[0]) / y_range
            top_ratio = (ylim[1] - y_mouse) / y_range
            
            new_ylim = (
                y_mouse - bottom_ratio * new_y_range,
                y_mouse + top_ratio * new_y_range
            )
            self.ax.set_ylim(new_ylim)
        
        self.canvas.draw()
        
    def reset_view(self):
        if self.current_signal_index >= 0 and self.signals:
            signal = self.signals[self.current_signal_index]
            
            # Reset to original signal bounds with a small margin
            x_margin = (signal['x'].max() - signal['x'].min()) * 0.05
            y_margin = (signal['y'].max() - signal['y'].min()) * 0.05
            
            self.ax.set_xlim(signal['x'].min() - x_margin, signal['x'].max() + x_margin)
            self.ax.set_ylim(signal['y'].min() - y_margin, signal['y'].max() + y_margin)
            self.canvas.draw()
        else:
            # Reset to default view
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
    
    def multiply_signal(self):
        if self.current_signal_index >= 0 and self.signals:
            try:
                factor = float(self.multiply_var.get())
                signal = self.signals[self.current_signal_index]
                signal['y'] = signal['y'] * factor
                self.plot_signal()
                messagebox.showinfo("Multiplication", f"Signal multiplied by {factor}")
            except ValueError:
                messagebox.showerror("Error", "Invalid multiplication factor")
        else:
            messagebox.showwarning("Multiplication", "No signal selected")
    
    def add_signals(self, selected_signals=None):
        """Add multiple signals together"""
        if selected_signals is None:
            if len(self.signals) < 2:
                messagebox.showwarning("Add Signals", "Need at least 2 signals to add")
                return
            selected_signals = self.signals[:2]  # Default to first two
        
        if len(selected_signals) < 2:
            messagebox.showwarning("Add Signals", "Need at least 2 signals to add")
            return
        
        # Find common x range
        x_min = max(signal['x'].min() for signal in selected_signals)
        x_max = min(signal['x'].max() for signal in selected_signals)
        
        if x_min >= x_max:
            messagebox.showerror("Error", "Signals have no overlapping x-range")
            return
        
        # Create a common x axis with the highest resolution
        min_dx = min(np.min(np.diff(signal['x'])) for signal in selected_signals)
        x_common = np.arange(x_min, x_max, min_dx)
        
        if len(x_common) == 0:
            messagebox.showerror("Error", "Could not create common x-axis for signals")
            return
        
        # Interpolate all signals to the common x axis and sum them
        y_sum = np.zeros_like(x_common)
        signal_names = []
        
        for signal in selected_signals:
            y_interp = np.interp(x_common, signal['x'], signal['y'])
            y_sum += y_interp
            signal_names.append(signal['filename'])
        
        # Create a new signal representing the sum
        new_signal = {
            'signal_type': selected_signals[0]['signal_type'],  # Use first signal's type
            'is_periodic': all(signal['is_periodic'] for signal in selected_signals),
            'x': x_common,
            'y': y_sum,
            'filename': f"Sum of {len(selected_signals)} signals"
        }
        
        self.signals.append(new_signal)
        self.update_signal_dropdown()
        self.current_signal_index = len(self.signals) - 1
        self.signal_var.set(new_signal['filename'])
        self.plot_signal()
        messagebox.showinfo("Add Signals", f"{len(selected_signals)} signals added successfully")
    
    def subtract_signals(self, selected_signals=None):
        """Subtract multiple signals (first - second - third - ...)"""
        if selected_signals is None:
            if len(self.signals) < 2:
                messagebox.showwarning("Subtract Signals", "Need at least 2 signals to subtract")
                return
            selected_signals = self.signals[:2]  # Default to first two
        
        if len(selected_signals) < 2:
            messagebox.showwarning("Subtract Signals", "Need at least 2 signals to subtract")
            return
        
        # Find common x range
        x_min = max(signal['x'].min() for signal in selected_signals)
        x_max = min(signal['x'].max() for signal in selected_signals)
        
        if x_min >= x_max:
            messagebox.showerror("Error", "Signals have no overlapping x-range")
            return
        
        # Create a common x axis with the highest resolution
        min_dx = min(np.min(np.diff(signal['x'])) for signal in selected_signals)
        x_common = np.arange(x_min, x_max, min_dx)
        
        if len(x_common) == 0:
            messagebox.showerror("Error", "Could not create common x-axis for signals")
            return
        
        # Interpolate all signals to the common x axis and subtract them
        y_result = None
        signal_names = []
        
        for i, signal in enumerate(selected_signals):
            y_interp = np.interp(x_common, signal['x'], signal['y'])
            if i == 0:
                y_result = y_interp
            else:
                y_result -= y_interp
            signal_names.append(signal['filename'])
        
        # Create a new signal representing the subtraction
        new_signal = {
            'signal_type': selected_signals[0]['signal_type'],  # Use first signal's type
            'is_periodic': all(signal['is_periodic'] for signal in selected_signals),
            'x': x_common,
            'y': y_result,
            'filename': f"Subtraction of {len(selected_signals)} signals"
        }
        
        self.signals.append(new_signal)
        self.update_signal_dropdown()
        self.current_signal_index = len(self.signals) - 1
        self.signal_var.set(new_signal['filename'])
        self.plot_signal()
        messagebox.showinfo("Subtract Signals", f"{len(selected_signals)} signals subtracted successfully")
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                signal_data = self.read_signal_file(file_path)
                if signal_data:
                    # Add filename to signal data
                    signal_data['filename'] = os.path.basename(file_path)
                    
                    # ENHANCED: Make loaded signals smoother by interpolating to higher resolution
                    if len(signal_data['x']) > 1:  # Only if we have at least 2 points
                        x_original = signal_data['x']
                        y_original = signal_data['y']
                        
                        # Create new x values with more points (10000 points for smoothness)
                        num_points = max(10000, len(x_original))  # Use at least 10000 points
                        x_new = np.linspace(x_original.min(), x_original.max(), num_points)
                        
                        # Interpolate y values
                        y_new = np.interp(x_new, x_original, y_original)
                        
                        # Replace with smoother signal
                        signal_data['x'] = x_new
                        signal_data['y'] = y_new
                    
                    self.signals.append(signal_data)
                    self.update_signal_dropdown()
                    # Select the newly loaded signal
                    self.current_signal_index = len(self.signals) - 1
                    self.signal_var.set(signal_data['filename'])
                    self.plot_signal()
                    messagebox.showinfo("Success", f"Signal loaded and smoothed to {len(signal_data['x'])} points")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def read_signal_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) < 3:
            raise ValueError("File does not contain enough data")
        
        # Parse header information
        signal_type = int(lines[0])
        is_periodic = int(lines[1])
        num_points = int(lines[2])
        
        if len(lines) < 3 + num_points:
            raise ValueError("File does not contain enough data points")
        
        # Parse data points
        x_values = []
        y_values = []
        
        for i in range(3, 3 + num_points):
            values = lines[i].split()
            if len(values) < 2:
                continue
                
            x_values.append(float(values[0]))
            y_values.append(float(values[1]))
        
        return {
            'signal_type': signal_type,
            'is_periodic': is_periodic,
            'x': np.array(x_values),
            'y': np.array(y_values)
        }
    
    def update_signal_dropdown(self):
        filenames = [signal['filename'] for signal in self.signals]
        self.signal_dropdown['values'] = filenames
        
        if self.signals:
            self.signal_dropdown.configure(state="readonly")
            if self.current_signal_index < 0:
                self.current_signal_index = 0
                self.signal_var.set(filenames[0])
        else:
            self.signal_dropdown.configure(state="disabled")
            self.signal_var.set("")
    
    def on_signal_select(self, event=None):
        selected_file = self.signal_var.get()
        for i, signal in enumerate(self.signals):
            if signal['filename'] == selected_file:
                self.current_signal_index = i
                self.plot_signal()
                break
    
    def plot_signal(self):
        if self.current_signal_index < 0 or not self.signals:
            return
            
        signal = self.signals[self.current_signal_index]
        
        # Clear the plot
        self.ax.clear()
        
        # Plot the signal based on the selected style
        if self.plot_style == "continuous":
            # Continuous plot (line)
            self.ax.plot(signal['x'], signal['y'], 'b-', linewidth=1.5)
        else:
            # Discrete plot (stem)
            markerline, stemlines, baseline = self.ax.stem(signal['x'], signal['y'], basefmt=" ")
            plt.setp(stemlines, 'linewidth', 1.5)
            plt.setp(markerline, 'markersize', 3)
        
        # Set labels based on signal type
        if signal['signal_type'] == 0:  # Time domain
            self.ax.set_xlabel("Time")
            self.ax.set_title(f"Time Domain Signal - {signal['filename']}")
        else:  # Frequency domain
            self.ax.set_xlabel("Frequency")
            self.ax.set_title(f"Frequency Domain Signal - {signal['filename']}")
            
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        
        # Set initial view with margins
        if len(signal['x']) > 0:
            x_margin = (signal['x'].max() - signal['x'].min()) * 0.05
            y_margin = (signal['y'].max() - signal['y'].min()) * 0.05
            
            self.ax.set_xlim(signal['x'].min() - x_margin, signal['x'].max() + x_margin)
            self.ax.set_ylim(signal['y'].min() - y_margin, signal['y'].max() + y_margin)
        
        # Update information label
        domain = "Time" if signal['signal_type'] == 0 else "Frequency"
        periodic = "Periodic" if signal['is_periodic'] == 1 else "Non-periodic"
        style = "Continuous" if self.plot_style == "continuous" else "Discrete"
        info_text = f"File: {signal['filename']} | Domain: {domain} | {periodic} | Style: {style} | Points: {len(signal['x'])}"
        self.info_label.config(text=info_text)
        
        # Refresh the canvas
        self.canvas.draw()
    
    def remove_signal(self):
        if self.current_signal_index >= 0 and self.signals:
            removed_signal = self.signals.pop(self.current_signal_index)
            self.update_signal_dropdown()
            
            if self.signals:
                self.current_signal_index = min(self.current_signal_index, len(self.signals) - 1)
                self.signal_var.set(self.signals[self.current_signal_index]['filename'])
                self.plot_signal()
            else:
                self.current_signal_index = -1
                self.signal_var.set("")
                self.ax.clear()
                self.ax.set_title("Signal Visualization")
                self.ax.set_xlabel("Time / Frequency")
                self.ax.set_ylabel("Amplitude")
                self.ax.grid(True)
                self.canvas.draw()
                self.info_label.config(text="No signal loaded")
    
    def clear_all(self):
        self.signals = []
        self.current_signal_index = -1
        self.update_signal_dropdown()
        self.ax.clear()
        self.ax.set_title("Signal Visualization")
        self.ax.set_xlabel("Time / Frequency")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.canvas.draw()
        self.info_label.config(text="No signal loaded")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalVisualizer(root)
    root.mainloop()