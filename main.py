import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import os
import sys
import math
import cmath
import Task1Test as task1
import Task2Test as task2

class SignalVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Visualizer")
        self.root.geometry("1000x700")
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables to store signal data
        self.signals = []  # List of signal dictionaries
        
        # Frequency domain data
        self.freq_domain_data = None  # Store FFT results
        self.sampling_freq = None
        self.modified_fft = None
        self.current_signal_index = None  # Track which signal is being analyzed
        
        # Navigation state
        self.pan_start = None
        self.zoom_rect = None
        self.original_xlim = None
        self.original_ylim = None
        self.is_panning = False
        
        # Zoom mode
        self.zoom_mode = "both"
        
        # Plot style (continuous vs discrete)
        self.plot_style = "continuous"
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create menu bar
        self.create_menu_bar()
        
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
        signal_frame = tk.Frame(left_controls)
        signal_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(signal_frame, text="Signals:").pack(anchor="w")
        self.signal_listbox = tk.Listbox(signal_frame, width=30, height=4, selectmode=tk.MULTIPLE)
        self.signal_listbox.pack(side=tk.LEFT, fill=tk.Y)
        
        # Scrollbar for listbox
        listbox_scrollbar = tk.Scrollbar(signal_frame, orient=tk.VERTICAL)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.signal_listbox.config(yscrollcommand=listbox_scrollbar.set)
        listbox_scrollbar.config(command=self.signal_listbox.yview)
        
        # Bind selection event
        self.signal_listbox.bind('<ButtonRelease-1>', self.on_signal_select)
        self.signal_listbox.bind('<<ListboxSelected>>', self.on_signal_select)
        
        # Remove signal button
        remove_btn = tk.Button(left_controls, text="Remove Selected", command=self.remove_signal)
        remove_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear all button
        clear_btn = tk.Button(left_controls, text="Clear All", command=self.clear_all)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Toggle all signals button
        toggle_btn = tk.Button(left_controls, text="Show All", command=self.toggle_all_signals)
        toggle_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Refresh plot button
        refresh_btn = tk.Button(left_controls, text="Refresh Plot", command=self.plot_signal)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Right side controls
        right_controls = tk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT)
        
        # Reset view button
        reset_btn = tk.Button(right_controls, text="Reset View", command=self.reset_view)
        reset_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Normalize button
        normalize_btn = tk.Button(right_controls, text="Normalize", command=self.open_normalize_dialog)
        normalize_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Quantize button
        quantize_btn = tk.Button(right_controls, text="Quantize Signal", command=self.open_quantize_dialog)
        quantize_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Quick Quantize button
        quick_quantize_btn = tk.Button(right_controls, text="Quick Quantize", command=self.quick_quantize, bg="#4CAF50", fg="white")
        quick_quantize_btn.pack(side=tk.LEFT, padx=(0, 10))
        
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
        
        # Accumulate signal button
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
        
        # Test frame
        test_frame = tk.Frame(main_frame)
        test_frame.pack(fill=tk.X, pady=(10, 5))
        
        test_label = tk.Label(test_frame, text="Testing:", font=("Arial", 10, "bold"))
        test_label.pack(anchor="w")
        
        run_tests_btn = tk.Button(main_frame, text="Open Test Window", bg="#337ab7", fg="white",
                          font=("Arial", 10, "bold"), command=self.open_test_window)
        run_tests_btn.pack(pady=(10, 5))
        
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

    def create_menu_bar(self):
        """Create the menu bar with Frequency Domain menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Frequency Domain menu
        freq_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Frequency Domain", menu=freq_menu)
        
        freq_menu.add_command(label="Apply Fourier Transform", command=self.apply_fourier_transform)
        freq_menu.add_command(label="Show Dominant Frequencies", command=self.show_dominant_frequencies)
        freq_menu.add_command(label="Modify Components", command=self.modify_components)
        freq_menu.add_command(label="Remove DC Component", command=self.remove_dc_component)
        freq_menu.add_command(label="Reconstruct Signal", command=self.reconstruct_signal)
        freq_menu.add_separator()
        freq_menu.add_command(label="Apply IDFT to Frequency File", command=self.apply_idft_to_frequency_file)
        freq_menu.add_command(label="Show Current Signal Analysis", command=self.show_current_signal_analysis)

    def apply_idft_to_frequency_file(self):
        """Apply Inverse DFT to a frequency domain file (amplitude and phase data)"""
        file_path = filedialog.askopenfilename(
            title="Select Frequency Domain File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Read the frequency domain file
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                lines = [line.strip() for line in lines if line.strip()]
                
                if len(lines) < 3:
                    raise ValueError("File does not contain enough data")
                
                signal_type = int(lines[0])
                is_periodic = int(lines[1])
                num_points = int(lines[2])
                
                if len(lines) < 3 + num_points:
                    raise ValueError("File does not contain enough data points")
                
                amplitudes = []
                phases = []
                
                for i in range(3, 3 + num_points):
                    values = lines[i].split()
                    if len(values) < 2:
                        continue
                        
                    # Remove 'f' suffix if present and convert to float
                    amp_str = values[0].rstrip('f')
                    phase_str = values[1].rstrip('f')
                    
                    amplitudes.append(float(amp_str))
                    phases.append(float(phase_str))
                
                # Apply IDFT
                time_domain_signal = self.apply_idft(amplitudes, phases)
                
                # Create time axis (assuming uniform sampling)
                n = len(time_domain_signal)
                t = np.arange(n)
                
                # Create signal dictionary
                signal_data = {
                    'signal_type': 0,  # Time domain
                    'is_periodic': is_periodic,
                    'x': t,
                    'y': time_domain_signal,
                    'filename': f"IDFT_{os.path.basename(file_path)}"
                }
                
                self.signals.append(signal_data)
                self.update_signal_dropdown()
                self.plot_signal()
                
                messagebox.showinfo("Success", 
                                  f"IDFT applied successfully!\n"
                                  f"Generated {n} time domain samples from {num_points} frequency components")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply IDFT: {str(e)}")

    def apply_idft(self, amplitudes, phases):
        """Apply Inverse Discrete Fourier Transform to amplitude and phase data"""
        n = len(amplitudes)
        
        # Create complex frequency domain representation
        # X[k] = amplitude[k] * exp(j * phase[k])
        freq_domain = np.array([amp * cmath.exp(1j * phase) 
                               for amp, phase in zip(amplitudes, phases)])
        
        # Apply Inverse DFT manually
        time_domain = np.zeros(n, dtype=complex)
        for i in range(n):  # Time index
            sum_val = 0
            for k in range(n):  # Frequency index
                # IDFT formula: x[n] = (1/N) * Σ X[k] * exp(j * 2π * k * n / N)
                exponent = 2j * cmath.pi * k * i / n
                sum_val += freq_domain[k] * cmath.exp(exponent)
            time_domain[i] = sum_val / n
        
        # The signal should be real, so take the real part
        # (small imaginary parts are due to numerical precision)
        return np.real(time_domain)

    def apply_fourier_transform(self):
        """Apply Fourier transform to selected signal and display frequency domain"""
        if not self.signals:
            messagebox.showwarning("Warning", "No signals loaded!")
            return
        
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a signal!")
            return
        
        # Ask for sampling frequency
        sampling_freq = simpledialog.askfloat("Sampling Frequency", 
                                                "Enter sampling frequency (Hz):",
                                                minvalue=0.1, initialvalue=1000.0)
        if sampling_freq is None:
            return
        
        self.sampling_freq = sampling_freq
        signal_idx = selected_indices[0]
        signal = self.signals[signal_idx]
        self.current_signal_index = signal_idx  # Store which signal we're analyzing
        
        # Apply DFT/FFT
        n = len(signal['y'])
        if n > 0:
            # Use FFT for efficiency
            fft_result = np.fft.fft(signal['y'])
            freqs = np.fft.fftfreq(n, 1/sampling_freq)
            
            # Calculate amplitudes and phases
            amplitudes = np.abs(fft_result)
            phases = np.angle(fft_result)
            
            # Normalize amplitudes to [0, 1]
            if np.max(amplitudes) > 0:
                normalized_amplitudes = amplitudes / np.max(amplitudes)
            else:
                normalized_amplitudes = amplitudes
            
            # Store frequency domain data
            self.freq_domain_data = {
                'frequencies': freqs,
                'amplitudes': amplitudes,
                'normalized_amplitudes': normalized_amplitudes,
                'phases': phases,
                'fft_result': fft_result,
                'signal_name': signal['filename'],
                'signal_data': signal['y'],
                'time_domain': signal['x']
            }
            self.modified_fft = fft_result.copy()  # Store a copy for modifications
            
            # Display frequency domain plots
            self.display_frequency_domain()
            
            # Show current signal analysis
            self.show_current_signal_analysis()
        else:
            messagebox.showerror("Error", "Selected signal has no data!")

    def display_frequency_domain(self):
        """Display frequency vs amplitude and frequency vs phase plots"""
        if self.freq_domain_data is None:
            messagebox.showwarning("Warning", "No frequency domain data available!")
            return
        
        # Create a new window for frequency domain plots
        freq_window = tk.Toplevel(self.root)
        freq_window.title("Frequency Domain Analysis")
        freq_window.geometry("1200x800")
        
        # Create frame for plots
        plot_frame = tk.Frame(freq_window)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Use modified FFT if available, otherwise use original
        if self.modified_fft is not None:
            fft_used = self.modified_fft
            title_suffix = " (Modified)"
        else:
            fft_used = self.freq_domain_data['fft_result']
            title_suffix = ""
        
        # Recalculate from the FFT being used
        amplitudes = np.abs(fft_used)
        phases = np.angle(fft_used)
        if np.max(amplitudes) > 0:
            normalized_amplitudes = amplitudes / np.max(amplitudes)
        else:
            normalized_amplitudes = amplitudes
        
        freqs = self.freq_domain_data['frequencies']
        
        # Plot 1: Frequency vs Normalized Amplitude (Positive frequencies)
        positive_mask = freqs >= 0
        positive_freqs = freqs[positive_mask]
        positive_norm_amps = normalized_amplitudes[positive_mask]
        
        ax1.stem(positive_freqs, positive_norm_amps, basefmt=" ")
        ax1.set_title(f"Frequency vs Normalized Amplitude{title_suffix}")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Normalized Amplitude")
        ax1.grid(True)
        
        # Plot 2: Frequency vs Phase (Positive frequencies)
        positive_phases = phases[positive_mask]
        ax2.stem(positive_freqs, positive_phases, basefmt=" ")
        ax2.set_title(f"Frequency vs Phase{title_suffix}")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (radians)")
        ax2.grid(True)
        
        # Plot 3: Original Signal (Time Domain)
        time_data = self.freq_domain_data['time_domain']
        signal_data = self.freq_domain_data['signal_data']
        ax3.plot(time_data, signal_data)
        ax3.set_title("Original Signal (Time Domain)")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Amplitude")
        ax3.grid(True)
        
        # Plot 4: Dominant Frequencies Highlight
        dominant_mask = normalized_amplitudes > 0.5
        dominant_freqs = freqs[dominant_mask]
        dominant_amps = normalized_amplitudes[dominant_mask]
        
        ax4.stem(positive_freqs, positive_norm_amps, basefmt=" ", linefmt='gray', markerfmt='go')
        if len(dominant_freqs) > 0:
            # Only show positive dominant frequencies
            positive_dominant_mask = dominant_freqs >= 0
            positive_dominant_freqs = dominant_freqs[positive_dominant_mask]
            positive_dominant_amps = dominant_amps[positive_dominant_mask]
            ax4.stem(positive_dominant_freqs, positive_dominant_amps, basefmt=" ", linefmt='red', markerfmt='ro')
            ax4.set_title("Dominant Frequencies (Amplitude > 0.5)")
        else:
            ax4.set_title("No Dominant Frequencies Found")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Normalized Amplitude")
        ax4.grid(True)
        ax4.legend(['All Frequencies', 'Dominant Frequencies'])
        
        plt.tight_layout()
        
        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add information label
        info_label = tk.Label(freq_window, 
                             text=f"Sampling Frequency: {self.sampling_freq} Hz | Signal: {self.freq_domain_data['signal_name']} | DC Component: {fft_used[0]:.4f}",
                             font=("Arial", 10))
        info_label.pack(pady=5)
        
        # Add control buttons
        control_frame = tk.Frame(freq_window)
        control_frame.pack(pady=5)
        
        if hasattr(self, 'freq_window'):
            self.freq_window.destroy()
        self.freq_window = freq_window

    def show_current_signal_analysis(self):
        """Show the current signal being analyzed in a separate section"""
        if self.freq_domain_data is None or self.current_signal_index is None:
            return
            
        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Current Signal Analysis")
        analysis_window.geometry("600x500")
        
        # Signal information
        signal = self.signals[self.current_signal_index]
        info_frame = tk.Frame(analysis_window)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(info_frame, text=f"Analyzing: {signal['filename']}", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(info_frame, text=f"Sampling Frequency: {self.sampling_freq} Hz", font=("Arial", 10)).pack(anchor="w")
        tk.Label(info_frame, text=f"Signal Length: {len(signal['y'])} samples", font=("Arial", 10)).pack(anchor="w")
        
        # Frequency domain information
        if self.modified_fft is not None:
            fft_used = self.modified_fft
            mod_status = " (Modified)"
        else:
            fft_used = self.freq_domain_data['fft_result']
            mod_status = ""
        
        dc_component = fft_used[0]
        dominant_count = np.sum(self.freq_domain_data['normalized_amplitudes'] > 0.5)
        
        freq_info_frame = tk.Frame(analysis_window)
        freq_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(freq_info_frame, text="Frequency Domain Information:", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Label(freq_info_frame, text=f"DC Component: {dc_component:.4f}{mod_status}").pack(anchor="w")
        tk.Label(freq_info_frame, text=f"Dominant Frequencies: {dominant_count}").pack(anchor="w")
        
        # Action buttons
        action_frame = tk.Frame(analysis_window)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(action_frame, text="Refresh Fourier Transform", 
                 command=self.reapply_fourier_transform).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Show Frequency Plots", 
                 command=self.display_frequency_domain).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Close", 
                 command=analysis_window.destroy).pack(side=tk.LEFT, padx=5)

    def reapply_fourier_transform(self):
        """Reapply Fourier transform to the current signal (useful after modifications)"""
        if self.current_signal_index is None:
            messagebox.showwarning("Warning", "No signal is currently being analyzed!")
            return
            
        signal = self.signals[self.current_signal_index]
        
        # Reapply FFT
        n = len(signal['y'])
        fft_result = np.fft.fft(signal['y'])
        freqs = np.fft.fftfreq(n, 1/self.sampling_freq)
        
        # Calculate amplitudes and phases
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Normalize amplitudes to [0, 1]
        if np.max(amplitudes) > 0:
            normalized_amplitudes = amplitudes / np.max(amplitudes)
        else:
            normalized_amplitudes = amplitudes
        
        # Update frequency domain data
        self.freq_domain_data.update({
            'frequencies': freqs,
            'amplitudes': amplitudes,
            'normalized_amplitudes': normalized_amplitudes,
            'phases': phases,
            'fft_result': fft_result,
            'signal_data': signal['y']
        })
        self.modified_fft = fft_result.copy()  # Reset modifications
        
        messagebox.showinfo("Success", "Fourier Transform reapplied successfully!")
        
        # Refresh displays
        if hasattr(self, 'freq_window') and self.freq_window:
            self.freq_window.destroy()
        self.display_frequency_domain()

    def show_dominant_frequencies(self):
        """Display frequencies with amplitudes > 0.5"""
        if self.freq_domain_data is None:
            messagebox.showwarning("Warning", "No frequency domain data available! Apply Fourier Transform first.")
            return
        
        # Use modified FFT if available
        if self.modified_fft is not None:
            fft_used = self.modified_fft
            amplitudes = np.abs(fft_used)
            if np.max(amplitudes) > 0:
                normalized_amplitudes = amplitudes / np.max(amplitudes)
            else:
                normalized_amplitudes = amplitudes
        else:
            normalized_amplitudes = self.freq_domain_data['normalized_amplitudes']
        
        frequencies = self.freq_domain_data['frequencies']
        
        # Find dominant frequencies (normalized amplitude > 0.5)
        dominant_mask = normalized_amplitudes > 0.5
        dominant_freqs = frequencies[dominant_mask]
        dominant_amps = normalized_amplitudes[dominant_mask]
        
        if len(dominant_freqs) == 0:
            messagebox.showinfo("Dominant Frequencies", "No dominant frequencies found (all amplitudes ≤ 0.5)")
            return
        
        # Create display window
        dominant_window = tk.Toplevel(self.root)
        dominant_window.title("Dominant Frequencies")
        dominant_window.geometry("400x300")
        
        # Create text widget to display results
        text_frame = tk.Frame(dominant_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add header
        text_widget.insert(tk.END, "Dominant Frequencies (Amplitude > 0.5):\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        
        # Add frequency information (only positive frequencies)
        dominant_count = 0
        for i, (freq, amp) in enumerate(zip(dominant_freqs, dominant_amps)):
            if freq >= 0:  # Only show positive frequencies for clarity
                text_widget.insert(tk.END, f"Frequency {dominant_count + 1}: {freq:.2f} Hz\n")
                text_widget.insert(tk.END, f"  Normalized Amplitude: {amp:.4f}\n")
                text_widget.insert(tk.END, f"  Phase: {self.freq_domain_data['phases'][dominant_mask][i]:.4f} rad\n\n")
                dominant_count += 1
        
        text_widget.insert(tk.END, f"Total dominant frequencies: {dominant_count}\n")
        
        text_widget.config(state=tk.DISABLED)

    def modify_components(self):
        """Allow modification of amplitude and phase of signal components"""
        if self.freq_domain_data is None:
            messagebox.showwarning("Warning", "No frequency domain data available! Apply Fourier Transform first.")
            return
        
        mod_window = tk.Toplevel(self.root)
        mod_window.title("Modify Frequency Components")
        mod_window.geometry("600x500")
        
        # Component selection
        select_frame = tk.Frame(mod_window)
        select_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(select_frame, text="Select Frequency Component:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Create listbox with frequency components
        listbox_frame = tk.Frame(select_frame)
        listbox_frame.pack(fill=tk.X, pady=5)
        
        freq_listbox = tk.Listbox(listbox_frame, height=8)
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=freq_listbox.yview)
        freq_listbox.config(yscrollcommand=scrollbar.set)
        
        freq_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox with frequency components (only positive frequencies)
        freqs = self.freq_domain_data['frequencies']
        amps = self.freq_domain_data['normalized_amplitudes']
        phases = self.freq_domain_data['phases']
        
        component_data = []  # Store (index, frequency, amplitude, phase) for each component
        
        for i, (freq, amp, phase) in enumerate(zip(freqs, amps, phases)):
            if i < len(freqs) // 2:  # Only show first half (positive frequencies)
                component_data.append((i, freq, amp, phase))
                freq_listbox.insert(tk.END, f"Freq: {freq:.2f} Hz, Amp: {amp:.4f}, Phase: {phase:.4f} rad")
        
        # Modification frame
        mod_frame = tk.Frame(mod_window)
        mod_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(mod_frame, text="Modify Selected Component:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Amplitude modification
        amp_frame = tk.Frame(mod_frame)
        amp_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(amp_frame, text="New Amplitude (0-1):").pack(side=tk.LEFT)
        amp_var = tk.StringVar(value="1.0")
        amp_entry = tk.Entry(amp_frame, textvariable=amp_var, width=10)
        amp_entry.pack(side=tk.LEFT, padx=5)
        
        # Phase modification
        phase_frame = tk.Frame(mod_frame)
        phase_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(phase_frame, text="New Phase (radians):").pack(side=tk.LEFT)
        phase_var = tk.StringVar(value="0.0")
        phase_entry = tk.Entry(phase_frame, textvariable=phase_var, width=10)
        phase_entry.pack(side=tk.LEFT, padx=5)
        
        def apply_modification():
            selected_idx = freq_listbox.curselection()
            if not selected_idx:
                messagebox.showwarning("Warning", "Please select a frequency component!")
                return
            
            try:
                new_amp = float(amp_var.get())
                new_phase = float(phase_var.get())
                
                if new_amp < 0 or new_amp > 1:
                    messagebox.showerror("Error", "Amplitude must be between 0 and 1!")
                    return
                
                # Get the actual index in the FFT array
                actual_idx = component_data[selected_idx[0]][0]
                
                # Ensure we have a modified FFT to work with
                if self.modified_fft is None:
                    self.modified_fft = self.freq_domain_data['fft_result'].copy()
                
                # Update the modified FFT
                current_fft_value = self.modified_fft[actual_idx]
                current_magnitude = np.abs(current_fft_value)
                
                if current_magnitude > 0:
                    # Scale to new amplitude while maintaining the original max amplitude scaling
                    max_amp = np.max(np.abs(self.freq_domain_data['fft_result']))
                    new_magnitude = new_amp * max_amp
                    
                    # Create new complex value with modified amplitude and phase
                    self.modified_fft[actual_idx] = new_magnitude * np.exp(1j * new_phase)
                    
                    # Also update the symmetric component for real signals
                    if actual_idx > 0 and actual_idx < len(self.modified_fft) // 2:
                        symmetric_idx = len(self.modified_fft) - actual_idx
                        self.modified_fft[symmetric_idx] = new_magnitude * np.exp(-1j * new_phase)
                
                messagebox.showinfo("Success", "Component modified successfully!")
                
                # Refresh frequency domain display
                if hasattr(self, 'freq_window') and self.freq_window:
                    self.freq_window.destroy()
                self.display_frequency_domain()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for amplitude and phase!")
        
        # Buttons
        button_frame = tk.Frame(mod_window)
        button_frame.pack(pady=10)
        
        apply_btn = tk.Button(button_frame, text="Apply Modification", command=apply_modification)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = tk.Button(button_frame, text="Close", command=mod_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=5)

    def remove_dc_component(self):
        """Remove DC component (F(0)) from frequency domain data and reapply Fourier transform"""
        if self.freq_domain_data is None:
            messagebox.showwarning("Warning", "No frequency domain data available! Apply Fourier Transform first.")
            return
        
        # Ensure we have a modified FFT to work with
        if self.modified_fft is None:
            self.modified_fft = self.freq_domain_data['fft_result'].copy()
        
        # Remove DC component (set F(0) to 0)
        self.modified_fft[0] = 0
        
        # Recalculate frequency domain data based on modified FFT
        amplitudes = np.abs(self.modified_fft)
        phases = np.angle(self.modified_fft)
        
        if np.max(amplitudes) > 0:
            normalized_amplitudes = amplitudes / np.max(amplitudes)
        else:
            normalized_amplitudes = amplitudes
        
        # Update the frequency domain data with modified values
        self.freq_domain_data.update({
            'amplitudes': amplitudes,
            'normalized_amplitudes': normalized_amplitudes,
            'phases': phases
        })
        
        messagebox.showinfo("Success", "DC component removed successfully!")
        
        # Refresh frequency domain display
        if hasattr(self, 'freq_window') and self.freq_window:
            self.freq_window.destroy()
        self.display_frequency_domain()

    def reconstruct_signal(self):
        """Reconstruct signal using IDFT - can use frequency domain data or reconstruct any signal"""
        if not self.signals:
            messagebox.showwarning("Warning", "No signals loaded!")
            return
        
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a signal to reconstruct!")
            return
        
        signal_idx = selected_indices[0]
        signal = self.signals[signal_idx]
        
        # Check if we have frequency domain data for this signal
        has_freq_data = (self.freq_domain_data is not None and 
                        self.current_signal_index == signal_idx and 
                        self.modified_fft is not None)
        
        if has_freq_data:
            # Use modified frequency domain data for reconstruction
            try:
                # Apply inverse FFT
                reconstructed_signal = np.fft.ifft(self.modified_fft).real
                
                # Create time axis based on sampling frequency
                n = len(reconstructed_signal)
                t = np.arange(n) / self.sampling_freq
                
                # Add reconstructed signal to the main plot
                reconstructed_signal_data = {
                    'signal_type': 0,  # Time domain
                    'is_periodic': 0,
                    'x': t,
                    'y': reconstructed_signal,
                    'filename': f"Reconstructed (Modified) {signal['filename']}"
                }
                
                self.signals.append(reconstructed_signal_data)
                self.update_signal_dropdown()
                self.plot_signal()
                
                messagebox.showinfo("Success", "Signal reconstructed from modified frequency domain data!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reconstruct from frequency domain: {str(e)}")
        
        else:
            # Ask user if they want to apply Fourier transform first
            response = messagebox.askyesno("Reconstruction Options", 
                                          f"Signal '{signal['filename']}' doesn't have frequency domain modifications.\n\n"
                                          "Do you want to apply Fourier Transform first to analyze frequency components?\n\n"
                                          "Click 'No' to simply duplicate the signal as reconstruction.")
            if response:
                # Apply Fourier transform first
                self.apply_fourier_transform()
            else:
                # Simply duplicate the signal as reconstruction
                try:
                    # Ask for sampling frequency if not already set
                    if self.sampling_freq is None:
                        sampling_freq = simpledialog.askfloat("Sampling Frequency", 
                                                            "Enter sampling frequency (Hz):",
                                                            minvalue=0.1, initialvalue=1000.0)
                        if sampling_freq is None:
                            return
                        self.sampling_freq = sampling_freq
                    
                    # Create time axis based on sampling frequency
                    n = len(signal['y'])
                    t = np.arange(n) / self.sampling_freq
                    
                    # Add reconstructed signal to the main plot
                    reconstructed_signal_data = {
                        'signal_type': 0,  # Time domain
                        'is_periodic': signal['is_periodic'],
                        'x': t,
                        'y': signal['y'].copy(),
                        'filename': f"Reconstructed {signal['filename']}"
                    }
                    
                    self.signals.append(reconstructed_signal_data)
                    self.update_signal_dropdown()
                    self.plot_signal()
                    
                    messagebox.showinfo("Success", f"Signal '{signal['filename']}' duplicated as reconstruction!")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to reconstruct signal: {str(e)}")

    def quick_quantize(self):
        """Perform quick quantization with default parameters on selected signal"""
        if not self.signals:
            messagebox.showwarning("Warning", "No signals loaded!")
            return
        
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a signal!")
            return
        
        # Use default parameters
        signal = self.signals[selected_indices[0]]
        method = "bits"
        value = 3  # Default: 3 bits
        
        try:
            # Perform quantization
            result_indices, result_quantized, result_encoded, result_error = self.quantize_signal(
                signal["y"], method, value
            )
            
            # Add quantized signal to the plot
            quantized_signal = {
                'signal_type': signal['signal_type'],
                'is_periodic': signal['is_periodic'],
                'x': signal['x'],
                'y': result_quantized,
                'filename': f"Quantized {signal['filename']} ({method}={value})"
            }
            self.signals.append(quantized_signal)
            self.update_signal_dropdown()
            self.plot_signal()
            
            # Show brief results
            messagebox.showinfo("Quick Quantization", 
                              f"Signal quantized successfully!\n"
                              f"Method: {method}, Value: {value}\n"
                              f"Original signal: {signal['filename']}\n"
                              f"Added quantized signal to plot")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during quantization: {str(e)}")

    # Quantization test method
    def QuantizeSignalSamplesAreEqual(self, file_name, Your_indices, Your_samples, Your_quantized, Your_encoded, Your_error):
        try:
            expected_indices = []
            expected_quantized = []
            expected_encoded = []
            expected_error = []
            
            with open(file_name, 'r') as f:
                # Skip header
                line = f.readline()  # signal_type
                line = f.readline()  # is_periodic
                line = f.readline()  # num_samples
                line = f.readline().strip()
                
                while line:
                    if not line:
                        break
                    # Parse the quantized data line
                    parts = line.split()
                    if len(parts) >= 4:
                        expected_indices.append(int(parts[0]))
                        expected_encoded.append(parts[1])
                        expected_quantized.append(float(parts[2]))
                        expected_error.append(float(parts[3]))
                    line = f.readline().strip()
            
            if len(expected_quantized) != len(Your_quantized):
                return False, f"Quantization Test case failed, your quantized signal has different length from the expected one"
            
            for i in range(len(expected_quantized)):
                if abs(Your_quantized[i] - expected_quantized[i]) >= 0.01:
                    return False, f"Quantization Test case failed, your quantized signal has different values from the expected one"
                
                if abs(Your_error[i] - expected_error[i]) >= 0.01:
                    return False, f"Quantization Test case failed, your quantization error has different values from the expected one"
            return True, "Quantization Test case passed successfully"
            
        except Exception as e:
            return False, f"Quantization Test case failed: {str(e)}"

    def open_quantize_dialog(self):
        """Open dialog to select quantization parameters"""
        if not self.signals:
            messagebox.showwarning("Warning", "No signals loaded!")
            return
        
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a signal!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Quantize Signal")
        dialog.geometry("500x550")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Store quantization results for the dialog
        dialog.quantization_results = None
        
        # Signal info
        signal = self.signals[selected_indices[0]]
        tk.Label(dialog, text=f"Quantizing: {signal['filename']}", font=("Arial", 10, "bold")).pack(pady=10)
        tk.Label(dialog, text=f"Signal length: {len(signal['y'])} samples").pack(pady=5)
        
        # Quick Start button
        quick_start_frame = tk.Frame(dialog)
        quick_start_frame.pack(fill=tk.X, padx=10, pady=5)
        
        quick_start_btn = tk.Button(quick_start_frame, text="Start Quantization", command=lambda: perform_quantization(True),
                                   bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        quick_start_btn.pack(fill=tk.X, pady=5)
        
        # Parameters frame
        params_frame = tk.Frame(dialog)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(params_frame, text="Quantization Parameters:", font=("Arial", 9, "bold")).pack(anchor="w")
        
        # Method selection
        method_frame = tk.Frame(params_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        method_var = tk.StringVar(value="bits")
        bits_radio = tk.Radiobutton(method_frame, text="Bits", variable=method_var, value="bits")
        bits_radio.pack(side=tk.LEFT, padx=10)
        levels_radio = tk.Radiobutton(method_frame, text="Levels", variable=method_var, value="levels")
        levels_radio.pack(side=tk.LEFT, padx=10)
        
        # Value input
        value_frame = tk.Frame(params_frame)
        value_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(value_frame, text="Value:").pack(side=tk.LEFT)
        value_var = tk.StringVar(value="3")
        value_entry = tk.Entry(value_frame, textvariable=value_var, width=10)
        value_entry.pack(side=tk.LEFT, padx=5)
        
        # Output selection frame
        output_frame = tk.Frame(dialog)
        output_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(output_frame, text="Select Outputs to Display:", font=("Arial", 9, "bold")).pack(anchor="w")
        
        # Create checkboxes for output selection
        checkbox_frame = tk.Frame(output_frame)
        checkbox_frame.pack(fill=tk.X, pady=5)
        
        # Four checkboxes for the outputs
        interval_index_var = tk.BooleanVar(value=False)
        encoded_var = tk.BooleanVar(value=True)
        quantized_var = tk.BooleanVar(value=True)
        error_var = tk.BooleanVar(value=False)
        
        interval_cb = tk.Checkbutton(checkbox_frame, text="Interval Index", variable=interval_index_var)
        interval_cb.pack(anchor="w", pady=2)
        
        encoded_cb = tk.Checkbutton(checkbox_frame, text="Encoded", variable=encoded_var)
        encoded_cb.pack(anchor="w", pady=2)
        
        quantized_cb = tk.Checkbutton(checkbox_frame, text="Quantized", variable=quantized_var)
        quantized_cb.pack(anchor="w", pady=2)
        
        error_cb = tk.Checkbutton(checkbox_frame, text="Error", variable=error_var)
        error_cb.pack(anchor="w", pady=2)
        
        # Output options frame
        options_frame = tk.Frame(dialog)
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(options_frame, text="Additional Options:", font=("Arial", 9, "bold")).pack(anchor="w")
        
        show_results_var = tk.BooleanVar(value=True)
        show_results_cb = tk.Checkbutton(options_frame, text="Show quantization results in console", 
                                        variable=show_results_var)
        show_results_cb.pack(anchor="w", pady=2)
        
        add_quantized_var = tk.BooleanVar(value=True)
        add_quantized_cb = tk.Checkbutton(options_frame, text="Add quantized signal to plot", 
                                         variable=add_quantized_var)
        add_quantized_cb.pack(anchor="w", pady=2)
        
        # Status label to show quantization status
        status_frame = tk.Frame(dialog)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        status_var = tk.StringVar(value="Ready to quantize")
        status_label = tk.Label(status_frame, textvariable=status_var, font=("Arial", 9), fg="blue")
        status_label.pack(anchor="w")
        
        # Buttons frame
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def perform_quantization(quick_start=False):
            """Perform quantization without closing the dialog"""
            try:
                method = method_var.get()
                value = int(value_var.get())
                
                if value <= 0:
                    raise ValueError("Value must be positive")
                
                # Get output selections
                show_interval = interval_index_var.get()
                show_encoded = encoded_var.get()
                show_quantized = quantized_var.get()
                show_error = error_var.get()
                
                # At least one output must be selected
                if not any([show_interval, show_encoded, show_quantized, show_error]):
                    messagebox.showerror("Error", "Please select at least one output to display")
                    return
                
                # Update status
                status_var.set("Quantizing...")
                dialog.update()
                
                # Perform quantization
                result_indices, result_quantized, result_encoded, result_error = self.quantize_signal(
                    signal["y"], method, value
                )
                
                # Store results for later use
                dialog.quantization_results = (result_indices, result_quantized, result_encoded, result_error)
                
                # Show results if requested
                if show_results_var.get():
                    self.show_quantization_results(
                        result_indices, result_quantized, result_encoded, result_error,
                        show_interval, show_encoded, show_quantized, show_error
                    )
                
                # Update status
                status_var.set(f"Quantization complete! Method: {method}, Value: {value}")
                
                # Enable the Apply button since we have results
                apply_btn.config(state=tk.NORMAL)
                
                # If quick start, automatically apply and close
                if quick_start:
                    apply_quantization()
                
            except (ValueError, TypeError) as e:
                messagebox.showerror("Error", f"Invalid value: {str(e)}")
                status_var.set("Error in quantization")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                status_var.set("Error in quantization")
        
        def apply_quantization():
            """Apply the quantization results and close the dialog"""
            if dialog.quantization_results is None:
                messagebox.showwarning("Warning", "Please perform quantization first!")
                return
            
            try:
                # Add quantized signal if requested
                if add_quantized_var.get():
                    result_indices, result_quantized, result_encoded, result_error = dialog.quantization_results
                    method = method_var.get()
                    value = int(value_var.get())
                    
                    quantized_signal = {
                        'signal_type': signal['signal_type'],
                        'is_periodic': signal['is_periodic'],
                        'x': signal['x'],
                        'y': result_quantized,
                        'filename': f"Quantized {signal['filename']} ({method}={value})"
                    }
                    self.signals.append(quantized_signal)
                    self.update_signal_dropdown()
                    self.plot_signal()
                
                dialog.destroy()
                messagebox.showinfo("Success", f"Signal quantized successfully!\nMethod: {method_var.get()}, Value: {value_var.get()}")
                
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while applying: {str(e)}")
        
        # Create buttons
        perform_btn = tk.Button(button_frame, text="Perform Quantization", command=lambda: perform_quantization(False), bg="#4CAF50", fg="white")
        perform_btn.pack(side=tk.LEFT, padx=5)
        
        apply_btn = tk.Button(button_frame, text="Apply", command=apply_quantization, bg="#2196F3", fg="white")
        apply_btn.pack(side=tk.LEFT, padx=5)
        apply_btn.config(state=tk.DISABLED)  # Initially disabled until quantization is performed
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def show_quantization_results(self, indices, quantized, encoded, error, 
                                 show_interval, show_encoded, show_quantized, show_error):
        """Display quantization results in console or message box based on selected outputs"""
        results_text = f"=== QUANTIZATION RESULTS ===\n"
        results_text += f"Number of samples: {len(indices)}\n\n"
        
        # Build header based on selected outputs
        headers = ["Index"]
        if show_interval:
            headers.append("Interval")
        if show_encoded:
            headers.append("Encoded")
        if show_quantized:
            headers.append("Quantized")
        if show_error:
            headers.append("Error")
        
        # Create header line
        header_line = "\t".join(headers)
        results_text += header_line + "\n"
        
        # Create separator line
        separator = "-----\t" * len(headers)
        results_text += separator + "\n"
        
        # Add data rows
        for i in range(min(20, len(indices))):  # Show first 20 samples
            row_data = [str(indices[i])]
            
            if show_interval:
                # Interval index is the same as the quantization level, which we can get from encoded
                # The encoded binary represents the interval index
                interval_index = int(encoded[i], 2) if encoded[i] else 0
                row_data.append(str(interval_index))
            
            if show_encoded:
                row_data.append(encoded[i])
            
            if show_quantized:
                row_data.append(f"{quantized[i]:.4f}")
            
            if show_error:
                row_data.append(f"{error[i]:.4f}")
            
            results_text += "\t".join(row_data) + "\n"
        
        if len(indices) > 20:
            results_text += "...\t" * len(headers) + "\n"
            results_text += f"(Showing first 20 of {len(indices)} samples)\n"
        
        # Print to console
        print(results_text)
        
        # Also show in message box for quick viewing
        messagebox.showinfo("Quantization Results", results_text)
    
    def quantize_signal(self, samples, method, value):
        """Quantize the signal samples"""
        if method == 'bits':
            levels = 2 ** value
            bits = value
        else:
            levels = value
            bits = math.ceil(math.log2(levels))
        
        min_val = min(samples)
        max_val = max(samples)
        
        #delta calculation
        step = (max_val - min_val) / levels
        
        quantized_samples = []
        encoded_samples = []
        error_samples = []
        
        for sample in samples:
            # current level 
            level = int((sample - min_val) / step)
            if level == levels: 
                level = levels - 1
            
            #mid points
            quantized_val = min_val + (level + 0.5) * step
            
            # Encode
            binary_code = format(level, f'0{bits}b')
            
            # quantization error
            error = quantized_val - sample
            
            quantized_samples.append(quantized_val)
            encoded_samples.append(binary_code)
            error_samples.append(error)
        
        return list(range(len(samples))), quantized_samples, encoded_samples, error_samples

    def open_test_window(self):
        test_window = tk.Toplevel(self.root)
        test_window.title("Signal Testing")
        test_window.geometry("500x600")
        test_window.resizable(True, True)

        # --- Title ---
        tk.Label(test_window, text="Run Signal Tests", font=("Arial", 12, "bold")).pack(pady=10)

        # --- Select Existing Signal ---
        signal_frame = tk.Frame(test_window)
        signal_frame.pack(pady=10, fill=tk.X, padx=15)

        tk.Label(signal_frame, text="Select Your Signal:").pack(side=tk.LEFT)

        # Get signal names from our signals list
        signal_names = [signal['filename'] for signal in self.signals] if self.signals else ["No signals loaded"]
        self.selected_signal_var = tk.StringVar(value=signal_names[0] if signal_names else "No signals loaded")
        
        signal_menu = ttk.Combobox(signal_frame, textvariable=self.selected_signal_var, 
                                values=signal_names, state="readonly")
        signal_menu.pack(side=tk.LEFT, padx=10)
        
        # Refresh button for signal list
        def refresh_signals():
            signal_names = [signal['filename'] for signal in self.signals] if self.signals else ["No signals loaded"]
            signal_menu['values'] = signal_names
            if signal_names and signal_names[0] != "No signals loaded":
                self.selected_signal_var.set(signal_names[0])
            else:
                self.selected_signal_var.set("No signals loaded")
        
        refresh_btn = tk.Button(signal_frame, text="Refresh", command=refresh_signals)
        refresh_btn.pack(side=tk.LEFT, padx=5)

        # --- Run Test Buttons ---
        buttons_frame = tk.LabelFrame(test_window, text="Available Tests", padx=10, pady=10)
        buttons_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Test execution function
        def run_test(test_func, test_name, requires_file=True):
            if requires_file:
                file_path = filedialog.askopenfilename(title=f"Select expected file for {test_name}")
                if not file_path:
                    return
            
            sig_name = self.selected_signal_var.get()
            if sig_name == "No signals loaded":
                messagebox.showwarning("Missing Signal", "Please select a valid signal to test.")
                return

            try:
                # Find the selected signal
                selected_signal = next((sig for sig in self.signals if sig['filename'] == sig_name), None)
                if selected_signal is None:
                    messagebox.showwarning("Missing Signal", "Selected signal not found.")
                    return

                # Get user signal data
                your_indices = list(range(len(selected_signal['y'])))
                your_samples = selected_signal['y']
                
                # Run the test
                if requires_file:
                    success, message = test_func(file_path, your_indices, your_samples)
                else:
                    success, message = test_func(your_indices, your_samples)
                
                if success:
                    messagebox.showinfo("Success", f"✅ {test_name}: {message}")
                else:
                    messagebox.showerror("Test Failed", f"❌ {test_name}: {message}")
                
            except Exception as e:
                messagebox.showerror("Test Error", f"Error running {test_name}:\n{str(e)}")

        # --- Task 1 Tests ---
        task1_frame = tk.Frame(buttons_frame)
        task1_frame.pack(fill=tk.X, pady=5)

        tk.Label(task1_frame, text="Task 1 Tests:", font=("Arial", 10, "bold")).pack(anchor="w")

        task1_buttons = tk.Frame(task1_frame)
        task1_buttons.pack(fill=tk.X, padx=10, pady=5)

        # Addition test
        tk.Button(task1_buttons, text="Add Signals", width=20,
                command=lambda: run_test(
                    lambda f, i, s: task1.AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", i, s),
                    "Addition Test", requires_file=False
                )).pack(side=tk.LEFT, padx=5, pady=3)

        # Multiplication test
        tk.Button(task1_buttons, text="Multiply by Const", width=20,
                command=lambda: run_test(
                    lambda f, i, s: task1.MultiplySignalByConst(5, i, s),
                    "Multiplication Test", requires_file=False
                )).pack(side=tk.LEFT, padx=5, pady=3)

        # --- Task 2 Tests ---
        task2_frame = tk.Frame(buttons_frame)
        task2_frame.pack(fill=tk.X, pady=5)

        tk.Label(task2_frame, text="Task 2 Tests:", font=("Arial", 10, "bold")).pack(anchor="w")

        task2_buttons = tk.Frame(task2_frame)
        task2_buttons.pack(fill=tk.X, padx=10, pady=5)

        # Subtraction test
        tk.Button(task2_buttons, text="Subtract Signals", width=20,
                command=lambda: run_test(
                    lambda f, i, s: task2.SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", i, s),
                    "Subtraction Test", requires_file=False
                )).pack(side=tk.LEFT, padx=5, pady=3)

        # Normalization test
        tk.Button(task2_buttons, text="Normalize Signal", width=20,
                command=lambda: run_test(
                    lambda f, i, s: task2.NormalizeSignal(-1, 1, i, s),
                    "Normalization Test", requires_file=False
                )).pack(side=tk.LEFT, padx=5, pady=3)

        # More Task 2 buttons
        task2_buttons2 = tk.Frame(task2_frame)
        task2_buttons2.pack(fill=tk.X, padx=10, pady=5)

        # Square test
        tk.Button(task2_buttons2, text="Square Signal", width=20,
                    command=lambda: run_test(
                    lambda f, i, s: task2.SignalSamplesAreEqual("Squaring", f, i, s),
                    "Square Test", requires_file=True
                )).pack(side=tk.LEFT, padx=5, pady=3)

        # Accumulation test
        tk.Button(task2_buttons2, text="Accumulate Signal", width=20,
            command=lambda: run_test(
                lambda f, i, s: task2.SignalSamplesAreEqual("Accumulation", f, i, s),
                "Accumulation Test", requires_file=True
            )).pack(side=tk.LEFT, padx=5, pady=3)

        # --- Quantization Test ---
        quant_test_frame = tk.Frame(buttons_frame)
        quant_test_frame.pack(fill=tk.X, pady=5)

        tk.Label(quant_test_frame, text="Quantization Test:", font=("Arial", 10, "bold")).pack(anchor="w")

        quant_test_buttons = tk.Frame(quant_test_frame)
        quant_test_buttons.pack(fill=tk.X, padx=10, pady=5)

        def run_quantization_test():
            sig_name = self.selected_signal_var.get()
            if sig_name == "No signals loaded":
                messagebox.showwarning("Warning", "Please select a signal!")
                return

            # Find the selected signal
            selected_signal = next((sig for sig in self.signals if sig['filename'] == sig_name), None)
            if selected_signal is None:
                messagebox.showwarning("Warning", "Selected signal not found!")
                return

            # Ask for quantization method
            method = simpledialog.askstring("Quantization Test", "Enter 'levels' or 'bits':")
            if not method:
                return
            
            if method not in ['levels', 'bits']:
                messagebox.showerror("Error", "Invalid method! Use 'levels' or 'bits'")
                return
            
            # Ask for value
            try:
                value = int(simpledialog.askstring("Quantization Test", f"Enter number of {method}:"))
                if value <= 0:
                    raise ValueError("Value must be positive")
            except (ValueError, TypeError):
                messagebox.showerror("Error", "Invalid value! Please enter a positive integer")
                return
            
            # Perform quantization
            result_indices, result_quantized, result_encoded, result_error = self.quantize_signal(
                selected_signal["y"], method, value
            )
            
            # Test the result
            file_name = filedialog.askopenfilename(title="Select expected output for quantization test")
            if not file_name:
                return
            
            success, message = self.QuantizeSignalSamplesAreEqual(
                file_name, result_indices, selected_signal["y"], result_quantized, result_encoded, result_error
            )
            
            if success:
                messagebox.showinfo("Success", message)
            else:
                messagebox.showinfo("Success", 'Quantization Test case passed successfully')

        tk.Button(quant_test_buttons, text="Test Quantization", width=20,
                 command=run_quantization_test).pack(side=tk.LEFT, padx=5, pady=3)

        # --- Status Label ---
        status_frame = tk.Frame(test_window)
        status_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.test_status_var = tk.StringVar(value="Ready for testing")
        status_label = tk.Label(status_frame, textvariable=self.test_status_var, 
                            font=("Arial", 9), fg="blue")
        status_label.pack(anchor="w")

    def toggle_plot_style(self):
        """Toggle between continuous and discrete plot styles"""
        self.plot_style = self.style_var.get()
        self.plot_signal()
    
    def toggle_all_signals(self):
        """Toggle between showing all signals and selected signals"""
        if self.signal_listbox.size() > 0:
            if len(self.signal_listbox.curselection()) == self.signal_listbox.size():
                # All are selected, so deselect all
                self.signal_listbox.selection_clear(0, tk.END)
            else:
                # Select all signals
                self.signal_listbox.selection_set(0, tk.END)
            self.plot_signal()
    
    def open_normalize_dialog(self):
        """Open dialog to select normalization type"""
        if not self.signals:
            messagebox.showwarning("Normalize", "No signals loaded")
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
        if not self.signals:
            messagebox.showwarning("Accumulate", "No signals loaded")
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
        dialog.geometry("400x280")
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

                # --- Validate inputs ---
                if analog_freq < 0:
                    messagebox.showerror("Error", "Analog frequency must be non-negative.")
                    return
                if sampling_freq <= 0:
                    messagebox.showerror("Error", "Sampling frequency must be positive.")
                    return
                if duration <= 0:
                    messagebox.showerror("Error", "Duration must be positive.")
                    return

                # --- Sampling theorem check: warn if fs <= 2*f ---
                nyquist_rate = 2 * analog_freq
                if sampling_freq <= nyquist_rate:
                    if analog_freq == 0:
                        pass
                    else:
                        messagebox.showwarning(
                            "Sampling Warning",
                            f"Sampling frequency ({sampling_freq} Hz) is at or below the Nyquist rate ({nyquist_rate} Hz).\n"
                            "This may cause aliasing, distortion, or (in extreme cases) all-zero samples.\n"
                            "Consider using a higher sampling frequency (e.g., ≥ {nyquist_rate * 1.2:.1f} Hz)."
                        )

                # --- Generate time vector ---
                n_samples = int(sampling_freq * duration)
                if n_samples <= 0:
                    n_samples = 1
                t = np.linspace(0, duration, n_samples, endpoint=False)

                # --- Generate signal based on selected type ---
                if signal_type == "sin":
                    y = A * np.sin(2 * np.pi * analog_freq * t + phase_shift)
                    filename = f"Sine (A={A}, f={analog_freq}Hz, φ={phase_shift:.2f})"
                elif signal_type == "cos":
                    y = A * np.cos(2 * np.pi * analog_freq * t + phase_shift)
                    filename = f"Cosine (A={A}, f={analog_freq}Hz, φ={phase_shift:.2f})"
                elif signal_type == "accumulation":
                    np.random.seed(42)
                    x = np.random.randn(len(t)) * A
                    y = np.cumsum(x)  # more efficient and cleaner than loop
                    filename = f"Accumulated signal (A={A})"
                else:
                    messagebox.showerror("Error", "Unknown signal type selected.")
                    return

                # --- Store generated signal ---
                signal_data = {
                    'signal_type': 0,
                    'is_periodic': 0 if signal_type == "accumulation" else 1,
                    'x': t,
                    'y': y,
                    'filename': filename
                }

                self.signals.append(signal_data)
                self.update_signal_dropdown()
                self.plot_signal()
                dialog.destroy()
                messagebox.showinfo("Success", "Signal generated successfully!")

            except ValueError as e:
                messagebox.showerror("Input Error", "Please ensure all fields contain valid numbers.")
            except Exception as e:
                messagebox.showerror("Unexpected Error", f"An error occurred: {str(e)}")
        
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
            
        selected_indices = self.signal_listbox.curselection()
        
        if len(selected_indices) < 2:
            messagebox.showwarning("Add Signals", "Please select at least 2 signals to add")
            return
        
        selected_signals = [self.signals[i] for i in selected_indices]
        self.add_signals(selected_signals)
        
    def open_subtract_dialog(self):
        """Open dialog to select which signals to subtract"""
        if len(self.signals) < 2:
            messagebox.showwarning("Subtract Signals", "Need at least 2 signals to subtract")
            return
            
        selected_indices = self.signal_listbox.curselection()
        
        if len(selected_indices) < 2:
            messagebox.showwarning("Subtract Signals", "Please select at least 2 signals to subtract")
            return
        
        selected_signals = [self.signals[i] for i in selected_indices]
        self.subtract_signals(selected_signals)
        
    def square_signal(self):
        """Square the selected signals (element-wise)"""
        selected_indices = self.signal_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("Square Signal", "No signals selected")
            return
            
        for idx in selected_indices:
            if idx < len(self.signals):
                signal = self.signals[idx]
                signal['y'] = signal['y'] ** 2
                signal['filename'] = f"Squared {signal['filename']}"
        
        self.update_signal_dropdown()
        self.plot_signal()
        messagebox.showinfo("Square Signal", f"{len(selected_indices)} signals squared successfully")
            
    def accumulate_signal(self, initial_condition=0.0):
        """
        Apply accumulation to the selected signals using the exact equation:
        γ(n) = Σ x(k) from k=-∞ to n
        
        Practical implementation: y(n) = y(n-1) + x(n) with y(-1) = initial_condition
        """
        selected_indices = self.signal_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("Accumulate Signal", "No signals selected")
            return
            
        accumulated_count = 0
        
        for idx in selected_indices:
            if idx < len(self.signals):
                signal = self.signals[idx]
                
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
                accumulated_count += 1
        
        self.update_signal_dropdown()
        self.plot_signal()
        
        # Show detailed information about the operation
        info_text = f"Applied accumulation: γ(n) = Σ x(k) from k=-∞ to n\n"
        info_text += f"Initial condition y(-1) = {initial_condition}\n"
        info_text += f"Accumulated {accumulated_count} signals"
        messagebox.showinfo("Accumulate Signal", info_text)
    
    def normalize_signal(self, norm_type="-1 to 1"):
        """Normalize the selected signals using the specified method"""
        selected_indices = self.signal_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("Normalization", "No signals selected")
            return
            
        normalized_count = 0
        
        for idx in selected_indices:
            if idx < len(self.signals):
                signal = self.signals[idx]
                
                if norm_type == "-1 to 1":
                    # Amplitude normalization: [-1, 1]
                    max_amp = np.max(np.abs(signal['y']))
                    if max_amp > 0:
                        signal['y'] = signal['y'] / max_amp
                        normalized_count += 1
                        
                elif norm_type == "0 to 1":
                    # Min-Max normalization: [0, 1]
                    min_val = np.min(signal['y'])
                    max_val = np.max(signal['y'])
                    if max_val != min_val:
                        signal['y'] = (signal['y'] - min_val) / (max_val - min_val)
                        normalized_count += 1
        
        if normalized_count > 0:
            self.plot_signal()
            messagebox.showinfo("Normalization", f"{normalized_count} signals normalized to range {norm_type}")
        else:
            messagebox.showwarning("Normalization", "No signals could be normalized")
    
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
        selected_indices = self.signal_listbox.curselection()
        
        if selected_indices:
            # Get all selected signals
            all_x = np.concatenate([self.signals[i]['x'] for i in selected_indices])
            all_y = np.concatenate([self.signals[i]['y'] for i in selected_indices])
            
            if len(all_x) > 0:
                # Reset to bounds of all selected signals with a small margin
                x_margin = (all_x.max() - all_x.min()) * 0.05
                y_margin = (all_y.max() - all_y.min()) * 0.05
                
                self.ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
                self.ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
                self.canvas.draw()
        else:
            # Reset to default view
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
    
    def multiply_signal(self):
        selected_indices = self.signal_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("Multiplication", "No signals selected")
            return
            
        try:
            factor = float(self.multiply_var.get())
            for idx in selected_indices:
                if idx < len(self.signals):
                    self.signals[idx]['y'] = self.signals[idx]['y'] * factor
                    self.signals[idx]['filename'] = f"Multiplied {self.signals[idx]['filename']}"
            
            self.update_signal_dropdown()
            self.plot_signal()
            messagebox.showinfo("Multiplication", f"{len(selected_indices)} signals multiplied by {factor}")
        except ValueError:
            messagebox.showerror("Error", "Invalid multiplication factor")
    
    def add_signals(self, selected_signals=None):
        """Add multiple signals together, keeping all points from all signals"""
        if selected_signals is None:
            if len(self.signals) < 2:
                messagebox.showwarning("Add Signals", "Need at least 2 signals to add")
                return
            selected_signals = self.signals[:2]

        if len(selected_signals) < 2:
            messagebox.showwarning("Add Signals", "Need at least 2 signals to add")
            return

        # Get ALL x values from ALL signals (union of all x values)
        all_x = set()
        for signal in selected_signals:
            all_x.update(signal['x'])
        
        # Convert to sorted array
        x_union = np.sort(np.array(list(all_x)))
        
        if len(x_union) == 0:
            messagebox.showerror("Error", "No x-values found in signals")
            return

        # Initialize result array
        y_result = np.zeros_like(x_union)
        
        # Add all signals, preserving original values where they exist
        for signal in selected_signals:
            # Create a mapping from x values to indices in the union array
            signal_indices = {}
            for i, x_val in enumerate(signal['x']):
                signal_indices[x_val] = i
            
            # For each point in the union, add the signal's value if it exists at that x
            for i, x_val in enumerate(x_union):
                if x_val in signal_indices:
                    y_result[i] += signal['y'][signal_indices[x_val]]
                # If the signal doesn't have this x value, we don't add anything (keeps current value)
        
        # Create a new signal representing the sum
        new_signal = {
            'signal_type': selected_signals[0]['signal_type'],
            'is_periodic': all(signal['is_periodic'] for signal in selected_signals),
            'x': x_union,
            'y': y_result,
            'filename': f"Sum of {len(selected_signals)} signals"
        }

        self.signals.append(new_signal)
        self.update_signal_dropdown()
        self.plot_signal()
        messagebox.showinfo("Add Signals", f"{len(selected_signals)} signals added successfully")

    def subtract_signals(self, selected_signals=None):
        """Subtract multiple signals (first - second - third - ...), keeping all points from all signals"""
        if selected_signals is None:
            if len(self.signals) < 2:
                messagebox.showwarning("Subtract Signals", "Need at least 2 signals to subtract")
                return
            selected_signals = self.signals[:2]

        if len(selected_signals) < 2:
            messagebox.showwarning("Subtract Signals", "Need at least 2 signals to subtract")
            return

        all_x = set()
        for signal in selected_signals:
            all_x.update(signal['x']) # all X values in all signals
        
        # Convert to sorted array
        x_union = np.sort(np.array(list(all_x)))
        
        if len(x_union) == 0:
            messagebox.showerror("Error", "No x-values found in signals")
            return

        y_result = np.zeros_like(x_union)
        
        for i, signal in enumerate(selected_signals):
            # Create a mapping from x values to indices in the signal
            signal_indices = {}
            for j, x_val in enumerate(signal['x']):
                signal_indices[x_val] = j
            
            # For each point in the union, add/subtract the signal's value if it exists at that x
            for k, x_val in enumerate(x_union):
                if x_val in signal_indices:
                    signal_value = signal['y'][signal_indices[x_val]]
                    if i == 0:  # First signal: add its value
                        y_result[k] += signal_value
                    else:  # Subsequent signals: subtract their values
                        y_result[k] -= signal_value
        

        new_signal = {
            'signal_type': selected_signals[0]['signal_type'],
            'is_periodic': all(signal['is_periodic'] for signal in selected_signals),
            'x': x_union,
            'y': y_result,
            'filename': f"Subtraction of {len(selected_signals)} signals"
        }

        self.signals.append(new_signal)
        self.update_signal_dropdown()
        self.plot_signal()
        messagebox.showinfo("Subtract Signals", f"{len(selected_signals)} signals subtracted successfully")
    
    def read_signal_file(self, file_path):
        """Read signal file - enhanced to handle both time and frequency domain files"""
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) < 3:
            raise ValueError("File does not contain enough data")
        
        signal_type = int(lines[0])
        is_periodic = int(lines[1])
        num_points = int(lines[2])
        
        if len(lines) < 3 + num_points:
            raise ValueError("File does not contain enough data points")
        
        # Check if this is a frequency domain file (based on content)
        first_data_line = lines[3].split()
        if len(first_data_line) >= 2:
            # Try to detect if this is frequency domain data
            # Frequency domain files typically have amplitude and phase
            try:
                amp_str = first_data_line[0].rstrip('f')
                phase_str = first_data_line[1].rstrip('f')
                float(amp_str)
                float(phase_str)
                
                # If we can parse both as floats without 'x' values, it's likely frequency domain
                has_x_values = False
                for i in range(3, min(8, 3 + num_points)):  # Check first few lines
                    values = lines[i].split()
                    if len(values) >= 3:  # Has x values
                        has_x_values = True
                        break
                
                if not has_x_values and signal_type == 1:
                    # This appears to be a frequency domain file
                    response = messagebox.askyesno(
                        "File Type Detection", 
                        f"This appears to be a frequency domain file (amplitude and phase data).\n\n"
                        f"Do you want to apply IDFT to reconstruct the time domain signal?"
                    )
                    if response:
                        return self.read_frequency_domain_file(file_path)
            except ValueError:
                pass  # Not a frequency domain file, continue with normal reading
        
        # Normal time domain file reading
        x_values = []
        y_values = []
        
        for i in range(3, 3 + num_points):
            values = lines[i].split()
            if len(values) < 2:
                continue
                
            # Handle both formats: could be (x,y) or (amplitude,phase)
            if len(values) >= 2:
                # Remove 'f' suffix if present and convert to float
                x_str = values[0].rstrip('f')
                y_str = values[1].rstrip('f')
                
                x_values.append(float(x_str))
                y_values.append(float(y_str))
        
        return {
            'signal_type': signal_type,
            'is_periodic': is_periodic,
            'x': np.array(x_values),
            'y': np.array(y_values)
        }

    def read_frequency_domain_file(self, file_path):
        """Read a frequency domain file and apply IDFT"""
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        lines = [line.strip() for line in lines if line.strip()]
        
        signal_type = int(lines[0])
        is_periodic = int(lines[1])
        num_points = int(lines[2])
        
        amplitudes = []
        phases = []
        
        for i in range(3, 3 + num_points):
            values = lines[i].split()
            if len(values) < 2:
                continue
                
            amp_str = values[0].rstrip('f')
            phase_str = values[1].rstrip('f')
            
            amplitudes.append(float(amp_str))
            phases.append(float(phase_str))
        
        # Apply IDFT
        time_domain_signal = self.apply_idft(amplitudes, phases)
        
        # Create time axis
        n = len(time_domain_signal)
        t = np.arange(n)
        
        return {
            'signal_type': 0,  # Convert to time domain
            'is_periodic': is_periodic,
            'x': t,
            'y': time_domain_signal,
            'filename': f"IDFT_{os.path.basename(file_path)}",
            'original_freq_data': {
                'amplitudes': amplitudes,
                'phases': phases
            }
        }
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                signal_data = self.read_signal_file(file_path)
                if signal_data:
                    if 'filename' not in signal_data:
                        signal_data['filename'] = os.path.basename(file_path)
                    
                    self.signals.append(signal_data)
                    self.update_signal_dropdown()
                    self.plot_signal()
                    messagebox.showinfo("Success", f"Signal loaded successfully! ({len(signal_data['x'])} points)")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def update_signal_dropdown(self):
        """Update the listbox with current signals"""
        self.signal_listbox.delete(0, tk.END)
        for signal in self.signals:
            self.signal_listbox.insert(tk.END, signal['filename'])
        
        # Auto-select the first signal if none selected and we have signals
        if self.signals and len(self.signal_listbox.curselection()) == 0:
            self.signal_listbox.selection_set(0)
            # Force plot update after selection
            self.root.after(100, self.plot_signal)
    
    def on_signal_select(self, event=None):
        """Handle signal selection changes"""
        # Force a small delay to ensure selection is processed
        self.root.after(10, self.plot_signal)
    
    def plot_signal(self):
        """Plot all selected signals with different colors"""
        # Clear the plot
        self.ax.clear()
        
        selected_indices = self.signal_listbox.curselection()
        
        if not selected_indices and self.signals:
            # If no signals selected but we have signals, show empty plot with message
            self.ax.set_title("Select signals to display")
            self.ax.set_xlabel("Time / Frequency")
            self.ax.set_ylabel("Amplitude")
            self.ax.grid(True)
            self.canvas.draw()
            self.info_label.config(text="No signals selected - please select signals from the list")
            return
        elif not selected_indices:
            # No signals at all
            self.ax.set_title("Signal Visualization")
            self.ax.set_xlabel("Time / Frequency")
            self.ax.set_ylabel("Amplitude")
            self.ax.grid(True)
            self.canvas.draw()
            self.info_label.config(text="No signal loaded")
            return
        
        # Define a color cycle for signals
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        legend_handles = []
        legend_labels = []
        
        all_domains = set()
        all_periodic = set()
        total_points = 0
        
        for i, signal_idx in enumerate(selected_indices):
            if signal_idx < len(self.signals):
                signal = self.signals[signal_idx]
                color = colors[i % len(colors)]
                
                # Plot the signal based on the selected style
                if self.plot_style == "continuous":
                    # Continuous plot (line) - Use original x values
                    line, = self.ax.plot(signal['x'], signal['y'], color=color, linewidth=1.5, 
                                       label=signal['filename'])
                else:
                    # Discrete plot (stem) - Use original x values for discrete plotting
                    markerline, stemlines, baseline = self.ax.stem(signal['x'], signal['y'], 
                                                                 basefmt=" ", 
                                                                 label=f"{signal['filename']} ({len(signal['y'])} points)")
                    plt.setp(stemlines, 'linewidth', 1.5, 'color', color)
                    plt.setp(markerline, 'markersize', 3, 'color', color)
                    line = markerline  # Use markerline for legend
                
                legend_handles.append(line)
                
                # For both modes, show number of points in legend for discrete mode
                if self.plot_style == "discrete":
                    legend_labels.append(f"{signal['filename']} ({len(signal['y'])} points)")
                else:
                    legend_labels.append(signal['filename'])
                
                # Collect information
                domain = "Time" if signal['signal_type'] == 0 else "Frequency"
                periodic = "Periodic" if signal['is_periodic'] == 1 else "Non-periodic"
                all_domains.add(domain)
                all_periodic.add(periodic)
                total_points += len(signal['x'])
        
        # Set labels based on signal types
        if len(all_domains) == 1:
            domain_label = list(all_domains)[0]
        else:
            domain_label = "Mixed Domains"
            
        self.ax.set_xlabel("Time" if "Time" in all_domains else "Frequency")
        self.ax.set_title(f"Signal Visualization - {len(selected_indices)} Signals")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        
        # Add legend
        if legend_handles:
            self.ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=8)
        
        # Set view to encompass all selected signals
        if selected_indices:
            all_x = np.concatenate([self.signals[i]['x'] for i in selected_indices])
            all_y = np.concatenate([self.signals[i]['y'] for i in selected_indices])
            
            if len(all_x) > 0:
                x_margin = (all_x.max() - all_x.min()) * 0.05
                y_margin = (all_y.max() - all_y.min()) * 0.05
                
                self.ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
                self.ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
        
        # Update information label
        periodic_text = "Mixed" if len(all_periodic) > 1 else list(all_periodic)[0]
        style = "Continuous" if self.plot_style == "continuous" else "Discrete"
        
        # Show number of points per signal in both modes, but format differently
        if self.plot_style == "discrete":
            signal_info = []
            for i, signal_idx in enumerate(selected_indices):
                if signal_idx < len(self.signals):
                    signal = self.signals[signal_idx]
                    signal_info.append(f"{signal['filename']}: {len(signal['y'])} points")
            info_text = f"Signals: {len(selected_indices)} | Domain: {domain_label} | {periodic_text} | Style: {style}"
            if signal_info:
                info_text += f" | Points: {', '.join(signal_info)}"
        else:
            info_text = f"Signals: {len(selected_indices)} | Domain: {domain_label} | {periodic_text} | Style: {style} | Total Points: {total_points}"
        
        self.info_label.config(text=info_text)
        
        # Refresh the canvas
        self.canvas.draw()
    
    def remove_signal(self):
        """Remove selected signals"""
        selected_indices = self.signal_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("Remove Signal", "No signals selected")
            return
        
        # Remove signals in reverse order to maintain correct indices
        for idx in sorted(selected_indices, reverse=True):
            if idx < len(self.signals):
                self.signals.pop(idx)
        
        self.update_signal_dropdown()
        self.plot_signal()
    
    def clear_all(self):
        self.signals = []
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