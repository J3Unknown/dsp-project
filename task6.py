import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from math import isclose, sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from TestCases.Shifting_and_Folding import Shift_Fold_Signal


def read_signal_file(file_path):
    signal_data = {}
    signal_samples = []
    signal_indices = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            raise ValueError("File too short")
        
        signal_data['signal_type'] = int(lines[0].strip())
        signal_data['is_periodic'] = int(lines[1].strip())
        N = int(lines[2].strip())
        signal_data['N'] = N
        
        for i in range(3, 3 + N):
            parts = lines[i].strip().split()
            if len(parts) == 2:
                index = int(parts[0])
                amplitude = float(parts[1])
                signal_indices.append(index)
                signal_samples.append(amplitude)
            
        signal_data['indices'] = signal_indices
        signal_data['samples'] = signal_samples
        
        if len(signal_samples) != N:
            print(f"Warning: Read {len(signal_samples)} samples, expected {N}")
            
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")
        
    return signal_data

def write_signal_file(file_path, signal_data):
    try:
        with open(file_path, 'w') as f:
            f.write(f"{signal_data.get('signal_type', 0)}\n")
            f.write(f"{signal_data.get('is_periodic', 0)}\n")
            f.write(f"{signal_data.get('N', len(signal_data['samples']))}\n")
            
            indices = signal_data['indices']
            samples = signal_data['samples']
            
            for i in range(len(indices)):
                f.write(f"{indices[i]} {samples[i]:.6f}\n")
    except Exception as e:
        raise Exception(f"Error writing file {file_path}: {e}")

def compare_signals(file_name, your_indices, your_samples, task_name="Test case"):
    expected_indices = []
    expected_samples = []
    
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines[3:]:
                L = line.strip()
                if len(L.split()) == 2:
                    L = L.split()
                    V1 = int(L[0])
                    V2 = float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                else:
                    break
    except Exception as e:
        return False, f"Failed to read test file {file_name}: {e}"

    if len(expected_samples) != len(your_samples):
        msg = f"Test failed: Length mismatch (expected: {len(expected_samples)}, actual: {len(your_samples)})"
        return False, msg
            
    tolerance = 0.01
    for i in range(len(expected_samples)):
        if not isclose(your_samples[i], expected_samples[i], abs_tol=tolerance):
            msg = f"Test failed: Value mismatch at index [{your_indices[i]}]. Expected: {expected_samples[i]:.6f}, Actual: {your_samples[i]:.6f}"
            return False, msg
            
    msg = f"Test passed: {task_name} - All lengths, indices and values match (tolerance 0.01)."
    return True, msg

def smoothing_moving_average(x_samples, M):
    """Smoothing signal using moving average of M points"""
    N = len(x_samples)
    if M <= 0 or M % 2 == 0 or M > N:
        return x_samples[:]
    
    y_n = []
    half_window = M // 2
    
    for n in range(half_window, N - half_window):
        start_index = n - half_window
        end_index = n + half_window + 1
        window_sum = sum(x_samples[start_index:end_index])
        window_avg = window_sum / M
        y_n.append(window_avg)
    
    return y_n

def sharpening_derivative(x_samples):
    N = len(x_samples)
    if N < 2:
        return [], []
    
    first_deriv = [x_samples[n] - x_samples[n-1] for n in range(1, N)]
    
    second_deriv = []
    if N >= 3:
        second_deriv = [x_samples[n+1] - (2 * x_samples[n]) + x_samples[n-1] for n in range(1, N - 1)]
    
    return first_deriv, second_deriv

def shifting_signal(x_indices, x_samples, k):
    y_indices = [n + k for n in x_indices]
    y_samples = x_samples[:]
    return y_indices, y_samples

def folding_signal(x_indices, x_samples):
    mapping = {}
    for n, sample in zip(x_indices, x_samples):
        mapping[-n] = sample
    
    y_indices = sorted([-n for n in x_indices])
    y_samples = [mapping[idx] for idx in y_indices]
    
    return y_indices, y_samples

def remove_dc_component(x_indices, x_samples):
    if not x_samples:
        return [], [], 0.0
    
    dc_value = sum(x_samples) / len(x_samples)
    y_samples = [x - dc_value for x in x_samples]
    
    return x_indices, y_samples, dc_value

def linear_convolution(x_indices, x_samples, h_indices, h_samples):
    N = len(x_samples)
    M = len(h_samples)
    if N == 0 or M == 0:
        return [], []
    
    start_index_y = x_indices[0] + h_indices[0]
    end_index_y = x_indices[-1] + h_indices[-1]
    y_indices = list(range(start_index_y, end_index_y + 1))
    y_samples = [0.0] * len(y_indices)
    
    for i, n in enumerate(y_indices):
        for j, k in enumerate(x_indices):
            h_index = n - k
            if h_index in h_indices:
                h_val = h_samples[h_indices.index(h_index)]
                y_samples[i] += x_samples[j] * h_val
    
    return y_indices, y_samples

def compute_auto_correlation_direct(signal):
    N = len(signal)
    correlation = []
    
    for lag in range(N):
        correlation_sum = 0
        for n in range(N):
            if n + lag < N:
                correlation_sum += signal[n] * signal[n + lag]
        correlation.append(correlation_sum / N)
    
    full_correlation = list(reversed(correlation[1:])) + correlation
    lags = list(range(-(N-1), N))
    
    energy = sum(x**2 for x in signal)
    denominator = sqrt(energy * energy) / N if energy != 0 else 1
    normalized = [x / denominator for x in full_correlation] if denominator != 0 else full_correlation
    
    return full_correlation, normalized, lags

def compute_cross_correlation_advanced(signal1, signal2, is_periodic=False):
    N1 = len(signal1)
    N2 = len(signal2)
    
    if N1 == N2 and is_periodic:
        N = N1
        r12 = []
        P12 = []
        
        energy_x1 = sum(x**2 for x in signal1)
        energy_x2 = sum(x**2 for x in signal2)
        denominator = sqrt(energy_x1 * energy_x2) / N
        
        for j in range(N):
            correlation_sum = 0
            for n in range(N):
                x2_index = (n + j) % N
                correlation_sum += signal1[n] * signal2[x2_index]
            r12_j = correlation_sum / N
            
            P12_j = r12_j / denominator if denominator != 0 else 0
            
            r12.append(r12_j)
            P12.append(P12_j)
        
        lags = list(range(len(r12)))
        return r12, P12, lags, True
        
    else:
        total_points = N1 + N2 - 1
        
        correlation = []
        
        x1_padded = signal1 + [0] * (total_points - N1)
        x2_padded = signal2 + [0] * (total_points - N2)
        
        for lag in range(total_points):
            correlation_sum = 0
            for n in range(total_points):
                x2_index = (n + lag) % total_points
                correlation_sum += x1_padded[n] * x2_padded[x2_index]
            correlation.append(correlation_sum / total_points)
        
        lags = list(range(total_points))
        
        energy_x1 = sum(x**2 for x in signal1)
        energy_x2 = sum(x**2 for x in signal2)
        denominator = sqrt(energy_x1 * energy_x2) / total_points if (energy_x1 * energy_x2) > 0 else 1
        
        normalized = [x / denominator for x in correlation] if denominator != 0 else correlation
        
        return correlation, normalized, lags, False

def time_delay_analysis_advanced(sig1_samples, sig2_samples, Fs, is_periodic=False):
    if Fs <= 0:
        raise ValueError("Sampling frequency must be greater than zero")
    
    correlation, normalized, lags, _ = compute_cross_correlation_advanced(sig1_samples, sig2_samples, is_periodic)
    
    if not correlation:
        return 0.0, 0, 0.0, []
    
    max_corr = max(normalized, key=abs)
    max_index = normalized.index(max_corr)
    best_lag = lags[max_index]
    
    Ts = 1.0 / Fs
    time_delay = best_lag * Ts
    
    return time_delay, best_lag, max_corr, normalized

def normalized_cross_correlation(x_samples, y_samples, is_periodic=False):
    correlation, normalized, lags, _ = compute_cross_correlation_advanced(x_samples, y_samples, is_periodic)
    return lags, normalized

def time_delay_analysis(sig1_samples, sig2_samples, Fs):
    time_delay, best_lag, max_corr, _ = time_delay_analysis_advanced(sig1_samples, sig2_samples, Fs, False)
    return time_delay, best_lag

class DSPFramework(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Time Domain Framework")
        self.geometry("1400x900")
        
        self.current_signal_data = None
        self.secondary_signal_data = None
        self.Fs = 1000
        self.last_operation_name = ""
        self.correlation_results = None

        self.create_widgets()
        
    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Main Operations")
        
        correlation_frame = ttk.Frame(self.notebook)
        self.notebook.add(correlation_frame, text="Advanced Correlation")
        
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        plot_frame = ttk.Frame(main_frame, padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        test_frame = ttk.LabelFrame(self, text="Testing Area", padding="10")
        test_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.setup_control_frame(control_frame)
        self.setup_plot_frame(plot_frame)
        self.setup_correlation_frame(correlation_frame)
        self.setup_test_frame(test_frame)
        

    def setup_control_frame(self, frame):
        input_label_frame = ttk.LabelFrame(frame, text="Input & Parameters", padding="10")
        input_label_frame.pack(fill=tk.X, pady=5)

        ttk.Button(input_label_frame, text="Load Primary Signal (Sig1)", command=lambda: self.load_signal(1)).pack(fill=tk.X, pady=2)
        self.primary_file_label = ttk.Label(input_label_frame, text="Sig1: No file loaded")
        self.primary_file_label.pack(pady=2)

        ttk.Button(input_label_frame, text="Load Secondary Signal (Sig2)", command=lambda: self.load_signal(2)).pack(fill=tk.X, pady=2)
        self.secondary_file_label = ttk.Label(input_label_frame, text="Sig2: Not required")
        self.secondary_file_label.pack(pady=2)
        
        ttk.Label(input_label_frame, text="Sampling Freq. (Fs-Hz):").pack(pady=5)
        self.fs_entry = ttk.Entry(input_label_frame)
        self.fs_entry.insert(0, str(self.Fs))
        self.fs_entry.pack(fill=tk.X)
        self.fs_entry.bind('<Return>', self.update_fs)

        menu_label_frame = ttk.LabelFrame(frame, text="Time Domain Operations", padding="10")
        menu_label_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(menu_label_frame, text="Smoothing (Moving Avg) M:").pack(pady=5)
        self.M_entry = ttk.Entry(menu_label_frame)
        self.M_entry.insert(0, "3")
        self.M_entry.pack(fill=tk.X)
        ttk.Button(menu_label_frame, text="Compute Smoothing", command=self.run_smoothing).pack(fill=tk.X, pady=2)
        
        ttk.Button(menu_label_frame, text="Compute Derivatives", command=self.run_sharpening).pack(fill=tk.X, pady=5)

        ttk.Label(menu_label_frame, text="Shifting (k) & Folding:").pack(pady=5)
        self.K_entry = ttk.Entry(menu_label_frame)
        self.K_entry.insert(0, "50")
        self.K_entry.pack(fill=tk.X)
        ttk.Button(menu_label_frame, text="Run Shifting (k)", command=self.run_shifting).pack(fill=tk.X, pady=2)
        ttk.Button(menu_label_frame, text="Run Folding (x[-n])", command=self.run_folding).pack(fill=tk.X, pady=2)
        
        ttk.Button(menu_label_frame, text="Remove DC Component", command=self.run_dc_removal).pack(fill=tk.X, pady=5)
        
        ttk.Button(menu_label_frame, text="Run Convolution (Sig1 * Sig2)", command=self.run_convolution).pack(fill=tk.X, pady=5)
        ttk.Button(menu_label_frame, text="Auto-correlation (Sig1)", command=self.run_auto_correlation).pack(fill=tk.X, pady=2)
        ttk.Button(menu_label_frame, text="Cross-correlation (Sig1, Sig2)", command=self.run_correlation).pack(fill=tk.X, pady=2)
        ttk.Button(menu_label_frame, text="Time Delay Analysis", command=self.run_time_delay).pack(fill=tk.X, pady=2)

    def setup_plot_frame(self, frame):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def setup_correlation_frame(self, frame):
        control_corr_frame = ttk.Frame(frame, padding="10")
        control_corr_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        corr_label_frame = ttk.LabelFrame(control_corr_frame, text="Correlation Controls", padding="10")
        corr_label_frame.pack(fill=tk.X, pady=5)
        
        self.corr_type = tk.StringVar(value="cross")
        ttk.Radiobutton(corr_label_frame, text="Cross-correlation", variable=self.corr_type, value="cross").pack(anchor=tk.W)
        ttk.Radiobutton(corr_label_frame, text="Auto-correlation", variable=self.corr_type, value="auto").pack(anchor=tk.W)
        
        self.corr_method = tk.StringVar(value="direct")
        ttk.Radiobutton(corr_label_frame, text="Direct Method", variable=self.corr_method, value="direct").pack(anchor=tk.W)
        
        ttk.Label(corr_label_frame, text="Sampling Period (Ts):").pack(pady=5)
        self.ts_entry = ttk.Entry(corr_label_frame)
        self.ts_entry.insert(0, "1.0")
        self.ts_entry.pack(fill=tk.X)
        
        ttk.Button(corr_label_frame, text="Compute Advanced Correlation", command=self.run_advanced_correlation).pack(fill=tk.X, pady=5)
        ttk.Button(corr_label_frame, text="Show Time Delay", command=self.show_time_delay_advanced).pack(fill=tk.X, pady=2)
        
        results_corr_frame = ttk.LabelFrame(control_corr_frame, text="Results", padding="10")
        results_corr_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = tk.Text(results_corr_frame, height=15, width=50)
        scrollbar = ttk.Scrollbar(results_corr_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        plot_corr_frame = ttk.Frame(frame, padding="10")
        plot_corr_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig_corr, (self.ax_corr1, self.ax_corr2) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas_corr = FigureCanvasTkAgg(self.fig_corr, master=plot_corr_frame)
        self.canvas_corr_widget = self.canvas_corr.get_tk_widget()
        self.canvas_corr_widget.pack(fill=tk.BOTH, expand=True)

    def setup_test_frame(self, frame):
        ttk.Label(frame, text="Test Output File:").pack(side=tk.LEFT, padx=5)
        self.test_file_entry = ttk.Entry(frame, width=35)
        self.test_file_entry.insert(0, "path/to/expected_output.txt")
        self.test_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(frame, text="Browse", command=self.browse_test_file).pack(side=tk.LEFT, padx=5)
        
        self.test_button = ttk.Button(frame, text="Run Test Comparison", command=self.run_comparison, state=tk.DISABLED)
        self.test_button.pack(side=tk.LEFT, padx=5)

    def browse_test_file(self):
        file_path = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            self.test_file_entry.delete(0, tk.END)
            self.test_file_entry.insert(0, file_path)
            
    def update_fs(self, event=None):
        try:
            self.Fs = float(self.fs_entry.get())
            if self.Fs <= 0:
                raise ValueError
            self.title(f"DSP Time Domain Framework (Fs: {self.Fs} Hz)")
        except ValueError:
            messagebox.showerror("Invalid Input", "Sampling frequency must be a positive number")

    def load_signal(self, index):
        file_path = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                data = read_signal_file(file_path)
                if index == 1:
                    self.current_signal_data = data
                    self.primary_file_label.config(text=f"Sig1: {file_path.split('/')[-1]}")
                    self.plot_signal(data, "Primary Signal (Sig1)")
                elif index == 2:
                    self.secondary_signal_data = data
                    self.secondary_file_label.config(text=f"Sig2: {file_path.split('/')[-1]}")
                
                self.test_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                if index == 1:
                    self.current_signal_data = None
                if index == 2:
                    self.secondary_signal_data = None
                
    def plot_signal(self, data, title, is_result=False, secondary_data=None):
        self.ax.clear()
        
        if is_result and data:
            self.ax.stem(data['indices'], data['samples'], label=f'{title} Result', markerfmt='C2o', linefmt='C2-', basefmt=" ")

        elif data:
             self.ax.stem(data['indices'], data['samples'], label='Input Signal (Sig1)', markerfmt='C0o', linefmt='C0-')
        
        if secondary_data:
             self.ax.stem(secondary_data['indices'], secondary_data['samples'], label='Secondary Input (Sig2)', markerfmt='C1o', linefmt='C1-', basefmt=" ")

        self.ax.set_title(title)
        self.ax.set_xlabel("Sample Index (n)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def plot_correlation_results(self, correlation, normalized, lags, title):
        self.ax_corr1.clear()
        self.ax_corr2.clear()
        
        self.ax_corr1.stem(lags, correlation, basefmt=" ", linefmt='b-', markerfmt='bo')
        self.ax_corr1.set_title(f"{title} - Correlation")
        self.ax_corr1.set_xlabel('Lag')
        self.ax_corr1.set_ylabel('Correlation')
        self.ax_corr1.grid(True, alpha=0.3)
        
        self.ax_corr2.stem(lags, normalized, basefmt=" ", linefmt='r-', markerfmt='ro')
        self.ax_corr2.set_title(f"{title} - Normalized Correlation")
        self.ax_corr2.set_xlabel('Lag')
        self.ax_corr2.set_ylabel('Normalized Correlation')
        self.ax_corr2.set_ylim(-1.1, 1.1)
        self.ax_corr2.grid(True, alpha=0.3)
        
        self.fig_corr.tight_layout()
        self.canvas_corr.draw()
        
    def run_comparison(self):
        file_name = self.test_file_entry.get()
        if not self.current_signal_data or not file_name or file_name == "path/to/expected_output.txt":
            messagebox.showwarning("Warning", "Please select test file and compute result first")
            return

        your_indices = self.current_signal_data['indices']
        your_samples = self.current_signal_data['samples']
        
        is_passed, message = compare_signals(file_name, your_indices, your_samples, task_name=self.last_operation_name)

        print(message)
        if is_passed:
            messagebox.showinfo("Test Result - PASSED", message)
        else:
            messagebox.showerror("Test Result - FAILED", message)

    def run_smoothing(self):
        if not self.current_signal_data:
            messagebox.showerror("Error", "Please load primary signal first")
            return
        try:
            self.last_operation_name = "Smoothing (Moving Average)"
            M = int(self.M_entry.get())
            if M <= 0 or M % 2 == 0:
                raise ValueError("M must be positive odd integer")
                
            x_samples = self.current_signal_data['samples']
            x_indices = self.current_signal_data['indices']
            
            y_samples = smoothing_moving_average(x_samples, M)
            
            if not y_samples:
                messagebox.showwarning("Warning", f"Cannot smooth signal: M={M} is too large for signal length {len(x_samples)}")
                return
                
            half_window = M // 2
            y_indices = x_indices[half_window:len(x_indices) - half_window]
            
            self.current_signal_data = {
                'signal_type': 0,
                'is_periodic': 0,
                'N': len(y_samples),
                'indices': y_indices,
                'samples': y_samples
            }
            self.plot_signal(self.current_signal_data, f"Smoothing (M={M})", is_result=True)
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Smoothing failed: {e}")

    def run_sharpening(self):
        if not self.current_signal_data:
            messagebox.showerror("Error", "Please load primary signal first")
            return
        try:
            self.last_operation_name = "Sharpening (Derivative)"
            x_samples = self.current_signal_data['samples']
            x_indices = self.current_signal_data['indices']
            
            if len(x_samples) < 3:
                messagebox.showinfo("Result", "Signal too short for second derivative")
            
            first_deriv, second_deriv = sharpening_derivative(x_samples)

            if not first_deriv:
                return
            
            first_deriv_indices = x_indices[1:1 + len(first_deriv)]
            
            self.current_signal_data = {
                'signal_type': 0,
                'is_periodic': 0,
                'N': len(first_deriv),
                'indices': first_deriv_indices,
                'samples': first_deriv
            }
            self.plot_signal(self.current_signal_data, "First Derivative", is_result=True)

            if second_deriv:
                second_deriv_indices = x_indices[1:1 + len(second_deriv)]
                plt.figure()
                plt.stem(second_deriv_indices, second_deriv)
                plt.title("Second Derivative (Separate Plot)")
                plt.xlabel("Sample Index (n)")
                plt.ylabel("Amplitude")
                plt.grid(True)
                plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Derivative computation failed: {e}")

    def run_shifting(self):
        if not self.current_signal_data:
            messagebox.showerror("Error", "Please load primary signal first")
            return
        try:
            self.last_operation_name = "Shifting / Advancing"
            k = int(self.K_entry.get())
            x_indices = self.current_signal_data['indices']
            x_samples = self.current_signal_data['samples']
            
            y_indices, y_samples = shifting_signal(x_indices, x_samples, k)
            
            self.current_signal_data = {
                'signal_type': 0,
                'is_periodic': 0,
                'N': len(y_samples),
                'indices': y_indices,
                'samples': y_samples
            }
            
            op_type = "Delay" if k > 0 else "Advance"
            self.plot_signal(self.current_signal_data, f"Shifting (k={k}, {op_type})", is_result=True)
        except ValueError:
            messagebox.showerror("Invalid Input", "k value must be integer")
        except Exception as e:
            messagebox.showerror("Error", f"Shifting failed: {e}")

    def run_folding(self):
        if not self.current_signal_data:
            messagebox.showerror("Error", "Please load primary signal first")
            return
        try:
            self.last_operation_name = "Folding"
            x_indices = self.current_signal_data['indices']
            x_samples = self.current_signal_data['samples']
            
            y_indices, y_samples = folding_signal(x_indices, x_samples)
            
            self.current_signal_data = {
                'signal_type': 0,
                'is_periodic': 0,
                'N': len(y_samples),
                'indices': y_indices,
                'samples': y_samples
            }
            self.plot_signal(self.current_signal_data, "Folding (x[-n])", is_result=True)
        except Exception as e:
            messagebox.showerror("Error", f"Folding failed: {e}")

    def run_dc_removal(self):
        if not self.current_signal_data:
            messagebox.showerror("Error", "Please load primary signal first")
            return
        try:
            self.last_operation_name = "DC Removal"
            x_samples = self.current_signal_data['samples']
            x_indices = self.current_signal_data['indices']
            
            y_indices, y_samples, dc_val = remove_dc_component(x_indices, x_samples)
            
            self.current_signal_data = {
                'signal_type': 0,
                'is_periodic': self.current_signal_data['is_periodic'],
                'N': len(y_samples),
                'indices': y_indices,
                'samples': y_samples
            }
            self.plot_signal(self.current_signal_data, f"DC Removal (DC={dc_val:.4f})", is_result=True)
        except Exception as e:
            messagebox.showerror("Error", f"DC removal failed: {e}")

    def run_convolution(self):
        if not self.current_signal_data or not self.secondary_signal_data:
            messagebox.showerror("Error", "Please load both signals (Sig1 & Sig2) first")
            return
        try:
            self.last_operation_name = "Convolution"
            x1_data = self.current_signal_data
            x2_data = self.secondary_signal_data
            
            y_indices, y_samples = linear_convolution(
                x1_data['indices'], x1_data['samples'],
                x2_data['indices'], x2_data['samples']
            )
            
            self.current_signal_data = {
                'signal_type': 0,
                'is_periodic': 0,
                'N': len(y_samples),
                'indices': y_indices,
                'samples': y_samples
            }
            
            plt.figure()
            plt.stem(y_indices, y_samples)
            plt.title("Linear Convolution Result (Sig1 * Sig2)")
            plt.xlabel("Sample Index (n)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Convolution failed: {e}")

    def run_auto_correlation(self):
        if not self.current_signal_data:
            messagebox.showerror("Error", "Please load primary signal first")
            return
        try:
            self.last_operation_name = "Auto-correlation"
            x_samples = self.current_signal_data['samples']
            
            correlation, normalized, lags = compute_auto_correlation_direct(x_samples)
            
            self.correlation_results = {
                'correlation': correlation,
                'normalized': normalized,
                'lags': lags,
                'type': 'auto'
            }
            
            self.notebook.select(1)
            self.plot_correlation_results(correlation, normalized, lags, "Auto-correlation")
            
            self.display_correlation_results(correlation, normalized, lags, "Auto-correlation")
            
        except Exception as e:
            messagebox.showerror("Error", f"Auto-correlation failed: {e}")

    def run_correlation(self):
        if not self.current_signal_data or not self.secondary_signal_data:
            messagebox.showerror("Error", "Please load both signals (Sig1 & Sig2) first")
            return
        try:
            self.last_operation_name = "Normalized Cross-Correlation"
            x1_samples = self.current_signal_data['samples']
            x2_samples = self.secondary_signal_data['samples']
            
            is_periodic = bool(self.current_signal_data.get('is_periodic', 0))
            
            y_indices, y_samples = normalized_cross_correlation(
                x1_samples, x2_samples, is_periodic=is_periodic
            )
            
            self.current_signal_data = {
                'signal_type': 0,
                'is_periodic': 0,
                'N': len(y_samples),
                'indices': y_indices,
                'samples': y_samples
            }
            self.plot_signal(self.current_signal_data, "Normalized Cross-Correlation", is_result=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cross-correlation failed: {e}")

    def run_advanced_correlation(self):
        if self.corr_type.get() == "auto" and not self.current_signal_data:
            messagebox.showerror("Error", "Please load primary signal for Auto-correlation")
            return
        elif self.corr_type.get() == "cross" and (not self.current_signal_data or not self.secondary_signal_data):
            messagebox.showerror("Error", "Please load both signals for Cross-correlation")
            return
        
        try:
            if self.corr_type.get() == "auto":
                self.last_operation_name = "Advanced Auto-correlation"
                x_samples = self.current_signal_data['samples']
                is_periodic = bool(self.current_signal_data.get('is_periodic', 0))
                
                correlation, normalized, lags = compute_auto_correlation_direct(x_samples)
                corr_type_str = "Auto-correlation"
                
            else:
                self.last_operation_name = "Advanced Cross-correlation"
                x1_samples = self.current_signal_data['samples']
                x2_samples = self.secondary_signal_data['samples']
                is_periodic = bool(self.current_signal_data.get('is_periodic', 0))
                
                correlation, normalized, lags, _ = compute_cross_correlation_advanced(
                    x1_samples, x2_samples, is_periodic
                )
                corr_type_str = "Cross-correlation"
            
            self.correlation_results = {
                'correlation': correlation,
                'normalized': normalized,
                'lags': lags,
                'type': self.corr_type.get()
            }
            
            self.plot_correlation_results(correlation, normalized, lags, f"Advanced {corr_type_str}")
            self.display_correlation_results(correlation, normalized, lags, f"Advanced {corr_type_str}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Advanced correlation failed: {e}")

    def display_correlation_results(self, correlation, normalized, lags, title):
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"=== {title} Results ===\n\n")
        self.results_text.insert(tk.END, f"Total points: {len(correlation)}\n\n")
        
        self.results_text.insert(tk.END, "Lag\tCorrelation\tNormalized\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n")
        
        for i, lag in enumerate(lags):
            corr_val = correlation[i]
            norm_val = normalized[i]
            self.results_text.insert(tk.END, f"{lag}\t{corr_val:.6f}\t{norm_val:.6f}\n")

    def show_time_delay_advanced(self):
        if self.correlation_results is None:
            messagebox.showwarning("Warning", "Please compute correlation first")
            return
        
        try:
            Ts = float(self.ts_entry.get())
            if Ts <= 0:
                raise ValueError
                
            correlation = self.correlation_results['normalized']
            lags = self.correlation_results['lags']
            
            max_index = np.argmax(np.abs(correlation))
            max_lag = lags[max_index]
            max_value = correlation[max_index]
            
            time_delay = max_lag * Ts
            
            result_msg = f"Time Delay Analysis Results:\n\n"
            result_msg += f"Maximum normalized correlation: {max_value:.6f}\n"
            result_msg += f"Optimal lag: {max_lag}\n"
            result_msg += f"Sampling period (Ts): {Ts}\n"
            result_msg += f"Time delay = {max_lag} × {Ts} = {time_delay:.6f} seconds"
            
            messagebox.showinfo("Time Delay Analysis", result_msg)
            
            self.ax_corr2.plot(max_lag, max_value, 'go', markersize=10, label=f'Max at lag {max_lag}')
            self.ax_corr2.legend()
            self.canvas_corr.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid Sampling Period")
        except Exception as e:
            messagebox.showerror("Error", f"Time delay calculation failed: {e}")

    def run_time_delay(self):
        if not self.current_signal_data or not self.secondary_signal_data:
            messagebox.showerror("Error", "Please load both signals (Sig1 & Sig2) first")
            return
        try:
            self.last_operation_name = "Time Delay Analysis"
            self.update_fs()
            x1_samples = self.current_signal_data['samples']
            x2_samples = self.secondary_signal_data['samples']
            
            time_delay, max_lag_index = time_delay_analysis(x1_samples, x2_samples, self.Fs)

            result_message = (f"Time Delay Analysis Complete:\n"
                              f"  - Sampling Frequency (Fs): {self.Fs} Hz\n"
                              f"  - Optimal Lag (l): {max_lag_index} samples\n"
                              f"  - Approximate Time Delay: {time_delay:.6f} seconds")
                              
            messagebox.showinfo("Time Delay Analysis", result_message)
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Time delay analysis failed: {e}")

if __name__ == '__main__':
    app = DSPFramework()
    app.mainloop()