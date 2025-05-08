import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.integrate import quad
import sympy as sp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

class FunctionVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Function Visualizer")
        self.root.geometry("1000x800")
        
        # Set theme - try to use a more modern theme if available
        try:
            style = ttk.Style()
            style.theme_use('clam')  # Try to use a more modern built-in theme
            
            # Configure colors for a modern look
            style.configure('TFrame', background='#f0f0f0')
            style.configure('TLabelframe', background='#f0f0f0')
            style.configure('TLabelframe.Label', background='#f0f0f0', foreground='#333333', font=('Arial', 10, 'bold'))
            style.configure('TButton', background='#4a7abc', foreground='white', borderwidth=0, font=('Arial', 10))
            style.map('TButton', background=[('active', '#5c8eda')])
            style.configure('TLabel', background='#f0f0f0', foreground='#333333', font=('Arial', 10))
            style.configure('TEntry', fieldbackground='white', borderwidth=1)
            style.configure('TNotebook', background='#f0f0f0')
            style.configure('TNotebook.Tab', background='#e0e0e0', foreground='#333333', padding=[10, 4], font=('Arial', 10))
            style.map('TNotebook.Tab', background=[('selected', '#4a7abc')], foreground=[('selected', 'white')])
        except:
            pass  # If styling fails, use default
        
        # Set up sympy
        self.x_sym = sp.Symbol('x')
        
        # Error tracking flags
        self.error_shown = False
        self.last_error_message = ""
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create input frame
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Function Input", padding="10")
        self.input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Function input
        ttk.Label(self.input_frame, text="Function f(x):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.function_var = tk.StringVar(value="x**2")
        self.function_entry = ttk.Entry(self.input_frame, width=40, textvariable=self.function_var)
        self.function_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # X range inputs
        ttk.Label(self.input_frame, text="X Range:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        range_frame = ttk.Frame(self.input_frame)
        range_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(range_frame, text="From:").pack(side=tk.LEFT, padx=2)
        self.x_min_var = tk.StringVar(value="-10")
        ttk.Entry(range_frame, width=6, textvariable=self.x_min_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(range_frame, text="To:").pack(side=tk.LEFT, padx=2)
        self.x_max_var = tk.StringVar(value="10")
        ttk.Entry(range_frame, width=6, textvariable=self.x_max_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(range_frame, text="Points:").pack(side=tk.LEFT, padx=2)
        self.points_var = tk.StringVar(value="1000")
        ttk.Entry(range_frame, width=6, textvariable=self.points_var).pack(side=tk.LEFT, padx=2)
        
        # Derivative order
        ttk.Label(self.input_frame, text="Derivative Order:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.derivative_order_var = tk.IntVar(value=1)
        derivative_frame = ttk.Frame(self.input_frame)
        derivative_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Create spinbox for derivative order (1-5)
        ttk.Label(derivative_frame, text="Order:").pack(side=tk.LEFT, padx=2)
        self.derivative_spinbox = ttk.Spinbox(derivative_frame, from_=1, to=5, width=5, 
                                             textvariable=self.derivative_order_var)
        self.derivative_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Method selection for derivative calculation
        ttk.Label(derivative_frame, text="Method:").pack(side=tk.LEFT, padx=10)
        self.derivative_method_var = tk.StringVar(value="symbolic")
        ttk.Radiobutton(derivative_frame, text="Symbolic", 
                       variable=self.derivative_method_var, value="symbolic").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(derivative_frame, text="Numerical", 
                       variable=self.derivative_method_var, value="numerical").pack(side=tk.LEFT, padx=5)
        
        # Definite integral bounds
        ttk.Label(self.input_frame, text="Definite Integral:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        def_integral_frame = ttk.Frame(self.input_frame)
        def_integral_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(def_integral_frame, text="Lower bound:").pack(side=tk.LEFT, padx=2)
        self.lower_bound_var = tk.StringVar(value="-5")
        ttk.Entry(def_integral_frame, width=6, textvariable=self.lower_bound_var).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(def_integral_frame, text="Upper bound:").pack(side=tk.LEFT, padx=2)
        self.upper_bound_var = tk.StringVar(value="5")
        ttk.Entry(def_integral_frame, width=6, textvariable=self.upper_bound_var).pack(side=tk.LEFT, padx=2)
        
        # Button frame
        button_frame = ttk.Frame(self.input_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Visualize", command=self.visualize).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Graph", command=self.save_graphs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        
        # Create a tabbed interface for plots
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for each graph type
        self.func_tab = ttk.Frame(self.notebook)
        self.deriv_tab = ttk.Frame(self.notebook)
        self.indef_tab = ttk.Frame(self.notebook)
        self.def_tab = ttk.Frame(self.notebook)
        self.all_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.func_tab, text="Function")
        self.notebook.add(self.deriv_tab, text="Derivative")
        self.notebook.add(self.indef_tab, text="Indefinite Integral")
        self.notebook.add(self.def_tab, text="Definite Integral")
        self.notebook.add(self.all_tab, text="All Graphs")
        
        # Initialize figures and canvases for each tab
        # Function tab
        self.func_fig = plt.Figure(figsize=(9, 6), dpi=100)
        self.func_canvas = FigureCanvasTkAgg(self.func_fig, master=self.func_tab)
        self.func_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.func_toolbar = NavigationToolbar2Tk(self.func_canvas, self.func_tab)
        self.func_toolbar.update()
        self.func_plot = self.func_fig.add_subplot(111)
        
        # Derivative tab
        self.deriv_fig = plt.Figure(figsize=(9, 6), dpi=100)
        self.deriv_canvas = FigureCanvasTkAgg(self.deriv_fig, master=self.deriv_tab)
        self.deriv_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.deriv_toolbar = NavigationToolbar2Tk(self.deriv_canvas, self.deriv_tab)
        self.deriv_toolbar.update()
        self.deriv_plot = self.deriv_fig.add_subplot(111)
        
        # Indefinite integral tab
        self.indef_fig = plt.Figure(figsize=(9, 6), dpi=100)
        self.indef_canvas = FigureCanvasTkAgg(self.indef_fig, master=self.indef_tab)
        self.indef_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.indef_toolbar = NavigationToolbar2Tk(self.indef_canvas, self.indef_tab)
        self.indef_toolbar.update()
        self.indef_plot = self.indef_fig.add_subplot(111)
        
        # Definite integral tab
        self.def_fig = plt.Figure(figsize=(9, 6), dpi=100)
        self.def_canvas = FigureCanvasTkAgg(self.def_fig, master=self.def_tab)
        self.def_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.def_toolbar = NavigationToolbar2Tk(self.def_canvas, self.def_tab)
        self.def_toolbar.update()
        self.def_plot = self.def_fig.add_subplot(111)
        
        # All graphs tab
        self.all_fig = plt.Figure(figsize=(9, 8), dpi=100)
        self.all_canvas = FigureCanvasTkAgg(self.all_fig, master=self.all_tab)
        self.all_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.all_toolbar = NavigationToolbar2Tk(self.all_canvas, self.all_tab)
        self.all_toolbar.update()
        
        # Create 4 subplots for the "All Graphs" tab
        self.all_func_plot = self.all_fig.add_subplot(221)  # 2 rows, 2 columns, plot 1
        self.all_deriv_plot = self.all_fig.add_subplot(222)  # 2 rows, 2 columns, plot 2
        self.all_indef_plot = self.all_fig.add_subplot(223)  # 2 rows, 2 columns, plot 3
        self.all_def_plot = self.all_fig.add_subplot(224)   # 2 rows, 2 columns, plot 4
        
        # Set initial titles
        for fig in [self.func_fig, self.deriv_fig, self.indef_fig, self.def_fig, self.all_fig]:
            fig.tight_layout()

        # Initialize status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def show_error(self, message):
        """Show an error message, but only if it's not already shown."""
        if not self.error_shown or self.last_error_message != message:
            self.error_shown = True
            self.last_error_message = message
            messagebox.showerror("Error", message)
            # Reset error state when user clicks OK
            self.root.after(100, self.reset_error_state)
    
    def reset_error_state(self):
        """Reset the error state after user dismisses the error dialog."""
        self.error_shown = False
    
    def parse_function(self, func_str):
        """Parse the function string using sympy."""
        try:
            # Replace 'x' with 'self.x_sym' for sympy
            expr = sp.sympify(func_str)
            
            # Convert sympy expression to a numpy-compatible function
            func = sp.lambdify(self.x_sym, expr, 'numpy')
            return func, expr
        except Exception as e:
            self.show_error(f"Failed to parse function: {e}")
            return None, None
    
    def compute_derivative_symbolic(self, expr, x_vals, order=1):
        """Compute derivative using symbolic differentiation via sympy."""
        try:
            # Compute the nth derivative symbolically
            deriv_expr = expr
            for _ in range(order):
                deriv_expr = sp.diff(deriv_expr, self.x_sym)
            
            # Convert to numerical function
            deriv_func = sp.lambdify(self.x_sym, deriv_expr, 'numpy')
            
            # Evaluate at all points
            try:
                # Handle both scalar and array inputs
                if np.isscalar(x_vals):
                    return float(deriv_func(x_vals))
                else:
                    return np.array([float(deriv_func(x)) for x in x_vals])
            except:
                return np.full_like(x_vals, np.nan)
        except Exception as e:
            self.show_error(f"Failed to compute symbolic derivative: {e}")
            if np.isscalar(x_vals):
                return np.nan
            else:
                return np.full_like(x_vals, np.nan)
    
    def compute_derivative_numerical(self, func, x_vals, order=1):
        """Compute derivatives using finite difference approximation."""
        try:
            # Use different step sizes for different derivative orders
            h = 1e-3 if order <= 2 else 1e-2
            
            if np.isscalar(x_vals):
                x = x_vals
                if order == 1:
                    # First derivative: central difference
                    return (func(x + h) - func(x - h)) / (2 * h)
                elif order == 2:
                    # Second derivative: central difference
                    return (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)
                elif order == 3:
                    # Third derivative
                    return (func(x + 2*h) - 2 * func(x + h) + 2 * func(x - h) - func(x - 2*h)) / (2 * h**3)
                elif order == 4:
                    # Fourth derivative
                    return (func(x + 2*h) - 4*func(x + h) + 6*func(x) - 4*func(x - h) + func(x - 2*h)) / (h ** 4)
                elif order == 5:
                    # Fifth derivative
                    return (func(x + 3*h) - 5*func(x + 2*h) + 10*func(x + h) - 10*func(x - h) + 5*func(x - 2*h) - func(x - 3*h)) / (2 * h**5)
                else:
                    return np.nan
            else:
                # For array input, compute derivative at each point
                return np.array([self.compute_derivative_numerical(func, x, order) for x in x_vals])
        except Exception as e:
            if np.isscalar(x_vals):
                return np.nan
            else:
                return np.full_like(x_vals, np.nan)
    
    def compute_derivative(self, func, expr, x_vals, order=1, method="symbolic"):
        """Compute derivative using specified method."""
        if method == "symbolic":
            return self.compute_derivative_symbolic(expr, x_vals, order)
        else:
            return self.compute_derivative_numerical(func, x_vals, order)
    
    def compute_indefinite_integral(self, func, x_val, x_min):
        """Compute the indefinite integral (antiderivative) from x_min to x_val."""
        try:
            # Skip integration when x_val is very close to x_min to avoid numerical issues
            if abs(x_val - x_min) < 1e-10:
                return 0.0
                
            result, _ = quad(func, x_min, x_val)
            return result
        except Exception:
            return np.nan
    
    def compute_definite_integral(self, func, lower, upper):
        """Compute the definite integral from lower to upper bound."""
        try:
            result, _ = quad(func, lower, upper)
            return result
        except Exception as e:
            self.show_error(f"Failed to compute definite integral: {e}")
            return np.nan
    
    def visualize(self):
        """Visualize the function, its derivative, and integrals."""
        # Reset error state at the start of visualization
        self.error_shown = False
        self.status_var.set("Calculating...")
        self.root.update()
        
        try:
            func_str = self.function_var.get()
            
            try:
                x_min = float(self.x_min_var.get())
                x_max = float(self.x_max_var.get())
                num_points = int(self.points_var.get())
                lower_bound = float(self.lower_bound_var.get())
                upper_bound = float(self.upper_bound_var.get())
                deriv_order = int(self.derivative_order_var.get())
                deriv_method = self.derivative_method_var.get()
                
                # Validate derivative order
                if deriv_order < 1 or deriv_order > 5:
                    self.show_error("Derivative order must be between 1 and 5.")
                    self.status_var.set("Error: Invalid derivative order")
                    return
            except ValueError:
                self.show_error("Invalid input: Please enter valid numbers.")
                self.status_var.set("Error: Invalid inputs")
                return
            
            # Parse function
            func, expr = self.parse_function(func_str)
            if func is None or expr is None:
                self.status_var.set("Error: Invalid function")
                return
            
            # Generate x values
            x_vals = np.linspace(x_min, x_max, num_points)
            
            # Compute function values
            try:
                y_vals = func(x_vals)
                
                # Check for invalid values
                if np.isnan(y_vals).any() or np.isinf(y_vals).any():
                    self.show_error("Function evaluation resulted in invalid values (NaN or Infinity).")
                    self.status_var.set("Error: Invalid function values")
                    return
                    
            except Exception as e:
                self.show_error(f"Failed to evaluate function: {e}")
                self.status_var.set("Error: Evaluation failed")
                return
            
            # Compute derivatives using selected method
            try:
                deriv_vals = self.compute_derivative(func, expr, x_vals, deriv_order, deriv_method)
                
                # Check if all derivative calculations failed
                if np.isnan(deriv_vals).all():
                    self.show_error(f"Failed to compute {deriv_order}{self.get_ordinal_suffix(deriv_order)} derivative.")
                    self.status_var.set("Error: Derivative calculation failed")
                    return
            except Exception as e:
                self.show_error(f"Error computing derivative: {e}")
                self.status_var.set("Error: Derivative calculation failed")
                return
            
            # Compute indefinite integral (antiderivative)
            try:
                indef_integ_vals = np.array([self.compute_indefinite_integral(func, x, x_min) for x in x_vals])
                
                # Check if all integral calculations failed
                if np.isnan(indef_integ_vals).all():
                    self.show_error("Failed to compute indefinite integral.")
                    self.status_var.set("Error: Indefinite integral calculation failed")
                    return
            except Exception as e:
                self.show_error(f"Error computing indefinite integral: {e}")
                self.status_var.set("Error: Indefinite integral calculation failed")
                return
            
            # Compute definite integral
            try:
                def_integral_value = self.compute_definite_integral(func, lower_bound, upper_bound)
                if np.isnan(def_integral_value):
                    self.show_error("Failed to compute definite integral.")
                    self.status_var.set("Error: Definite integral calculation failed")
                    return
                    
                # Generate data for definite integral visualization
                # Create x values within the bounds
                if lower_bound >= x_min and upper_bound <= x_max:
                    # Get indices for the bounds within x_vals
                    lower_idx = np.abs(x_vals - lower_bound).argmin()
                    upper_idx = np.abs(x_vals - upper_bound).argmin()
                    
                    # Get x and y values within the bounds
                    x_bounds = x_vals[lower_idx:upper_idx+1]
                    y_bounds = y_vals[lower_idx:upper_idx+1]
                else:
                    # Handle case when bounds are outside the plotting range
                    x_bounds = np.linspace(max(lower_bound, x_min), min(upper_bound, x_max), 100)
                    y_bounds = func(x_bounds)
            except Exception as e:
                self.show_error(f"Error computing definite integral: {e}")
                self.status_var.set("Error: Definite integral calculation failed")
                return
            
            # Clear all plots
            self.func_plot.clear()
            self.deriv_plot.clear()
            self.indef_plot.clear()
            self.def_plot.clear()
            self.all_func_plot.clear()
            self.all_deriv_plot.clear()
            self.all_indef_plot.clear()
            self.all_def_plot.clear()
            
            # Set colors
            func_color = '#2271b3'  # Blue
            deriv_color = '#d62728'  # Red
            indef_color = '#2ca02c'  # Green
            fill_color = '#9ecae1'   # Light blue
            
            # Plot function (individual tab)
            self.func_plot.plot(x_vals, y_vals, color=func_color, linewidth=2.5)
            self.func_plot.set_title(f"Function: f(x) = {func_str}", fontsize=12, fontweight='bold')
            self.func_plot.grid(True, linestyle='--', alpha=0.7)
            self.func_plot.set_xlabel("x", fontsize=10)
            self.func_plot.set_ylabel("f(x)", fontsize=10)
            
            # Get ordinal suffix for derivative order
            ordinal_suffix = self.get_ordinal_suffix(deriv_order)
            
            # Plot derivative (individual tab)
            self.deriv_plot.plot(x_vals, deriv_vals, color=deriv_color, linewidth=2.5)
            method_name = "Symbolic" if deriv_method == "symbolic" else "Numerical"
            self.deriv_plot.set_title(f"{deriv_order}{ordinal_suffix} Derivative ({method_name})", fontsize=12, fontweight='bold')
            self.deriv_plot.grid(True, linestyle='--', alpha=0.7)
            self.deriv_plot.set_xlabel("x", fontsize=10)
            self.deriv_plot.set_ylabel(f"f{'^'*deriv_order}(x)", fontsize=10)
            
            # Plot indefinite integral (individual tab)
            self.indef_plot.plot(x_vals, indef_integ_vals, color=indef_color, linewidth=2.5)
            self.indef_plot.set_title(f"Indefinite Integral: ∫f(x)dx from {x_min} to x", fontsize=12, fontweight='bold')
            self.indef_plot.grid(True, linestyle='--', alpha=0.7)
            self.indef_plot.set_xlabel("x", fontsize=10)
            self.indef_plot.set_ylabel("∫f(x)dx", fontsize=10)
            
            # Plot definite integral (individual tab)
            self.def_plot.plot(x_vals, y_vals, color=func_color, linewidth=2.5)
            if lower_bound >= x_min and upper_bound <= x_max:
                # Fill the area under the curve between bounds
                self.def_plot.fill_between(x_bounds, y_bounds, alpha=0.4, color=fill_color)
                # Add a text annotation for the integral value
                self.def_plot.text(0.05, 0.95, 
                                   f"∫({lower_bound}, {upper_bound}) f(x)dx = {def_integral_value:.4f}", 
                                   transform=self.def_plot.transAxes,
                                   fontsize=10, fontweight='bold',
                                   bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
            self.def_plot.set_title(f"Definite Integral: ∫({lower_bound}, {upper_bound}) f(x)dx = {def_integral_value:.4f}",
                                     fontsize=12, fontweight='bold')
            self.def_plot.grid(True, linestyle='--', alpha=0.7)
            self.def_plot.set_xlabel("x", fontsize=10)
            self.def_plot.set_ylabel("f(x)", fontsize=10)
            
            # Plot all graphs in the "All Graphs" tab
            # Original function
            self.all_func_plot.plot(x_vals, y_vals, color=func_color, linewidth=1.5)
            self.all_func_plot.set_title(f"Function", fontsize=10, fontweight='bold')
            self.all_func_plot.grid(True, linestyle='--', alpha=0.7)
            self.all_func_plot.set_xlabel("x", fontsize=8)
            self.all_func_plot.set_ylabel("f(x)", fontsize=8)
            
            # Derivative
            self.all_deriv_plot.plot(x_vals, deriv_vals, color=deriv_color, linewidth=1.5)
            self.all_deriv_plot.set_title(f"{deriv_order}{ordinal_suffix} Derivative", fontsize=10, fontweight='bold')
            self.all_deriv_plot.grid(True, linestyle='--', alpha=0.7)
            self.all_deriv_plot.set_xlabel("x", fontsize=8)
            self.all_deriv_plot.set_ylabel(f"f{'^'*deriv_order}(x)", fontsize=8)
            
            # Indefinite integral
            self.all_indef_plot.plot(x_vals, indef_integ_vals, color=indef_color, linewidth=1.5)
            self.all_indef_plot.set_title(f"Indefinite Integral", fontsize=10, fontweight='bold')
            self.all_indef_plot.grid(True, linestyle='--', alpha=0.7)
            self.all_indef_plot.set_xlabel("x", fontsize=8)
            self.all_indef_plot.set_ylabel("∫f(x)dx", fontsize=8)
            
            # Definite integral
            self.all_def_plot.plot(x_vals, y_vals, color=func_color, linewidth=1.5)
            if lower_bound >= x_min and upper_bound <= x_max:
                # Fill the area under the curve between bounds
                self.all_def_plot.fill_between(x_bounds, y_bounds, alpha=0.4, color=fill_color)
            self.all_def_plot.set_title(f"Definite Integral: {def_integral_value:.4f}", fontsize=10, fontweight='bold')
            self.all_def_plot.grid(True, linestyle='--', alpha=0.7)
            self.all_def_plot.set_xlabel("x", fontsize=8)
            self.all_def_plot.set_ylabel("f(x)", fontsize=8)
            
            # Update figures
            for fig in [self.func_fig, self.deriv_fig, self.indef_fig, self.def_fig, self.all_fig]:
                fig.tight_layout()
                
            self.func_canvas.draw()
            self.deriv_canvas.draw()
            self.indef_canvas.draw()
            self.def_canvas.draw()
            self.all_canvas.draw()
            
            # Switch to derivative tab if user changed the order
            if deriv_order > 1:
                self.notebook.select(1)  # Select derivative tab
            
            self.status_var.set(f"Visualized f(x) = {func_str} with {deriv_order}{ordinal_suffix} derivative")
            
        except Exception as e:
            self.show_error(f"Visualization failed: {e}")
            self.status_var.set("Error: Visualization failed")
    
    def get_ordinal_suffix(self, n):
        """Return the ordinal suffix for a number (1st, 2nd, 3rd, etc.)."""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return suffix
    
    def save_graphs(self):
        """Save the current graph as an image file."""
        try:
            # Determine which tab is currently active to save that graph
            current_tab = self.notebook.index(self.notebook.select())
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Graph"
            )
            
            if not file_path:
                return
                
            # Save the appropriate figure based on the active tab
            if current_tab == 0:  # Function tab
                self.func_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 1:  # Derivative tab
                self.deriv_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 2:  # Indefinite integral tab
                self.indef_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 3:  # Definite integral tab
                self.def_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 4:  # All graphs tab
                self.all_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                
            messagebox.showinfo("Success", f"Graph saved to {file_path}")
            self.status_var.set(f"Saved graph to {os.path.basename(file_path)}")
        except Exception as e:
            self.show_error(f"Failed to save graph: {e}")
            self.status_var.set("Error: Failed to save graph")
    
    def clear(self):
        """Clear all inputs and graphs."""
        self.function_var.set("x**2")
        self.x_min_var.set("-10")
        self.x_max_var.set("10")
        self.points_var.set("1000")
        self.lower_bound_var.set("-5")
        self.upper_bound_var.set("5")
        self.derivative_order_var.set(1)
        self.derivative_method_var.set("symbolic")
        
        # Reset error flags
        self.error_shown = False
        self.last_error_message = ""
        
        # Clear all plots
        self.func_plot.clear()
        self.deriv_plot.clear()
        self.indef_plot.clear()
        self.def_plot.clear()
        self.all_func_plot.clear()
        self.all_deriv_plot.clear()
        self.all_indef_plot.clear()
        self.all_def_plot.clear()
        
        # Reset plot titles
        self.func_plot.set_title("Function")
        self.deriv_plot.set_title("Derivative")
        self.indef_plot.set_title("Indefinite Integral")
        self.def_plot.set_title("Definite Integral")
        
        # Update all canvases
        self.func_canvas.draw()
        self.deriv_canvas.draw()
        self.indef_canvas.draw()
        self.def_canvas.draw()
        self.all_canvas.draw()
        
        self.status_var.set("Ready")

def main():
    root = tk.Tk()
    app = FunctionVisualizer(root)
    
    # Configure error dialog to be modal
    root.option_add('*Dialog.msg.width', 300)
    
    # Set application icon if available
    try:
        root.iconbitmap("calc_icon.ico")
    except:
        pass
    
    # Handle window closing
    def on_closing():
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main() 