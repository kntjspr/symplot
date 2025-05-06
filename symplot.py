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
        self.root.geometry("1200x800")
        
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
        derivative_order_frame = ttk.Frame(self.input_frame)
        derivative_order_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        for i, val in enumerate([1, 2, 3]):
            ttk.Radiobutton(derivative_order_frame, text=f"{val}st" if val == 1 else f"{val}nd" if val == 2 else f"{val}rd", 
                            variable=self.derivative_order_var, value=val).pack(side=tk.LEFT, padx=10)
        
        # Button frame
        button_frame = ttk.Frame(self.input_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Visualize", command=self.visualize).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Graphs", command=self.save_graphs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        
        # Create frame for plots
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure and canvas for plotting
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Initialize plots
        self.orig_plot = self.fig.add_subplot(311)
        self.deriv_plot = self.fig.add_subplot(312)
        self.integ_plot = self.fig.add_subplot(313)
        
        self.orig_plot.set_title("Original Function")
        self.deriv_plot.set_title("Derivative")
        self.integ_plot.set_title("Integral")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
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
    
    def compute_derivative_sympy(self, expr, x_val, order=1):
        """Compute derivative using sympy's symbolic differentiation."""
        try:
            # Compute the nth derivative symbolically
            deriv_expr = expr
            for _ in range(order):
                deriv_expr = sp.diff(deriv_expr, self.x_sym)
            
            # Convert to numerical function
            deriv_func = sp.lambdify(self.x_sym, deriv_expr, 'numpy')
            
            # Evaluate at the point
            return deriv_func(x_val)
        except Exception:
            return np.nan
    
    def compute_derivative(self, func, x_val, order=1):
        """Compute the derivative using finite difference approximation."""
        try:
            # Use different step sizes for different derivative orders
            h = 1e-3 if order <= 2 else 1e-2
            
            if order == 1:
                # First derivative: central difference
                return (func(x_val + h) - func(x_val - h)) / (2 * h)
            elif order == 2:
                # Second derivative: central difference
                return (func(x_val + h) - 2 * func(x_val) + func(x_val - h)) / (h ** 2)
            elif order == 3:
                # Third derivative: central difference
                return (func(x_val + 2*h) - 2 * func(x_val + h) + 2 * func(x_val - h) - func(x_val - 2*h)) / (2 * h**3)
            else:
                return np.nan
        except Exception:
            return np.nan
    
    def compute_integral(self, func, x_val, x_min):
        """Compute the definite integral from x_min to x_val."""
        try:
            # Skip integration when x_val is very close to x_min to avoid numerical issues
            if abs(x_val - x_min) < 1e-10:
                return 0.0
                
            result, _ = quad(func, x_min, x_val)
            return result
        except Exception:
            return np.nan
    
    def visualize(self):
        """Visualize the function, its derivative, and integral."""
        # Reset error state at the start of visualization
        self.error_shown = False
        
        try:
            func_str = self.function_var.get()
            
            try:
                x_min = float(self.x_min_var.get())
                x_max = float(self.x_max_var.get())
                num_points = int(self.points_var.get())
            except ValueError:
                self.show_error("Invalid range or points: Please enter valid numbers.")
                return
                
            deriv_order = self.derivative_order_var.get()
            
            # Parse function
            func, expr = self.parse_function(func_str)
            if func is None or expr is None:
                return
            
            # Generate x values
            x_vals = np.linspace(x_min, x_max, num_points)
            
            # Compute function values
            try:
                y_vals = func(x_vals)
                
                # Check for invalid values
                if np.isnan(y_vals).any() or np.isinf(y_vals).any():
                    self.show_error("Function evaluation resulted in invalid values (NaN or Infinity).")
                    return
                    
            except Exception as e:
                self.show_error(f"Failed to evaluate function: {e}")
                return
            
            # Try to compute derivatives symbolically first, then fall back to numerical if needed
            try:
                # Try symbolic differentiation for higher precision
                if deriv_order <= 3:
                    # Compute the symbolic derivative
                    deriv_expr = expr
                    for _ in range(deriv_order):
                        deriv_expr = sp.diff(deriv_expr, self.x_sym)
                    
                    # Convert to numerical function
                    deriv_func = sp.lambdify(self.x_sym, deriv_expr, 'numpy')
                    
                    # Evaluate at all points
                    try:
                        # Make sure the input is properly vectorized
                        # This ensures the output has the same shape as x_vals
                        deriv_vals = np.array([float(deriv_func(x)) for x in x_vals])
                        
                        if np.isnan(deriv_vals).all() or np.isinf(deriv_vals).all():
                            raise ValueError("Symbolic differentiation failed")
                    except Exception:
                        # Fall back to numerical differentiation
                        deriv_vals = np.array([self.compute_derivative(func, x, deriv_order) for x in x_vals])
                else:
                    self.show_error(f"Derivative order {deriv_order} is not supported.")
                    return
                
                # Check if all derivative calculations failed
                if np.isnan(deriv_vals).all():
                    self.show_error(f"Failed to compute {deriv_order}{'st' if deriv_order == 1 else 'nd' if deriv_order == 2 else 'rd'} derivative.")
                    return
            except Exception as e:
                self.show_error(f"Error computing derivative: {e}")
                return
            
            # Compute integrals
            try:
                integ_vals = np.array([self.compute_integral(func, x, x_min) for x in x_vals])
                
                # Check if all integral calculations failed
                if np.isnan(integ_vals).all():
                    self.show_error("Failed to compute integral.")
                    return
            except Exception as e:
                self.show_error(f"Error computing integral: {e}")
                return
            
            # Clear previous plots
            self.orig_plot.clear()
            self.deriv_plot.clear()
            self.integ_plot.clear()
            
            # Plot function
            self.orig_plot.plot(x_vals, y_vals, 'b-')
            self.orig_plot.set_title(f"Original Function: f(x) = {func_str}")
            self.orig_plot.grid(True)
            
            # Plot derivative
            self.deriv_plot.plot(x_vals, deriv_vals, 'r-')
            self.deriv_plot.set_title(f"{deriv_order}{'st' if deriv_order == 1 else 'nd' if deriv_order == 2 else 'rd'} Derivative")
            self.deriv_plot.grid(True)
            
            # Plot integral
            self.integ_plot.plot(x_vals, integ_vals, 'g-')
            self.integ_plot.set_title(f"Integral: ∫f(x)dx from {x_min} to x")
            self.integ_plot.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Visualization failed: {e}")
    
    def save_graphs(self):
        """Save the current graphs as image files."""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Graphs"
            )
            
            if file_path:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Graphs saved to {file_path}")
        except Exception as e:
            self.show_error(f"Failed to save graphs: {e}")
    
    def clear(self):
        """Clear all inputs and graphs."""
        self.function_var.set("x**2")
        self.x_min_var.set("-10")
        self.x_max_var.set("10")
        self.points_var.set("1000")
        self.derivative_order_var.set(1)
        
        # Reset error flags
        self.error_shown = False
        self.last_error_message = ""
        
        # Clear plots
        self.orig_plot.clear()
        self.deriv_plot.clear()
        self.integ_plot.clear()
        
        self.orig_plot.set_title("Original Function")
        self.deriv_plot.set_title("Derivative")
        self.integ_plot.set_title("Integral")
        
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = FunctionVisualizer(root)
    
    # Configure error dialog to be modal
    root.option_add('*Dialog.msg.width', 300)
    
    # Handle window closing
    def on_closing():
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main() 