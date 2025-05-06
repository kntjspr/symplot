# Function Visualizer

A Python application for visualizing mathematical functions, their derivatives, and integrals using computational methods.

## Features

- Input custom mathematical functions using Python syntax
- Visualize the original function, its derivatives, and integrals
- Support for higher-order derivatives (up to 3rd order)
- Interactive plotting with zoom and pan capabilities
- Save visualizations as image files
- User-friendly Tkinter GUI

## Requirements

- Python 3.7 or higher
- Required packages: numpy, matplotlib, scipy, sympy, tkinter

## Installation

1. Clone this repository or download the source code
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python calc_visualizer.py
```

2. Enter a mathematical function in the text field (using Python syntax)
   - Examples: `x**2`, `sin(x)`, `exp(-x**2)`, `x**3 - 2*x + 5`
   
3. Set the x-range and number of points for visualization
   
4. Select the derivative order (1st, 2nd, or 3rd)
   
5. Click "Visualize" to display the graphs
   
6. Use the toolbar for zooming, panning, and other plot manipulations
   
7. Click "Save Graphs" to save the visualization as an image file

## Function Input Syntax

The application uses Python syntax for function input. Some examples:

- `x**2` for x²
- `x**3 - 2*x + 5` for x³ - 2x + 5
- `sin(x)` for sine of x
- `cos(x)` for cosine of x
- `exp(x)` for e^x
- `log(x)` for natural logarithm of x
- `tan(x)` for tangent of x
- `sqrt(x)` for square root of x

You can combine these functions using standard Python operators.

## Mathematical Methods

- **Differentiation**: Uses scipy's numerical differentiation with finite difference approximation
- **Integration**: Uses scipy's numerical integration with adaptive quadrature

## License

This project is open-source and available under the MIT License. 