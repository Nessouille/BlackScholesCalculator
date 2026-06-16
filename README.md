# BlackScholesCalculator
This project provides a simple Python implementation of the Black-Scholes option pricing model.

Files:
- `Main.py`: entry point that runs the calculator without logging.
- `Model.py`: computes call and put prices, validates inputs, and plots option price curves.
- `VannaVolga.py`: calculates Greeks and applies a Vanna-Volga smile correction. It is important in FX and options pricing because it adjusts Black-Scholes prices for the volatility smile observed in real markets.
- `DataLogger.py`: writes log events and errors to a file.

Libraries:
- `numpy` for numeric calculations.
- `scipy` for normal distribution functions and option Greeks.
- `pandas` for data handling in `Model.py`.
- `matplotlib` for plotting option price curves.

Improvements:
- separate user input, pricing logic, and plotting into clear modules;
- add tests for pricing and volatility correction functions;
- reduce assumptions about constant volatility and no dividends.
