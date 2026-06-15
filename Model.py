#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Black-Scholes Option Pricing Model
Author: Sandy Herho
License: WTFPL - Do What The F*** You Want To Public License
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import argparse
import time
import os

from DataLogger import log_event, log_error


POTENTIAL_ERRORS = [
    "ValueError: missing, non-numeric, or out-of-range option inputs",
    "ZeroDivisionError: zero volatility or zero time-to-maturity in the pricing formula",
    "FloatingPointError: invalid math domain values while computing d1 and d2",
    "OverflowError: extreme inputs during exponential or logarithmic calculations",
    "KeyboardInterrupt: user stops the interactive session",
    "OSError: log file or plot backend cannot be created or accessed",
]

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome():
    """Print a welcome message."""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "BLACK-SCHOLES OPTION CALCULATOR")
    print(" " * 30 + "by Sandy Herho")
    print(" " * 10 + "License: WTFPL - Do What The F*** You Want To Public License")
    print("=" * 80 + "\n")
    print("This calculator helps you price European call and put options using the")
    print("Black-Scholes model, a cornerstone of modern financial theory.\n")
    print("Example tickers and values:")
    print("  - AAPL (Apple): Stock Price $175.50, Volatility 25%")
    print("  - MSFT (Microsoft): Stock Price $420.00, Volatility 22%")
    print("  - TSLA (Tesla): Stock Price $175.00, Volatility 40%")
    print("  - SPY (S&P 500 ETF): Stock Price $510.00, Volatility 15%\n")
    time.sleep(1)

def setup_style(style="bmh"):
    """Set the plotting style."""
    try:
        plt.style.use(style)
    except:
        print(f"Style {style} not found. Using default style.")
        plt.style.use("default")


def get_potential_errors():
    """Return the common errors this calculator may encounter."""
    return list(POTENTIAL_ERRORS)


def validate_option_inputs(S, E, rf, sigma, T, t=0):
    """Validate option pricing inputs before running the formula."""
    if S <= 0:
        raise ValueError("Stock price must be greater than 0.")
    if E <= 0:
        raise ValueError("Strike price must be greater than 0.")
    if sigma <= 0:
        raise ValueError("Volatility must be greater than 0.")
    if T <= t:
        raise ValueError("Time to maturity must be greater than the current time.")
    if rf < 0:
        raise ValueError("Risk-free rate cannot be negative in this calculator.")

def call_option_price(S, E, rf, sigma, T, t=0, verbose=True):
    """
    Calculate the call option price using the Black-Scholes model.
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free rate (annual)
    sigma : float
        Volatility of the underlying stock (annual)
    T : float
        Time to maturity (in years)
    t : float, optional
        Current time, default is 0
    verbose : bool, optional
        Whether to print intermediate calculations
        
    Returns:
    --------
    float
        Call option price
    """
    validate_option_inputs(S, E, rf, sigma, T, t)

    # Calculate time to maturity
    tau = T - t
    
    # Calculate d1 and d2 parameters
    d1 = (np.log(S/E) + (rf + (sigma**2)/2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    if verbose:
        print(f"The d1 & d2 parameters: ({d1:.6f}, {d2:.6f})")
    
    # Calculate call option price
    return S*stats.norm.cdf(d1) - E*np.exp(-rf*tau)*stats.norm.cdf(d2)

def put_option_price(S, E, rf, sigma, T, t=0, verbose=True):
    """
    Calculate the put option price using the Black-Scholes model.
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free rate (annual)
    sigma : float
        Volatility of the underlying stock (annual)
    T : float
        Time to maturity (in years)
    t : float, optional
        Current time, default is 0
    verbose : bool, optional
        Whether to print intermediate calculations
        
    Returns:
    --------
    float
        Put option price
    """
    validate_option_inputs(S, E, rf, sigma, T, t)

    # Calculate time to maturity
    tau = T - t
    
    # Calculate d1 and d2 parameters
    d1 = (np.log(S/E) + (rf + (sigma**2)/2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    if verbose:
        print(f"The d1 & d2 parameters: ({d1:.6f}, {d2:.6f})")
    
    # Calculate put option price
    return E*np.exp(-rf*tau)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)

def plot_option_prices(S_range, E, rf, sigma, T, t=0):
    """
    Plot call and put option prices for a range of stock prices.
    
    Parameters:
    -----------
    S_range : array-like
        Range of stock prices to plot
    E : float
        Strike price
    rf : float
        Risk-free rate (annual)
    sigma : float
        Volatility of the underlying stock (annual)
    T : float
        Time to maturity (in years)
    t : float, optional
        Current time, default is 0
    """
    call_prices = [call_option_price(S, E, rf, sigma, T, t, verbose=False) for S in S_range]
    put_prices = [put_option_price(S, E, rf, sigma, T, t, verbose=False) for S in S_range]
    
    plt.figure(figsize=(12, 7))
    plt.plot(S_range, call_prices, 'b-', linewidth=2.5, label='Call Option Price')
    plt.plot(S_range, put_prices, 'r-', linewidth=2.5, label='Put Option Price')
    plt.axvline(x=E, color='g', linestyle='--', linewidth=1.5, label='Strike Price')
    
    # Highlight the current stock price
    if S_range[0] <= S_range[-1]:
        plt.axvline(x=S_range[0], color='orange', linestyle=':', linewidth=1.5, label='Current Stock Price')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Stock Price ($)', fontsize=12)
    plt.ylabel('Option Price ($)', fontsize=12)
    plt.title('Black-Scholes Option Prices', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    
    # Add text annotations
    info_text = f"Strike: ${E:.2f}\nRisk-free rate: {rf*100:.2f}%\nVolatility: {sigma*100:.2f}%\nMaturity: {T:.2f} years"
    plt.annotate(info_text, xy=(0.02, 0.97), xycoords='axes fraction', 
                 fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def get_user_input(prompt, example, input_type=float):
    """
    Get user input with validation.
    
    Parameters:
    -----------
    prompt : str
        The prompt to display to the user
    example : any
        An example value to show the user
    input_type : type
        The type to convert the input to
        
    Returns:
    --------
    input_type
        The validated user input
    """
    while True:
        try:
            user_input = input(f"{prompt} (example: {example}): ")
            if user_input == "":
                print(f"Please enter a value. '{example}' is just an example.")
                continue
            return input_type(user_input)
        except ValueError:
            print(f"Error: Please enter a valid {input_type.__name__} value.")

def get_yes_no_input(prompt):
    """
    Get a yes/no input from the user.
    
    Parameters:
    -----------
    prompt : str
        The prompt to display to the user
        
    Returns:
    --------
    bool
        True for 'yes', False for 'no'
    """
    while True:
        user_input = input(f"{prompt} (y/n): ").strip().lower()
        if user_input in ["y", "yes"]:
            return True
        if user_input in ["n", "no"]:
            return False
        print("Error: Please enter 'y' or 'n'.")

def display_parameter_explanation():
    """Display explanations for the different parameters."""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 30 + "PARAMETER EXPLANATIONS")
    print("=" * 80 + "\n")
    
    explanations = [
        ("Stock Price (S)", "The current market price of the underlying asset.", "For AAPL: $175.50, For MSFT: $420.00, For TSLA: $175.00"),
        ("Strike Price (E)", "The price at which the option holder can buy (call) or sell (put) the underlying asset.", "Often set at intervals like $170.00, $175.00, $180.00 for major stocks"),
        ("Risk-Free Rate (rf)", "The interest rate available on a riskless investment over the life of the option.", "Current 3-month Treasury yield: around 4.75%"),
        ("Volatility (σ)", "A measure of the expected fluctuation in the underlying asset's price.", "Low volatility stock: 15.00%, Average: 25.00%, High tech stock: 40.00%"),
        ("Time to Maturity (T)", "The time remaining until the option expires, expressed in years.", "Weekly options: 0.02 (1 week), Monthly: 0.08 (1 month), Quarterly: 0.25 (3 months)")
    ]
    
    for param, explanation, examples in explanations:
        print(f"{param}:")
        print(f"  {explanation}")
        print(f"  Examples: {examples}\n")
    
    input("\nPress Enter to continue...")

def display_bs_explanation():
    """Display an explanation of the Black-Scholes model."""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "BLACK-SCHOLES MODEL EXPLAINED")
    print("=" * 80 + "\n")
    
    explanation = """The Black-Scholes model is a mathematical model used to determine the theoretical price 
of European-style options. Developed by Fischer Black and Myron Scholes in 1973, it 
revolutionized the field of finance and earned Scholes and Robert Merton (who extended 
the model) the Nobel Prize in Economics in 1997.

The model makes several key assumptions:
1. The stock follows a lognormal distribution of prices
2. The risk-free rate and volatility of the stock are known and constant
3. No dividends are paid during the option's life
4. Markets are efficient (no arbitrage opportunities)
5. No transaction costs or taxes
6. The options are European style (exercisable only at expiration)

The formula for a call option price is:
C = S·N(d₁) - E·e^(-r·T)·N(d₂)

And for a put option price:
P = E·e^(-r·T)·N(-d₂) - S·N(-d₁)

Where:
- N(x) is the cumulative distribution function of the standard normal distribution
- d₁ = [ln(S/E) + (r + σ²/2)·T] / (σ·√T)
- d₂ = d₁ - σ·√T

Despite its limitations, the Black-Scholes model remains a foundational tool in options 
pricing and risk management."""
    
    print(explanation)
    input("\nPress Enter to continue...")

def interactive_menu(enable_logging=True):
    """Interactive menu for the Black-Scholes calculator."""
    print_welcome()
    if enable_logging:
        log_event("Interactive menu started")
    
    while True:
        try:
            print("\n" + "-" * 80)
            print("MAIN MENU:")
            print("1. Calculate Option Prices")
            print("2. Parameter Explanations")
            print("3. About the Black-Scholes Model")
            print("4. Exit")
            print("-" * 80)
            
            choice = input("\nEnter your choice (1-4): ")
        
            if choice == "1":
                # Collect input parameters
                clear_screen()
                print("\n" + "=" * 80)
                print(" " * 25 + "OPTION PRICING CALCULATOR")
                print("=" * 80 + "\n")
                print("Enter your parameters (examples are provided for guidance):\n")
                
                # Example values for AAPL (Apple Inc.)
                stock_price = get_user_input("Current Stock Price ($)", "175.50")
                strike_price = get_user_input("Strike Price ($)", "180.00")
                risk_free_rate = get_user_input("Risk-Free Rate (%)", "4.75") / 100
                volatility = get_user_input("Volatility (%)", "25.30") / 100
                time_to_maturity = get_user_input("Time to Maturity (years)", "0.25")
                
                # Calculate and display results
                print("\nCalculating option prices...\n")
                time.sleep(0.5)
                
                try:
                    call_price = call_option_price(stock_price, strike_price, risk_free_rate, 
                                                 volatility, time_to_maturity)
                    
                    put_price = put_option_price(stock_price, strike_price, risk_free_rate, 
                                               volatility, time_to_maturity)
                except Exception as error:
                    print(f"\nError while calculating option prices: {error}")
                    if enable_logging:
                        log_error("Option price calculation failed", error, {
                            "stock_price": stock_price,
                            "strike_price": strike_price,
                            "risk_free_rate": risk_free_rate,
                            "volatility": volatility,
                            "time_to_maturity": time_to_maturity,
                        })
                    continue
                
                print("\n" + "-" * 50)
                print("RESULTS:")
                print("-" * 50)
                print(f"Stock Price: ${stock_price:.2f}")
                print(f"Strike Price: ${strike_price:.2f}")
                print(f"Risk-free Rate: {risk_free_rate*100:.2f}%")
                print(f"Volatility: {volatility*100:.2f}%")
                print(f"Time to Maturity: {time_to_maturity:.2f} years")
                print("-" * 50)
                print(f"Call Option Price: ${call_price:.4f}")
                print(f"Put Option Price: ${put_price:.4f}")
                print("-" * 50)
                
                # Check if call option is in/out of the money
                if stock_price > strike_price:
                    print("Call Option: In-the-money")
                    print("Put Option: Out-of-the-money")
                elif stock_price < strike_price:
                    print("Call Option: Out-of-the-money")
                    print("Put Option: In-the-money")
                else:
                    print("Both options: At-the-money")
                print("-" * 50)
                
                # Offer to view a plot
                if get_yes_no_input("\nWould you like to see a price graph?"):
                    # Create a range of stock prices for plotting
                    price_range = max(stock_price, strike_price) * 0.5
                    S_range = np.linspace(stock_price - price_range, stock_price + price_range, 100)
                    print("\nGenerating plot...\n")
                    plot_option_prices(S_range, strike_price, risk_free_rate, volatility, time_to_maturity)
                
                # Offer to calculate another set of prices
                if get_yes_no_input("\nWould you like to calculate with different parameters?"): 
                    continue
                
            elif choice == "2":
                display_parameter_explanation()
                
            elif choice == "3":
                display_bs_explanation()
                
            elif choice == "4":
                print("\nThank you for using the Black-Scholes Option Calculator!")
                print("Exiting...\n")
                break
                
            else:
                print("\nInvalid choice. Please enter a number between 1 and 4.")
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
            if enable_logging:
                log_error("Interactive session interrupted by user", KeyboardInterrupt())
            break

def parse_args_or_use_defaults():
    """Parse command line arguments or use defaults if in interactive environment."""
    # Check if we're in a Jupyter notebook or IPython environment
    is_notebook = 'ipykernel' in sys.modules
    
    if is_notebook or len(sys.argv) <= 1:
        # Default to interactive mode in notebook environment
        class Args:
            pass
        
        args = Args()
        args.interactive = True
        args.stock_price = None  # These will be requested interactively
        args.strike_price = None
        args.risk_free_rate = None
        args.volatility = None
        args.time_to_maturity = None
        args.plot = None
        args.style = 'bmh'
        
        return args
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Black-Scholes Option Pricing Model')
        
        parser.add_argument('--interactive', '-i', action='store_true',
                            help='Run in interactive mode')
        parser.add_argument('--stock_price', '-s', type=float, required=False,
                            help='Current stock price (example: 175.50 for AAPL)')
        parser.add_argument('--strike_price', '-e', type=float, required=False,
                            help='Strike price (example: 180.00)')
        parser.add_argument('--risk_free_rate', '-r', type=float, required=False,
                            help='Risk-free interest rate (example: 0.0475 for 4.75%%)')
        parser.add_argument('--volatility', '-v', type=float, required=False,
                            help='Volatility of the underlying stock (example: 0.25 for 25%%)')
        parser.add_argument('--time_to_maturity', '-t', type=float, required=False,
                            help='Time to maturity in years (example: 0.25 for 3 months)')
        parser.add_argument('--plot', '-p', action='store_true',
                            help='Plot option prices for a range of stock prices')
        parser.add_argument('--style', type=str, default='bmh',
                            help='Matplotlib style to use for plotting (example: "seaborn")')
        
        args = parser.parse_args()
        
        # If not in interactive mode, ensure all required parameters are provided
        if not args.interactive:
            missing = []
            if args.stock_price is None: missing.append("--stock_price")
            if args.strike_price is None: missing.append("--strike_price")
            if args.risk_free_rate is None: missing.append("--risk_free_rate")
            if args.volatility is None: missing.append("--volatility")
            if args.time_to_maturity is None: missing.append("--time_to_maturity")
            
            if missing:
                parser.error(f"The following arguments are required when not in interactive mode: {', '.join(missing)}")
        
        return args

def black_scholes_calculator(stock_price, strike_price, risk_free_rate, 
                            volatility, time_to_maturity, plot=False, style='bmh', enable_logging=True):
    """
    Calculate option prices using specified parameters and optionally plot results.
    
    This function is designed to be called directly from a notebook or script.
    
    Parameters:
    -----------
    stock_price : float
        Current stock price (e.g., 175.50 for AAPL)
    strike_price : float
        Strike price (e.g., 180.00)
    risk_free_rate : float
        Risk-free interest rate as a decimal (e.g., 0.0475 for 4.75%)
    volatility : float
        Volatility of the underlying stock as a decimal (e.g., 0.25 for 25%)
    time_to_maturity : float
        Time to maturity in years (e.g., 0.25 for 3 months)
    plot : bool, optional
        Whether to plot the option prices
    style : str, optional
        Matplotlib style to use for plotting
    
    Returns:
    --------
    dict
        Dictionary containing the calculated call and put option prices
    """
    try:
        # Set up plotting style
        setup_style(style)
        
        if enable_logging:
            log_event("Non-interactive calculation started", {
                "stock_price": stock_price,
                "strike_price": strike_price,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "time_to_maturity": time_to_maturity,
                "plot": plot,
                "style": style,
            })
        
        # Calculate option prices
        call_price = call_option_price(stock_price, strike_price, risk_free_rate, 
                                       volatility, time_to_maturity)
        
        put_price = put_option_price(stock_price, strike_price, risk_free_rate, 
                                     volatility, time_to_maturity)
        
        print("\nBlack-Scholes Option Pricing Results:")
        print(f"Stock Price: ${stock_price:.2f}")
        print(f"Strike Price: ${strike_price:.2f}")
        print(f"Risk-free Rate: {risk_free_rate*100:.2f}%")
        print(f"Volatility: {volatility*100:.2f}%")
        print(f"Time to Maturity: {time_to_maturity:.2f} years")
        print(f"Call Option Price: ${call_price:.4f}")
        print(f"Put Option Price: ${put_price:.4f}")
        
        # Additional information
        if stock_price > strike_price:
            print("Status: Call Option is in-the-money, Put Option is out-of-the-money")
        elif stock_price < strike_price:
            print("Status: Call Option is out-of-the-money, Put Option is in-the-money")
        else:
            print("Status: Both options are at-the-money")
        
        # Plot option prices if requested
        if plot:
            # Create a range of stock prices centered around the current price
            price_range = max(stock_price, strike_price) * 0.5
            S_range = np.linspace(stock_price - price_range, stock_price + price_range, 100)
            plot_option_prices(S_range, strike_price, risk_free_rate, 
                               volatility, time_to_maturity)
        
        if enable_logging:
            log_event("Non-interactive calculation completed", {
                "call_price": call_price,
                "put_price": put_price,
            })
        
        return {'call_price': call_price, 'put_price': put_price}
    except Exception as error:
        if enable_logging:
            log_error("Non-interactive calculation failed", error, {
                "stock_price": stock_price,
                "strike_price": strike_price,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "time_to_maturity": time_to_maturity,
            })
        print(f"\nError: {error}")
        return None

def main(enable_logging=True):
    """Main function to handle command line arguments and run the program."""
    try:
        args = parse_args_or_use_defaults()
        
        if enable_logging:
            log_event("Model main started", {
                "interactive": args.interactive,
                "style": getattr(args, "style", None),
            })
        
        if args.interactive:
            interactive_menu(enable_logging=enable_logging)
        else:
            # If not in interactive mode, all parameters must be provided via command line
            print("\nRunning Black-Scholes calculator in non-interactive mode...\n")
            return black_scholes_calculator(
                stock_price=args.stock_price,
                strike_price=args.strike_price,
                risk_free_rate=args.risk_free_rate,
                volatility=args.volatility,
                time_to_maturity=args.time_to_maturity,
                plot=args.plot,
                style=args.style,
                enable_logging=enable_logging,
            )
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        if enable_logging:
            log_error("Model main interrupted by user", KeyboardInterrupt())
    except Exception as error:
        print(f"\nUnexpected error: {error}")
        if enable_logging:
            log_error("Model main failed", error)

if __name__ == '__main__':
    main(enable_logging=True)