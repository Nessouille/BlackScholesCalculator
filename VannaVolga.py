#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vanna-Volga Smile Correction for Exotic Options
Applies volatility smile corrections to Black-Scholes prices
Author: Sandy Herho
License: WTFPL - Do What The F*** You Want To Public License
"""

import numpy as np
from scipy import stats


def validate_vv_inputs(S, E, rf, sigma, T, moneyness=1.0, t=0):
    """Validate Vanna-Volga correction inputs."""
    if S <= 0:
        raise ValueError("Stock price must be greater than 0.")
    if E <= 0:
        raise ValueError("Strike price must be greater than 0.")
    if sigma <= 0:
        raise ValueError("Volatility must be greater than 0.")
    if T <= t:
        raise ValueError("Time to maturity must be greater than the current time.")
    if rf < 0:
        raise ValueError("Risk-free rate cannot be negative.")
    if moneyness <= 0:
        raise ValueError("Moneyness must be greater than 0.")


def calculate_greeks(S, E, rf, sigma, T, t=0, option_type='call'):
    """
    Calculate delta, gamma, vega, vanna, and volga for vanilla options.
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free rate (annual)
    sigma : float
        Volatility (annual)
    T : float
        Time to maturity (years)
    t : float
        Current time
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    dict
        Greeks dictionary with delta, gamma, vega, vanna, volga
    """
    tau = T - t
    
    d1 = (np.log(S / E) + (rf + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    # Standard Greeks
    if option_type.lower() == 'call':
        delta = stats.norm.cdf(d1)
    else:
        delta = stats.norm.cdf(d1) - 1
    
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(tau))
    vega = S * stats.norm.pdf(d1) * np.sqrt(tau) / 100  # per 1% change in volatility
    
    # Vanna: sensitivity to spot and volatility
    vanna = -stats.norm.pdf(d1) * d2 / sigma / 100  # per 1% change in volatility
    
    # Volga: second-order sensitivity to volatility
    volga = S * stats.norm.pdf(d1) * np.sqrt(tau) * d1 * d2 / sigma / 10000  # per 1% change squared
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga,
        'd1': d1,
        'd2': d2,
    }


def vanna_volga_correction(S, E, rf, sigma_atm, sigma_25c, sigma_25p, T, 
                          call_price_bs, put_price_bs, t=0, verbose=True):
    """
    Calculate Vanna-Volga smile correction to vanilla option prices.
    
    Uses three reference volatilities (ATM, 25-delta call, 25-delta put) to 
    estimate the volatility smile correction.
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free rate (annual)
    sigma_atm : float
        At-the-money volatility
    sigma_25c : float
        25-delta call volatility (typically higher due to smile)
    sigma_25p : float
        25-delta put volatility (typically higher due to smile)
    T : float
        Time to maturity (years)
    call_price_bs : float
        Black-Scholes call option price (calculated with sigma_atm)
    put_price_bs : float
        Black-Scholes put option price (calculated with sigma_atm)
    t : float
        Current time
    verbose : bool
        Whether to print intermediate calculations
        
    Returns:
    --------
    dict
        Corrected call and put prices, smile corrections, and Greeks
    """
    validate_vv_inputs(S, E, rf, sigma_atm, T, t=t)
    
    tau = T - t
    
    # Calculate Greeks at ATM
    greeks_atm = calculate_greeks(S, E, rf, sigma_atm, T, t=t, option_type='call')
    
    # Calculate 25-delta strike prices
    # For 25-delta call: N(d1) = 0.25
    d1_25c = stats.norm.ppf(0.25)
    E_25c = S * np.exp((rf - sigma_atm**2 / 2) * tau + sigma_atm * np.sqrt(tau) * d1_25c)
    
    # For 25-delta put: N(d1) = -0.75
    d1_25p = stats.norm.ppf(-0.75)
    E_25p = S * np.exp((rf - sigma_atm**2 / 2) * tau + sigma_atm * np.sqrt(tau) * d1_25p)
    
    # Greeks for 25-delta strikes
    greeks_25c = calculate_greeks(S, E_25c, rf, sigma_25c, T, t=t, option_type='call')
    greeks_25p = calculate_greeks(S, E_25p, rf, sigma_25p, T, t=t, option_type='put')
    
    # Volatility smile differences
    smile_25c = (sigma_25c - sigma_atm) * 100  # in basis points
    smile_25p = (sigma_25p - sigma_atm) * 100
    
    # Vanna-Volga correction formula
    # Correction = (1/2) * Volga * (smile²) + Vanna * smile
    
    # Correction for call option
    call_correction = (
        0.5 * greeks_atm['volga'] * ((smile_25c + smile_25p) / 2)**2 / 10000 +
        greeks_atm['vanna'] * (smile_25c + smile_25p) / 2 / 100
    )
    
    # Correction for put option (similar calculation)
    put_correction = (
        0.5 * greeks_atm['volga'] * ((smile_25c + smile_25p) / 2)**2 / 10000 +
        greeks_atm['vanna'] * (smile_25c + smile_25p) / 2 / 100
    )
    
    call_price_vv = call_price_bs + call_correction
    put_price_vv = put_price_bs + put_correction
    
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Vanna-Volga Smile Analysis:")
        print(f"{'─' * 60}")
        print(f"ATM Volatility: {sigma_atm*100:.2f}%")
        print(f"25-Delta Call Volatility: {sigma_25c*100:.2f}% (Smile: {smile_25c:.1f} bps)")
        print(f"25-Delta Put Volatility: {sigma_25p*100:.2f}% (Smile: {smile_25p:.1f} bps)")
        print(f"\nGreeks at ATM:")
        print(f"  Vega (per 1%): ${greeks_atm['vega']:.4f}")
        print(f"  Vanna (per 1%): ${greeks_atm['vanna']:.6f}")
        print(f"  Volga (per 1%²): ${greeks_atm['volga']:.6f}")
        print(f"\nSmile Correction:")
        print(f"  Call Correction: ${call_correction:.4f}")
        print(f"  Put Correction: ${put_correction:.4f}")
        print(f"{'─' * 60}")
    
    return {
        'call_price_bs': call_price_bs,
        'call_price_vv': call_price_vv,
        'call_correction': call_correction,
        'put_price_bs': put_price_bs,
        'put_price_vv': put_price_vv,
        'put_correction': put_correction,
        'greeks': greeks_atm,
        'smile_25c': smile_25c,
        'smile_25p': smile_25p,
    }


def get_vanna_volga_explanation():
    """Return explanation of the Vanna-Volga method."""
    explanation = """
The Vanna-Volga method is a smile correction technique used to adjust Black-Scholes 
prices for volatility smile effects when pricing exotic options.

KEY CONCEPTS:
─────────────

1. Volatility Smile:
   In reality, implied volatility varies with strike price (not flat as Black-Scholes assumes).
   This creates a "smile" pattern in IV plots.

2. Vanna (sensitivity to spot AND volatility):
   How much the delta changes with volatility changes.
   Exotic options with path dependence are sensitive to Vanna.

3. Volga (second-order volatility sensitivity):
   How much vega itself changes with volatility changes.
   Important for convexity effects in options pricing.

4. The Correction:
   
   Price_VV = Price_BS + Vanna × smile + (1/2) × Volga × smile²
   
   Where "smile" is the difference between reference volatilities.

APPLICATION:
─────────────

The method uses three reference volatilities:
  • ATM: At-the-money volatility
  • 25Δ Call: Volatility for 25-delta call strike
  • 25Δ Put: Volatility for 25-delta put strike

This allows the model to capture smile effects without explicit smile parametrization.

ADVANTAGES:
───────────
✓ Simple and intuitive
✓ Requires only three volatility quotes
✓ Captures first and second-order smile effects
✓ Computationally efficient
✓ Works well for moderate moneyness levels

LIMITATIONS:
────────────
✗ Linear approximation (breaks down for large smiles)
✗ Assumes specific smile shape
✗ Not ideal for extreme OTM options
✗ Does not capture term structure of volatility
    """
    return explanation
