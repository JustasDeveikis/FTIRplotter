# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:30:18 2026

@author: justas
"""

from classes.FTIRreader import FTIRreader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Global Matplotlib settings
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'lines.linewidth': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,    # ticks on top
    'ytick.right': True,  # ticks on right
    'xtick.minor.visible': True,  # enable minor ticks globally
    'ytick.minor.visible': True,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,
})



if __name__ == "__main__":
    
    """
    FTIR Analysis Entry Point
    -------------------------
    This script initializes the FTIRreader to process OPUS format files and generates polarization-dependent reflectance plots.
    
    Usage:
        Toggle the 'if 0:' / 'if 1:' blocks to select the desired dataset 
        for analysis. Ensure the relative path to the 'classes' folder is 
        maintained.
    """
    
    if 0:
        dataBBrefl = FTIRreader(
            folder="dataBlueBronzeCRY14Refl",
            sample_prefix="BlueBronzeCRY14",
            reference_prefix="GoldMirrorCRY14",
            x_limits = (200, 2000),
            )
    
    
    if 1:
        dataBBrefl = FTIRreader(
            folder="dataBlueBronzePolWrong",
            sample_prefix="BlueBronzeBB1",
            reference_prefix="GoldMirror",
            x_limits = (350, 2000),
            )
        
        # --- Plotting ---
        # Generates a two-panel figure
        dataBBrefl.plot_R_at_pol_angle(pol_angle=0)
    
    