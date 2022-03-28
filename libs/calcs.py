import sys
import argparse


def ior_seawater(wavelength, temp): 
    
    output = -1.50156*10**(-6)*temp**2 + 1.07085*10**(-7)*wavelength**2 + -4.27594*10**(-5)*temp + -1.60476*10**(-4)*wavelength + 1.39807
    
    return output

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run calculations related to ayeris')
    parser.add_argument('--ior', default=None, help='calculate index of refraction')