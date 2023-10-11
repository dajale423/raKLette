from math import log10, floor
import math
# from pyliftover import LiftOver
import csv


def round_sig(x, sig=6, small_value=1.0e-9):
    if math.isnan(x):
        return x
    return round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)
