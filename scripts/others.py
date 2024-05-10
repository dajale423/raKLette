from math import log10, floor
import math
# from pyliftover import LiftOver
import csv


def round_sig(x, sig=6, small_value=1.0e-9):
    if math.isnan(x):
        return x
    return round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)


# lo = LiftOver('hg19', 'hg38')

# def get_hg38_pos(chrom, pos):
#     coordinate = lo.convert_coordinate('chr' + str(chrom), int(pos) - 1)
    
#     if len(coordinate) > 0:
#         coordinate = coordinate[0]
#     else:
#         return None
#     pos = str(coordinate[1] + 1)

#     return pos

# lo19 = LiftOver('hg38', 'hg19')


# def get_hg19_pos(chrom, pos):
#     coordinate = lo19.convert_coordinate('chr' + str(chrom), int(pos) - 1)
    
#     if len(coordinate) > 0:
#         coordinate = coordinate[0]
#     else:
#         return None
#     pos = str(coordinate[1] + 1)

#     return pos


# def get_hg19_variant_id(chrom, pos, ref, alt):
#     coordinate = lo19.convert_coordinate('chr' + str(chrom), pos - 1)
    
#     if len(coordinate) > 0:
#         coordinate = coordinate[0]
#     else:
#         return None
#     pos = str(coordinate[1] + 1)
#     return coordinate[0][3:] + "-" + pos + "-" + ref + "-" + alt


# def get_unique_names(region_file):
#     names = []
#     with open(region_file , "r") as f_in:
#         region_reader = csv.reader(f_in, delimiter="\t")
#         next(region_reader)
#         for region in region_reader:
#             name = region[0]+"_"+region[3]
#             if not name in names:
#                 names.append(name)
#     return names
