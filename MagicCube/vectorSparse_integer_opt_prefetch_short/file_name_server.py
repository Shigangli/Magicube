import string
import os
import csv


def get_file_name(bm, dimK, dimV, kernel, sort, job, sddmm_alg, precision, mem):
    # Select the kernel to profile
    if kernel == 'wmma':
        suffix = '_wmma'
    elif kernel == 'cuda':
        suffix = '_cuda'
    elif kernel == 'sputnik':
        suffix = '_sputnik'
    elif kernel == 'cusparse':
        suffix = '_cusparse'
    elif kernel == 'bell':
        suffix = '_blocked_ell'
    else:
        suffix = '_dense'

    # Select the precision
    if precision == 'half':
        suffix += '_half'
    else:
        suffix += '_single'

    if sort:
        suffix += '_sort'

    # If the job is SpMM
    if job == 'spmm':
        if mem:
            output_file = './csv/mem/spmm/%s_k%d_v%d.csv' % (bm.replace('/raid/datasets/', '').replace('.smtx', '') + suffix, dimK, dimV)
        else:
            output_file = './csv/spmm/%s_k%d_v%d.csv' % (bm.replace('/raid/datasets/', '').replace('.smtx', '') + suffix, dimK, dimV)
    # Else if the job is SDDMM
    else:
        if kernel == 'wmma':
            suffix += '_%s' % sddmm_alg
        if mem:
            output_file = './csv/mem/sddmm/%s_k%d_v%d.csv' % (bm.replace('/raid/datasets/', '').replace('.smtx', '') + suffix, dimK, dimV)
        else:
            output_file = './csv/sddmm/%s_k%d_v%d.csv' % (bm.replace('/raid/datasets/', '').replace('.smtx', '') + suffix, dimK, dimV)
    
    return output_file


def get_bm_size(bm, dimK, dimV):
    with open("/raid/datasets/dlmc/%s.smtx" % bm) as bm_file:
        lines = bm_file.readlines()
        # Get the problem size
        [m, n, nnz] = list(map(int, lines[0].split(', ')))
        # return dimK * dimV * nnz
        return dimK * dimV * m * n

# function that collects the result
def extract_duration_ncu(file):
    if os.path.exists(file):
        with open(file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            unit = 'unknown'
            dur_accumulate = 0
            for row in csvreader:
                if len(row) >= 3 and "Duration" in row[-4]:
                    unit = row[-3]
                    try:
                        dur = float(row[-2])
                        if unit == 'second':
                            dur *= 1000
                        elif unit == 'usecond':
                            dur /= 1000
                        elif unit == 'nsecond':
                            dur /= 1e+6
                        else:
                            print('unknown unit')
                        dur_accumulate += dur
                    except:
                        # print(file)
                        return -1.0
            return dur_accumulate
    else:
        print('file %s does not exist' % file)


def geometric_mean(list_in):
    geo_mean = []
    for sub_list in list_in:
        mean = 1.
        for s in sub_list:
            mean *= s
        geo_mean.append(pow(mean, 1./len(sub_list)))
    
    return geo_mean