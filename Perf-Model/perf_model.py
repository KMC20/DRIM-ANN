#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: KMC20
# Date: 2024/7/28
# Function: Estimate the performance of DRIM-ANN to avoid expensive index generation during DSE.

import argparse
from xlwt import Workbook
from re import search
from copy import deepcopy
from ast import literal_eval
from math import ceil, log


C_STRUCT_ALIGNBYTES = 4
BitWidth = {
    'c': 1,
    'q': 1,
    'r': 2,
    'cb': 1,
    'a': 1,
    'sqt': 2,
    'l': 4,
    'd': 4
}  # Unit: Byte
RANDOM_MEM_COEFFICIENT = 622.36 / 77.86

def performance_model(N, D, K, P, C, M, CB, Q, P_dpu, F, PE, BW, BW_device):
    '''
    Estimate performance based on input configures.
    :param N:           The size of the base dataset.
    :param D:           The dimension of queries/centroids.
    :param K:           The amount of neighbors.
    :param P:           The amount of fetched clusters.
    :param C:           The average size of clusters.
    :param M:           The slice amount of vectors.
    :param CB:          The amount of codebook entries.
    :param Q:           The size of query batches.
    :param P_dpu:       The average amount of fetched clusters on each DPU.
    :param F:           The frequency of the processor used by each phase.
    :param PE:          The amount of the processor (for CPU, the total amount of hyperthreads) used by each phase.
    :param BW:          The bandwidth of the memory chip used by each phase.
    :param BW_device:   The bandwidth of memory devices.
    :return:            Estimated latency.
    '''
    CP = {}
    IO = {}
    IO_W = {}
    IO_M = {}
    CP["CL"] = Q * ceil(N / C) * (3 * ceil(D * BitWidth["q"] * 2 * 8 / 512) - 1 + log(P) - 1)  # ceil(D * BitWidth["q"] * 2 * 8 / 512): AVX512 for vector processing
    IO["CL"] = Q * ceil(N / C) * ((BitWidth["c"] + BitWidth["q"]) * D + (BitWidth["q"] * 4 + BitWidth["a"]) * (log(P) + 1))  # Not struct, non-alignment
    CP["CL"] += Q * ceil(N / C) * (2 * ceil(D * BitWidth["q"] * 2 * 8 / 512)  + (log(P) + 1))  # Cache loading instructions
    P = P_dpu  # The following phases are completed on DPUs
    CP["RC"] = Q * P * D
    IO["RC"] = (BitWidth["c"] + BitWidth["q"]) * Q * P * D
    IO_W["RC"] = BitWidth["r"] * Q * P * D
    IO_M["RC"] = (BitWidth["c"] + BitWidth["q"]) * Q * P * D
    CP["LC"] = Q * P * CB * (M * 3 - 1) * D / M
    IO["LC"] = Q * P * CB * (BitWidth["q"] * 2 * D + BitWidth["l"] * M)
    IO_W["LC"] = Q * P * (CB * BitWidth["l"] * M + BitWidth["r"] * D * CB + CB * BitWidth["sqt"] * D)
    IO_M["LC"] = Q * P * CB * BitWidth["cb"] * D
    CP["DC"] = Q * P * C * (M - 1)
    IO["DC"] = Q * P * C * (M * (BitWidth["a"] + BitWidth["l"]) + BitWidth["l"])
    IO_W["DC"] = Q * P * C * (M * BitWidth["l"] + BitWidth["d"])
    IO_M["DC"] = Q * P * C * (M * BitWidth["a"])
    CP["TS"] = Q * P * C * (log(K) - 1)
    IO["TS"] = Q * P * C * (log(K) + 1) * (BitWidth["l"] + BitWidth["a"])
    IO_W["TS"] = Q * P * C * (log(K) + 1) * ceil((BitWidth["l"] + BitWidth["a"]) / C_STRUCT_ALIGNBYTES) * C_STRUCT_ALIGNBYTES  # C struct: 4-byte alignment on DPUs
    IO_M["TS"] = Q * K * ceil((BitWidth["l"] + BitWidth["a"]) / C_STRUCT_ALIGNBYTES) * C_STRUCT_ALIGNBYTES                     # C struct: 4-byte alignment on DPUs
    # Each WRAM loading also costs a cycle according to DPU traces. Appended below
    CP["RC"] += 3 * Q * P * D
    CP["LC"] += Q * P * CB * (M + 3 * D) * BitWidth["l"] / BitWidth["r"]
    CP["DC"] += Q * P * C * (M * 2 + 1)
    CP["TS"] += Q * P * C * (log(K) + 1) + Q * K
    lat = {}
    lat["CL"] = max(CP["CL"] / (F["CL"] * PE["CL"]), min(N * M * BitWidth["a"], IO["CL"]) / BW["CL"])
    lat["RC"] = max(CP["RC"] / (F["RC"] * PE["RC"]), IO_W["RC"] / BW_device["WRAM"] + IO_M["RC"] / BW_device["MRAM"])
    lat["LC"] = max(CP["LC"] / (F["LC"] * PE["LC"]), IO_W["LC"] / BW_device["WRAM"] + IO_M["LC"] / BW_device["MRAM"]) * 1.6
    lat["DC"] = max(CP["DC"] / (F["DC"] * PE["DC"]), IO_W["DC"] / BW_device["WRAM"] + IO_M["DC"] / BW_device["MRAM"]) * 1.6
    lat["TS"] = max(CP["TS"] / (F["TS"] * PE["TS"]), IO_W["TS"] / BW_device["WRAM"] + IO_M["TS"] / BW_device["MRAM"])
    latency = max(lat["CL"], lat["RC"] + lat["LC"] + lat["DC"] + lat["TS"])

    return latency


if __name__ == '__main__':
    # Get configure parameters.
    parser = argparse.ArgumentParser()
    # Disk file params
    parser.add_argument('--config-file-name', type=str, default='config.json', help='The full name, i.e. path + name, of the input json file (default: config.json)')
    parser.add_argument('--result-file-name', type=str, default='result.xlsx', help='The full name, i.e. path + name, of the output result excel workbook based on the data in the input log file (default: result.xlsx)')
    # Constant params
    parser.add_argument('--codebook-entry-amount', type=int, default='256', help='The amount of codebook entries. Used by the performance model (default: 256)')
    parser.add_argument('--dpu-amount', type=int, default='2560', help='The amount of PIM-DPUs. Used by the performance model (default: 2560)')
    parser.add_argument('--host-thread-amount', type=int, default='64', help='The amount of host threads. Used by the performance model (default: 64)')
    parser.add_argument('--query-batch-size', type=int, default='10000', help='The size of query batches. Used by the performance model (default: 10000)')
    parser.add_argument('--dim', type=int, default='128', help='The dimension of query set. Used by the performance model (default: 128)')
    args = parser.parse_args()

    # Check for configure params.
    if args.codebook_entry_amount < 1:
        raise ValueError(f'Argument `--codebook-entry-amount` should be positive! The input `{args.codebook_entry_amount}` is illegal!')
    if args.dpu_amount < 1:
        raise ValueError(f'Argument `--dpu-amount` should be positive! The input `{args.dpu_amount}` is illegal!')
    if args.query_batch_size < 1:
        raise ValueError(f'Argument `--query-batch-size` should be positive! The input `{args.query_batch_size}` is illegal!')

    # Record configure params.
    for arg_key in parser._actions[1:]:
        arg_key = arg_key.dest
        print(('{}: {}').format(arg_key, eval('args.{}'.format(arg_key))))

    # Hardware params.
    F = {
        'CL': 2100_000_000,
        'RC': 450_000_000,
        'LC': 450_000_000,
        'DC': 450_000_000,
        'TS': 450_000_000
    }  # Unit: Hz
    PE = {
        'CL': 64,
        'RC': args.dpu_amount,
        'LC': args.dpu_amount,
        'DC': args.dpu_amount,
        'TS': args.dpu_amount
    }
    BW = {
        'CL': 19.2 * 1024 * 1024 * 1024,
        'RC': args.dpu_amount * 1024 * 1024 * 1024,
        'LC': args.dpu_amount * 1024 * 1024 * 1024,
        'DC': args.dpu_amount * 1024 * 1024 * 1024,
        'TS': args.dpu_amount * 1024 * 1024 * 1024
    }  # Unit: B/s
    BW_device = {  # The reproduced bandwidth of DRAM-PIMs according to the official benchmarks are different from the theoretical one. The former is used here
        'DDR4': 19.2 * 1024 * 1024 * 1024,
        'WRAM': args.dpu_amount * 1612.56 * 1024 * 1024 / RANDOM_MEM_COEFFICIENT,  # Random accesses take the major part. The coeffcient comes from the official benchmark
        'MRAM': args.dpu_amount * 573.79 * 1024 * 1024
    }  # Unit: B/s

    # Load configures.
    legal_settings = {}
    with open(args.config_file_name, 'r') as config_file:
        config_row = config_file.readline()
        while config_row:
            if config_row[0] != '{':
                config_row = config_file.readline()
                continue
            local_setting = {}
            try:
                local_setting = literal_eval(config_row)
            except SyntaxError as e:  # Illegal str
                config_row = config_file.readline()
                continue
            if type(local_setting) is dict:
                for key_d in local_setting.keys():
                    if key_d not in legal_settings.keys():
                        legal_settings[key_d] = deepcopy(local_setting[key_d])
                        continue
                    for key_i in local_setting[key_d].keys():
                        if key_i not in legal_settings[key_d].keys():
                            legal_settings[key_d][key_i] = deepcopy(local_setting[key_d][key_i])
                            continue
                        for key_k in local_setting[key_d][key_i].keys():
                            if key_k not in legal_settings[key_d][key_i].keys():
                                legal_settings[key_d][key_i][key_k] = deepcopy(local_setting[key_d][key_i][key_k])
                                continue
                            legal_settings[key_d][key_i][key_k].update(local_setting[key_d][key_i][key_k])
            config_row = config_file.readline()
    
    # Estimate and record the performance.
    pat_n = r'(\d+)'
    pat_pm = r'(\d+)\D*(\d+)'
    result_excel = Workbook()
    sheet = result_excel.add_sheet('Estimated performance')
    sheet.write(0, 0, label='Dataset')
    sheet.write(0, 1, label='K')
    sheet.write(0, 2, label='nlist')
    sheet.write(0, 3, label='nprobe')
    sheet.write(0, 4, label='M')
    sheet.write(0, 5, label='CB')
    sheet.write(0, 6, label='Latency')
    result_idx = 1
    for key_d in legal_settings.keys():
        for key_i in legal_settings[key_d].keys():
            for key_k in legal_settings[key_d][key_i].keys():
                for key_r in legal_settings[key_d][key_i][key_k].keys():
                    N = int(search(pat_n, key_d).groups()[0]) * 1024 * 1024
                    nlist, M = list(map(int, search(pat_pm, key_i).groups()))
                    nprobe = legal_settings[key_d][key_i][key_k][key_r]
                    latency = None if nprobe is None else performance_model(N, args.dim, key_k, nprobe, N / nlist, M, args.codebook_entry_amount, args.query_batch_size, ceil(nprobe * args.query_batch_size / args.dpu_amount) * args.dpu_amount / args.query_batch_size, F, PE, BW, BW_device)
                    sheet.write(result_idx, 0, label=key_d)
                    sheet.write(result_idx, 1, label=key_k)
                    sheet.write(result_idx, 2, label=nlist)
                    sheet.write(result_idx, 3, label=nprobe)
                    sheet.write(result_idx, 4, label=M)
                    sheet.write(result_idx, 5, label=args.codebook_entry_amount)
                    sheet.write(result_idx, 6, label=latency)
                    result_idx += 1
    result_excel.save(args.result_file_name)
