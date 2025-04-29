/*
Author: KMC20
Date: 2024/2/27
Function: Public definitions for PIM-IVFPQ.
*/

#ifndef REQUEST_H
#define REQUEST_H

#include <stdint.h>

/* Public definitions for DRIM-ANN are listed below */
typedef unsigned char ELEMTYPE;
typedef uint32_t ADDRTYPE;  // sizeof(uint32_t) == sizeof(ELEMTYPE *)
#define MRAM_SIZE (62 << 20)
#define MRAM_ALIGN_BYTES 8
#define ADDRTYPE_NULL 0  // NULL pointer for ADDRTYPE
// Used for priority queue
typedef uint32_t pqueue_pri_t;
typedef struct {
    pqueue_pri_t pri;
	ADDRTYPE val;
    // union {  // Note: keep in coincidence with `pqueue_elem_t` (not including `pos` below)
    //     ADDRTYPE val;
    //     uint8_t pos;
    // };
} pqueue_elem_t_mram;
typedef pqueue_pri_t VECSUMTYPE;  // sizeof(uint64_t) == 4 * sizeof(ELEMTYPE)
typedef uint8_t POINTTYPE;  // sizeof(uint32_t) > sizeof(uint8_t) == log2(maxCodebookEntryAmt)
typedef uint32_t SQUAREROOTDISTTYPE;  // ceil(sizeof(VECSUMTYPE) / 2) == sizeof(SQUAREROOTDISTTYPE)
#define LARGE_SQUARE_RES_SIZE 1024  // 2 ^ sizeof(ELEMTYPE) * squareRoot(sliceAmt)
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
enum MODULE_TYPES { PRE_DEFINING, AFFILIATE_OPS, CAL_RESIDUAL, CONSTR_LUT, CLUSTER_LOADING, CAL_DISTANCE, TOPK_SORT, TOPK_SAVING, MODULE_TYPES_END };
#endif
typedef uint32_t CLUSTER_SIZES_TYPE;
enum SearchingPhases {clusterLocating, resiCal, lutCal, distCal, topkSort, borderPhase};
typedef ELEMTYPE CENTROIDS_TYPE;
typedef int8_t CB_TYPE;
typedef int16_t RESIDUAL_TYPE;

#endif  // REQUEST_H
