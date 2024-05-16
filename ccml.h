#if !defined(CCML_IMPL)
#define CCML_IMPL

#define CCML_SRCS_MAX 2
#define CCML_TYPE_MAX 3
#define CCML_DIMS_MAX 4

#define CCML_CHAR_MAX 100
#define CCML_NODE_MAX 128

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// known issues:
// - including ccml.h in separate compilation units compiles separate/independent symbols
// - tensors of size 1 aren't supported, due to attempting to embed them into the kernel
// - metal lacks support for floating point atomics, so rn handling w/ int atomics
// - a lot of function return statuses aren't checked, mostly snprintf/fread/fwrite
// - if a tensor isn't part of a graph it won't get freed by ccml_graph_free()

//
//  ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗
//  ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
//     ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
//     ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
//     ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
//     ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
//

typedef enum ccml_backend {
    CCML_BACKEND_METAL
} ccml_backend;

typedef enum ccml_type {
    CCML_TYPE_FP32
} ccml_type;

const static int ccml_type_sizes[CCML_TYPE_MAX] = {
    [CCML_TYPE_FP32] = sizeof(float),
};

typedef enum ccml_buff {
    // no buffer, values only existing as intermediary scalars in the kernel
    CCML_BUFF_NONE,
    // dedicated buffer for constant values
    CCML_BUFF_CNST,
    // dedicated buffer for loaded/saved values
    CCML_BUFF_PERM
} ccml_buff;

typedef enum ccml_oper {
    CCML_OPER_NONE,
    CCML_OPER_LOG,
    CCML_OPER_EXP,
    CCML_OPER_SIN,
    CCML_OPER_REC, // reciprocal
    CCML_OPER_SQRT,
    CCML_OPER_ADD,
    CCML_OPER_MUL,
    CCML_OPER_SUM,
    CCML_OPER_RES, // reshape
    CCML_OPER_PER, // permute
} ccml_oper;

typedef struct ccml_tensor ccml_tensor;

typedef struct ccml_tensor {
    ccml_type type;
    ccml_buff buff;
    int oper;

    int shape[CCML_DIMS_MAX];
    int stride[CCML_DIMS_MAX];

    ccml_tensor * src[CCML_SRCS_MAX];
    ccml_tensor * grad;

    int index;
    void * data;
    bool requires_grad;
} ccml_tensor;

//
//  ██╗███╗   ██╗██╗████████╗
//  ██║████╗  ██║██║╚══██╔══╝
//  ██║██╔██╗ ██║██║   ██║
//  ██║██║╚██╗██║██║   ██║
//  ██║██║ ╚████║██║   ██║
//  ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝
//
//  ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
//  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

static ccml_tensor * ccml_new_tensor_impl(ccml_type type, int * shape) {
    ccml_tensor * result = malloc(sizeof(ccml_tensor));

    *result = (ccml_tensor){
    /*.type          =*/ type,
    /*.buff          =*/ CCML_BUFF_NONE,
    /*.oper          =*/ CCML_OPER_NONE,
    /*.shape         =*/ {shape[0], shape[1], shape[2], shape[3]},
    /*.stride        =*/ {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1},
    /*.src           =*/ {NULL},
    /*.grad          =*/ NULL,
    /*.index         =*/ -1,
    /*.data          =*/ NULL,
    /*.requires_grad =*/ false,
    };

    return result;
}

static ccml_tensor * ccml_new_tensor_1d(ccml_type type, int ne0,
                                        bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, 1, 1, 1};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_PERM;
    result->requires_grad = requires_grad;

    return result;
}

static ccml_tensor * ccml_new_tensor_2d(ccml_type type, int ne0, int ne1,
                                        bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, 1, 1};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_PERM;
    result->requires_grad = requires_grad;

    return result;
}

static ccml_tensor * ccml_new_tensor_3d(ccml_type type, int ne0, int ne1, int ne2,
                                        bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, ne2, 1};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_PERM;
    result->requires_grad = requires_grad;

    return result;
}

static ccml_tensor * ccml_new_tensor_4d(ccml_type type, int ne0, int ne1, int ne2, int ne3,
                                        bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, ne2, ne3};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_PERM;
    result->requires_grad = requires_grad;

    return result;
}

static int ccml_size(ccml_tensor * tensor) {
    return tensor->shape[0] * tensor->shape[1] * tensor->shape[2] * tensor->shape[3];
}

static void ccml_set(ccml_tensor * tensor, float * data) {
    tensor->type = CCML_TYPE_FP32;
    tensor->buff = CCML_BUFF_CNST;

    int size = ccml_size(tensor);
    tensor->data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        *((float*)tensor->data + i) = data[i];
    }
}

static void ccml_fill(ccml_tensor * tensor, float value) {
    tensor->type = CCML_TYPE_FP32;
    tensor->buff = CCML_BUFF_CNST;

    int size = ccml_size(tensor);
    tensor->data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        *((float*)tensor->data + i) = value;
    }
}

//
//  ███╗   ███╗██╗███████╗ ██████╗
//  ████╗ ████║██║██╔════╝██╔════╝
//  ██╔████╔██║██║███████╗██║
//  ██║╚██╔╝██║██║╚════██║██║
//  ██║ ╚═╝ ██║██║███████║╚██████╗
//  ╚═╝     ╚═╝╚═╝╚══════╝ ╚═════╝
//
//  ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
//  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

static bool ccml_can_broadcast(ccml_tensor * lhs, ccml_tensor * rhs) {
    if (rhs == NULL || lhs == NULL)
        return true;
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if (lhs->shape[i] != rhs->shape[i] && lhs->shape[i] != 1 && rhs->shape[i] != 1) {
            return false;
        }
    }

    return true;
}

static void ccml_tensor_print(ccml_tensor * tensor) {
    if (tensor->data == NULL) {
        printf("tensor %d: tensor data field is NULL\n", tensor->index);
    }
    printf("tensor %d: ", tensor->index);
    for (int i = 0; i < ccml_size(tensor); i++) {
        printf("%f ", *((float *)tensor->data + i));
    }
    printf("\n");
}

static bool ccml_is_permute(ccml_tensor * tensor) {
    return tensor->oper == CCML_OPER_RES || tensor->oper == CCML_OPER_PER;
}

static bool ccml_is_leaf(ccml_tensor * tensor) {
    return tensor->src[0] == NULL && tensor->src[1] == NULL;
}

static bool ccml_has_buffer(ccml_tensor * tensor) {
    return tensor->buff != CCML_BUFF_NONE;
}

static int ccml_ndim(ccml_tensor * tensor) {
    int last_dim = 0;
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if(tensor->shape[i] != 1) last_dim = i;
    }
    return last_dim == 0 ? 1 : last_dim + 1;
}

static bool ccml_is_vector(ccml_tensor * tensor) {
    return tensor->shape[1] == 1 && tensor->shape[2] == 1 &&
        tensor->shape[3] == 1;
}

static bool ccml_is_matrix(ccml_tensor * tensor) {
    return tensor->shape[0] != 1 && tensor->shape[1] != 1 &&
        tensor->shape[2] == 1 && tensor->shape[3] == 1;
}

//
//  ██████╗ ██████╗ ██╗███╗   ███╗ █████╗ ██████╗ ██╗   ██╗
//  ██╔══██╗██╔══██╗██║████╗ ████║██╔══██╗██╔══██╗╚██╗ ██╔╝
//  ██████╔╝██████╔╝██║██╔████╔██║███████║██████╔╝ ╚████╔╝
//  ██╔═══╝ ██╔══██╗██║██║╚██╔╝██║██╔══██║██╔══██╗  ╚██╔╝
//  ██║     ██║  ██║██║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║
//  ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
//
//   ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ╚██████╔╝██║     ███████╗██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
//   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

#define CCML_UNARY_OPERATION(function, operation)                                              \
static ccml_tensor * function(ccml_tensor * tensor) {                                          \
    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, tensor->shape);                  \
                                                                                            \
    for (int i = 0; i < CCML_DIMS_MAX; i++) {                                                  \
        result->stride[i] = tensor->stride[i];                                                 \
    }                                                                                          \
                                                                                            \
    result->oper = operation;                                                                  \
    result->src[0] = tensor;                                                                   \
    result->requires_grad = tensor->requires_grad;                                             \
                                                                                            \
    return result;                                                                             \
}

CCML_UNARY_OPERATION(ccml_log, CCML_OPER_LOG);
CCML_UNARY_OPERATION(ccml_exp, CCML_OPER_EXP);
CCML_UNARY_OPERATION(ccml_sin, CCML_OPER_SIN);
CCML_UNARY_OPERATION(ccml_rec, CCML_OPER_REC);
CCML_UNARY_OPERATION(ccml_sqrt, CCML_OPER_SQRT);

static ccml_tensor * ccml_add(ccml_tensor * lhs, ccml_tensor * rhs) {
    assert(ccml_can_broadcast(lhs, rhs) && "incompatible dimensions for broadcasting");

    bool null_input = lhs == NULL || rhs == NULL;

    int shape[CCML_DIMS_MAX] = {0};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[i] :
            (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    ccml_tensor * result = ccml_new_tensor_impl(lhs->type, shape);

    result->oper = CCML_OPER_ADD;
    result->src[0] = lhs;
    result->src[1] = rhs;
    result->requires_grad = null_input ? lhs->requires_grad : lhs->requires_grad || rhs->requires_grad;

    return result;
}

static ccml_tensor * ccml_mul(ccml_tensor * lhs, ccml_tensor * rhs) {
    assert(ccml_can_broadcast(lhs, rhs) && "incompatible dimensions for broadcasting");

    bool null_input = lhs == NULL || rhs == NULL;

    int shape[CCML_DIMS_MAX] = {0};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[i] :
            (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    ccml_tensor * result = ccml_new_tensor_impl(lhs->type, shape);

    result->oper = CCML_OPER_MUL;
    result->src[0] = lhs;
    result->src[1] = rhs;
    result->requires_grad = null_input ? lhs->requires_grad : lhs->requires_grad || rhs->requires_grad;

    return result;
}

static ccml_tensor * ccml_reshape(ccml_tensor * tensor, int shape[CCML_DIMS_MAX]) {
    int size = ccml_size(tensor);
    int new_size = shape[0] * shape[1] * shape[2] * shape[3];
    assert(size == new_size && "reshaped and source tensor must have the same size");

    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, shape);

    result->oper = CCML_OPER_RES;
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

static ccml_tensor * ccml_permute(ccml_tensor * tensor, int perm[CCML_DIMS_MAX]) {
    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, tensor->shape);
    int ndim = ccml_ndim(tensor);
    for (int i = 0; i < ndim; i++) {
        result->shape[i] = tensor->shape[perm[i]];
        result->stride[i] = tensor->stride[perm[i]];
    }

    result->oper = CCML_OPER_PER;
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

static ccml_tensor * ccml_sum(ccml_tensor * tensor, int n_axes, int axes[CCML_DIMS_MAX]) {
    assert(n_axes > 0 && n_axes < CCML_DIMS_MAX && "invalid number of summed axes");

    int shape[CCML_DIMS_MAX] = {1, 1, 1, 1};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = tensor->shape[i];
    }

    for (int i = 0; i < n_axes; i++) {
        shape[axes[i]] = 1;
    }

    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, shape);

    result->oper = CCML_OPER_SUM;
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

//
//  ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗  █████╗ ██████╗ ██╗   ██╗
//  ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
//  ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║███████║██████╔╝ ╚████╔╝
//  ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║██╔══██║██╔══██╗  ╚██╔╝
//  ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝██║  ██║██║  ██║   ██║
//  ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
//
//   ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ╚██████╔╝██║     ███████╗██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
//   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

typedef enum ccml_oper_misc {
    CCML_OPER_MISC_NEG = 11,
    CCML_OPER_MISC_SUB,
    CCML_OPER_MISC_DIV,
    CCML_OPER_MISC_SQR,
    CCML_OPER_MISC_COS,
    CCML_OPER_MISC_TANH,
    CCML_OPER_MISC_MMUL,
} ccml_oper_misc;

static ccml_tensor * ccml_neg(ccml_tensor * tensor) {
    return ccml_log(ccml_rec(ccml_exp(tensor)));
}

static ccml_tensor * ccml_sub(ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_add(lhs, ccml_neg(rhs));
}

static ccml_tensor * ccml_div(ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_mul(lhs, ccml_rec(rhs));
}

static ccml_tensor * ccml_square(ccml_tensor * tensor) {
    return ccml_mul(tensor, tensor);
}

static ccml_tensor * ccml_cos(ccml_tensor * tensor) {
    ccml_tensor * pi_2 = ccml_new_tensor_impl(tensor->type, (int[]){1, 1, 1, 1});
    ccml_fill(pi_2, M_PI_2);

    return ccml_sin(ccml_add(tensor, pi_2));
}

static ccml_tensor * ccml_tanh(ccml_tensor * tensor) {
    return ccml_div(ccml_sub(ccml_exp(tensor), ccml_exp(ccml_neg(tensor))),
                    ccml_add(ccml_exp(tensor), ccml_exp(ccml_neg(tensor))));
}

static ccml_tensor * ccml_matmul(ccml_tensor * lhs, ccml_tensor * rhs) {
    assert(ccml_is_matrix(lhs) && "tensor must be a matrix for matmul");
    assert(ccml_is_matrix(rhs) && "tensor must be a matrix for matmul");

    assert(lhs->shape[1] == rhs->shape[0]);

    ccml_tensor * lhs_r = ccml_reshape(lhs, (int[]){lhs->shape[0], lhs->shape[1], 1, 1});
    ccml_tensor * rhs_r = ccml_reshape(rhs, (int[]){1, rhs->shape[0], rhs->shape[1], 1});

    ccml_tensor * mul = ccml_mul(lhs_r, rhs_r);
    ccml_tensor * sum = ccml_sum(mul, 1, (int[]){1});

    return sum;
}

//
//  ██╗  ██╗ █████╗ ███████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗
//  ██║  ██║██╔══██╗██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗
//  ███████║███████║███████╗███████║██╔████╔██║███████║██████╔╝
//  ██╔══██║██╔══██║╚════██║██╔══██║██║╚██╔╝██║██╔══██║██╔═══╝
//  ██║  ██║██║  ██║███████║██║  ██║██║ ╚═╝ ██║██║  ██║██║
//  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝
//

#define CCML_FNV_PRIME 1099511628211LU
#define CCML_FNV_OFFSET 14695981039346656037LU

typedef struct ccml_hashmap_entry {
    uintptr_t key;
    int value;
} ccml_hashmap_entry;

typedef struct ccml_hashmap {
    int used;
    ccml_hashmap_entry entries[CCML_NODE_MAX];
    int capacity;
} ccml_hashmap;

static uint64_t ccml_hash_key(void * key) {
    uint64_t hash = CCML_FNV_OFFSET;
    hash ^= (uint64_t)(uintptr_t)key;
    hash *= CCML_FNV_PRIME;
    return hash;
}

static ccml_hashmap * ccml_new_hashmap() {
    int capacity = CCML_NODE_MAX;

    ccml_hashmap * map = malloc(sizeof(ccml_hashmap));
    *map = (ccml_hashmap){
        .used = 0,
        .entries = {0},
        .capacity = capacity,
    };

    for (int i = 0; i < capacity; i++) {
        map->entries[i].key = 0;
        map->entries[i].value = -1;
    }

    return map;
}

static int ccml_hashmap_get(ccml_hashmap * map, void * key) {
    if (key == NULL) {
        return -1;
    }

    uint64_t hash = ccml_hash_key(key);
    int index = (int)(hash & (uint64_t)(map->capacity - 1));

    while (map->entries[index].key != 0) {
        if ((uintptr_t)key == map->entries[index].key) {
            return map->entries[index].value;
        }

        index++;
        if (index >= map->capacity) {
            index = 0;
        }
    }

    return -1;
};

static void ccml_hashmap_set(ccml_hashmap * map, void * key, int value) {
    if (map->used >= map->capacity) {
        assert(false && "hashmap size overflow");
    }

    uint64_t hash = ccml_hash_key(key);
    int index = (int)(hash & (int)(map->capacity - 1));

    while (map->entries[index].key != 0) {
        if ((uintptr_t)key == map->entries[index].key) {
            // Found key (it already exists), update value.
            map->entries[index].value = value;
            return;
        }
        // Key wasn't in this slot, move to next (linear
        // probing).
        index++;
        if (index >= map->capacity) {
            index = 0;
        }
    }

    map->entries[index].key = (uintptr_t)key;
    map->entries[index].value = value;
    map->used++;
}

//
//   ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
//  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
//  ██║  ███╗██████╔╝███████║██████╔╝███████║
//  ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
//  ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
//   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
//

typedef struct ccml_graph {
    int n_nodes;
    ccml_tensor * nodes[CCML_NODE_MAX];
    ccml_hashmap * map;
} ccml_graph;

static void ccml_graph_forward(struct ccml_graph * graph, ccml_tensor * tensor,
                            int * node_counter) {
    if (tensor == NULL) return;

    // also checking if tensor has itself as a child to prevent (infinite)
    // cycles

    if (ccml_hashmap_get(graph->map, tensor->src[0]) == -1) {
        ccml_graph_forward(graph, tensor->src[0], node_counter);
    }
    if (ccml_hashmap_get(graph->map, tensor->src[1]) == -1) {
        ccml_graph_forward(graph, tensor->src[1], node_counter);
    }

    if (ccml_hashmap_get(graph->map, tensor) == -1) {
        assert(*node_counter < CCML_NODE_MAX - 1 && "more nodes created than CCML_NODE_MAX");
        tensor->index = *node_counter;
        graph->nodes[*node_counter] = tensor;
        ccml_hashmap_set(graph->map, tensor, (*node_counter)++);
    }
}

static void ccml_graph_backward(ccml_graph * graph, ccml_tensor * root) {
    if (root->requires_grad == false) return;
    root->grad = ccml_new_tensor_impl(root->type, root->shape);
    ccml_fill(root->grad, 1.0f);

    // in this loop create gradient tensors corresponding to each tensor
    // that requires gradient tracking, and set their buffers to the correct
    // option respectively (because, for example, a intermediary tensor w/o
    // a buffer also needs an intermediary gradient tensor w/o a buffer)

    ccml_tensor * queue[CCML_NODE_MAX] = {NULL};
    int queue_start = 0;
    int queue_end = 0;
    queue[queue_end++] = root;

    while (queue_end != queue_start) {
        ccml_tensor * tensor = queue[queue_start++];
        if (tensor->requires_grad == true) {
            // declaring partials d(tensor)/d(tensor->src[0]) and
            // d(tensor)/d(tensor->src[1])

            ccml_tensor * one = ccml_new_tensor_impl(tensor->type, (int[]){1, 1, 1, 1});
            ccml_tensor * two = ccml_new_tensor_impl(tensor->type, (int[]){1, 1, 1, 1});

            ccml_fill(one, 1.0f);
            ccml_fill(two, 2.0f);

            ccml_tensor * partial_0 = one;
            ccml_tensor * partial_1 = one;

            // calculating partials

            switch (tensor->oper) {
                case CCML_OPER_NONE:
                    partial_0 = one; break;
                case CCML_OPER_LOG:
                    partial_0 = ccml_rec(tensor->src[0]); break;
                case CCML_OPER_EXP:
                    partial_0 = ccml_exp(tensor->src[0]); break;
                case CCML_OPER_SIN:
                    partial_0 = ccml_cos(tensor->src[0]); break;
                case CCML_OPER_REC:
                    partial_0 = ccml_neg(ccml_rec(ccml_square(tensor->src[0]))); break;
                case CCML_OPER_SQRT:
                    partial_0 = ccml_rec(ccml_mul(two, ccml_sqrt(tensor->src[0]))); break;
                case CCML_OPER_ADD:
                    partial_0 = one; partial_1 = one; break;
                case CCML_OPER_MUL:
                    partial_0 = tensor->src[1]; partial_1 = tensor->src[0]; break;
                case CCML_OPER_RES:
                    partial_0 = one; break;
                case CCML_OPER_PER:
                    partial_0 = one; break;
                case CCML_OPER_SUM:
                    partial_0 = one; break;
                default:
                    assert(false && "unknown variant of ccml_oper");
            }

            // multiplying tensor->grad by partials and adding them to
            // the gradients of the tensor's children (we have to do a
            // mini DFS traversal w/ ccml_graph_forward() since the gradient
            // calculation forms a mini sub-graph that needs to be tra-
            // versed separately)

            if (tensor->src[0] != NULL) {
                if (ccml_is_leaf(tensor->src[0])) {
                    tensor->src[0]->grad = ccml_add(ccml_mul(tensor->grad, partial_0), tensor->src[0]->grad);
                } else {
                    tensor->src[0]->grad = ccml_mul(ccml_mul(tensor->grad, partial_0), tensor->src[0]->grad);
                }
                ccml_fill(tensor->src[0]->grad, 1.0f);
                ccml_graph_forward(graph, tensor->src[0]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[0];
            }
            if (tensor->src[1] != NULL) {
                if (ccml_is_leaf(tensor->src[1])) {
                    tensor->src[1]->grad = ccml_add(ccml_mul(tensor->grad, partial_1), tensor->src[1]->grad);
                } else {
                    tensor->src[1]->grad = ccml_mul(ccml_mul(tensor->grad, partial_1), tensor->src[1]->grad);
                }
                ccml_fill(tensor->src[1]->grad, 1.0f);
                ccml_graph_forward(graph, tensor->src[1]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[1];
            }
        }
    }
}

static void ccml_graph_allocate_nodes(ccml_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        int size = ccml_size(tensor);
        if (ccml_has_buffer(tensor) && tensor->data == NULL) {
            tensor->data = malloc(size * ccml_type_sizes[tensor->type]);
        }
        if (tensor->buff == CCML_BUFF_PERM && tensor->grad != NULL) {
            tensor->grad->buff = CCML_BUFF_PERM;
            tensor->grad->data = malloc(size * ccml_type_sizes[tensor->type]);
        }
    }
}

// common subexpression elimination
static void ccml_graph_cse(ccml_graph * graph) {
    // an expression takes the following format
    // (oper-int) * 10^8 + (src0-index) * 10^4 + (src1-index)

    assert(CCML_NODE_MAX < 9999 && "ccml_graph_cse must be adjusted w/ the increased node count");

    // erasing graph->map hashmap
    for (int i = 0; i < graph->map->capacity; i++) {
        graph->map->used = 0;
        graph->map->entries[i].key = 0;
        graph->map->entries[i].value = -1;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_is_leaf(tensor) == false) {
            int operation = tensor->oper;
            int src_0 = tensor->src[0] == NULL ? CCML_NODE_MAX : tensor->src[0]->index;
            int src_1 = tensor->src[1] == NULL ? CCML_NODE_MAX : tensor->src[1]->index;
            int expression = operation * pow(10, 8) + src_0 * pow(10, 4) + src_1;

            if (ccml_hashmap_get(graph->map, (void *)(uintptr_t)expression) == -1) {
                ccml_hashmap_set(graph->map, (void *)(uintptr_t)expression, tensor->index);
            } else {
                int index = ccml_hashmap_get(graph->map, (void *)(uintptr_t)expression);
                tensor->src[0] = graph->nodes[index];
                tensor->src[1] = NULL;
                tensor->oper = CCML_OPER_NONE;
            }
        }
    }
}

static ccml_graph * ccml_new_graph(ccml_tensor * root) {
    struct ccml_graph * graph = malloc(sizeof(struct ccml_graph));
    root->buff = CCML_BUFF_PERM;

    *graph = (struct ccml_graph){
        /*.n_nodes =*/ 0,
        /*.nodes   =*/ {NULL},
        /*.map     =*/ ccml_new_hashmap()
    };

    ccml_graph_forward(graph, root, &graph->n_nodes);
    ccml_graph_backward(graph, root);

    ccml_graph_cse(graph);
    ccml_graph_allocate_nodes(graph);

    return graph;
};

static void ccml_graph_free(ccml_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        // only freeing when tensor isn't of type reshape/permute, because those tensors
        // just use their children's data pointer, so we avoid a double free this way :)
        free(tensor->data);
        free(tensor);
    }
    free(graph);
}

//
//  ██╗███╗   ██╗██████╗ ██╗ ██████╗███████╗███████╗
//  ██║████╗  ██║██╔══██╗██║██╔════╝██╔════╝██╔════╝
//  ██║██╔██╗ ██║██║  ██║██║██║     █████╗  ███████╗
//  ██║██║╚██╗██║██║  ██║██║██║     ██╔══╝  ╚════██║
//  ██║██║ ╚████║██████╔╝██║╚██████╗███████╗███████║
//  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝╚══════╝╚══════╝
//

#define ccml_index(tensor, condition) ({                                         \
    int size = CCML_CHAR_MAX * CCML_DIMS_MAX;                                    \
    char * result = malloc(size * sizeof(char));                                 \
    *result = '\0';                                                              \
    strncat(result, "[", size);                                                  \
                                                                                \
    for (int i = 0; i < ccml_ndim(tensor); i++) {                                \
        snprintf(result + strlen(result), size - strlen(result), "%sid%d*%d*%d", \
                i != 0 && i != ccml_ndim(tensor) ? "+" : "", i,                 \
                tensor->stride[i], (condition) ? 1 : 0);                        \
    }                                                                            \
                                                                                \
    strncat(result + strlen(result), "]", size - strlen(result));                \
    if ((!ccml_has_buffer(tensor) && !ccml_is_permute(tensor)) ||                \
        ccml_size(tensor) == 1) {                                               \
        snprintf(result, size, "");                                              \
    }                                                                            \
    result;                                                                      \
})

//
//  ██████╗  █████╗  ██████╗██╗  ██╗███████╗███╗   ██╗██████╗
//  ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝████╗  ██║██╔══██╗
//  ██████╔╝███████║██║     █████╔╝ █████╗  ██╔██╗ ██║██║  ██║
//  ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██║╚██╗██║██║  ██║
//  ██████╔╝██║  ██║╚██████╗██║  ██╗███████╗██║ ╚████║██████╔╝
//  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝
//
//   ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗
//  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║
//  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║
//  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║
//  ╚██████╔╝██║     ███████╗██║ ╚████║╚██████╗███████╗
//   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
//

static const char * ccml_oper_metal(ccml_oper oper) {
    switch (oper) {
        case CCML_OPER_LOG: return "log";
        case CCML_OPER_EXP: return "exp";
        case CCML_OPER_SIN: return "sin";
        case CCML_OPER_REC: return "1/";
        case CCML_OPER_SQRT: return "sqrt";
        case CCML_OPER_ADD: return "+";
        case CCML_OPER_MUL: return "*";
        default: assert(false && "no meaningful conversion to string exists");
    }
}

static const char * ccml_type_metal(ccml_type type) {
    switch (type) {
        case CCML_TYPE_FP32: return "float ";
        default: assert(false && "unknown variant of ccml_type");
    }
}

static const char * ccml_kernel_metal(struct ccml_graph * graph) {
    int size = CCML_NODE_MAX * CCML_CHAR_MAX;
    char * kernel = malloc(size * sizeof(char));
    char * string = kernel;
    *string = '\0';

    // atomic floating point addition function bc metal doesn't support it natively
    string += snprintf(string, size - (kernel - string),
                    "#include <metal_stdlib>\n#include <metal_atomic>\n"
                    "using namespace metal;\n\nkernel void my_kernel(");

    // adding kernel input parameters to the kernel string

    int n_kernel_parameters = 0;
    int largest_tensor = 0;

    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_size(tensor) > ccml_size(graph->nodes[largest_tensor]))
            largest_tensor = i;

        if (ccml_has_buffer(tensor) && ccml_size(tensor) != 1) {
            if (tensor->oper == CCML_OPER_SUM) {}
            if (n_kernel_parameters == 0) {
                string += snprintf(string, size - (kernel - string), "device %s%s* data_%d [[buffer(0)]]",
                                tensor->oper == CCML_OPER_SUM ? "atomic_" : "", ccml_type_metal(tensor->type), i);
                n_kernel_parameters++;
            } else {
                string += snprintf(string, size - (kernel - string), ", device %s%s* data_%d [[buffer(%d)]]",
                                tensor->oper == CCML_OPER_SUM ? "atomic_" : "",
                                ccml_type_metal(tensor->type), i, n_kernel_parameters);
                n_kernel_parameters++;
            }
        }
    }

    string += snprintf(string, size - (kernel - string),
                    ", uint3 gid [[thread_position_in_grid]]) {\n"
                    "\tuint id0 = gid.x / %d;\n\tuint id1 = gid.x %% %d;\n"
                    "\tuint id2 = gid.y;\n\tuint id3 = gid.z;\n\n",
                    graph->nodes[0]->shape[1], graph->nodes[0]->shape[1]);

    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];

        switch (tensor->oper) {
            case CCML_OPER_NONE:
                if (ccml_is_leaf(tensor) == false) {
                    string += snprintf(string, size - (kernel - string), "\t%s%s data_%d%s = data_%d%s;\n",
                                    ccml_type_metal(tensor->type), ccml_has_buffer(tensor) ? "*" : "",
                                    tensor->index, ccml_index(tensor, true), tensor->src[0]->index,
                                    ccml_index(tensor->src[0], true));
                }
                // tensor data is embeddeable directly into the kernel string
                if (ccml_has_buffer(tensor) && ccml_size(tensor) == 1) {
                    string += snprintf(string, size - (kernel - string), "\t%sdata_%d = %f;\n",
                                    ccml_type_metal(tensor->type), i, *(float *)tensor->data);
                }
                break;
            case CCML_OPER_LOG:
            case CCML_OPER_EXP:
            case CCML_OPER_SIN:
            case CCML_OPER_REC:
            case CCML_OPER_SQRT:
                string += snprintf(string, size - (kernel - string), "\t%sdata_%d%s = ",
                                ccml_has_buffer(tensor) ? "" : ccml_type_metal(tensor->type),
                                tensor->index, ccml_index(tensor, true));
                string += snprintf(string, size - (kernel - string), "%s(data_%d%s);\n",
                                ccml_oper_metal(tensor->oper),
                                tensor->src[0]->index, ccml_index(tensor->src[0], true));
                break;
            case CCML_OPER_ADD:
            case CCML_OPER_MUL:
                string += snprintf(string, size - (kernel - string), "\t%sdata_%d%s = ",
                                ccml_has_buffer(tensor) ? "" : ccml_type_metal(tensor->type),
                                tensor->index, ccml_index(tensor, true));
                string += snprintf(string, size - (kernel - string), "data_%d%s %s ",
                                tensor->src[0]->index, ccml_index(tensor->src[0], tensor->src[0]->shape[i] != 1),
                                ccml_oper_metal(tensor->oper));
                string += snprintf(string, size - (kernel - string), "data_%d%s;\n",
                                tensor->src[1] != NULL ? tensor->src[1]->index : tensor->index,
                                tensor->src[1] != NULL ? ccml_index(tensor->src[1], tensor->src[1]->shape[i] != 1) :
                                ccml_index(tensor, true));
                break;
            case CCML_OPER_RES:
            case CCML_OPER_PER:
                string += snprintf(string, size - (kernel - string), "\t%s* data_%d = ",
                                ccml_type_metal(tensor->type), tensor->index);
                string += snprintf(string, size - (kernel - string), "data_%d;\n",
                                tensor->src[0]->index);
                break;
            case CCML_OPER_SUM:
                string += snprintf(string, size - (kernel - string),
                                "\tatomic_fetch_add_explicit(&data_%d%s, ",
                                tensor->index, ccml_index(tensor, tensor->shape[i] != 1));
                string += snprintf(string, size - (kernel - string), "data_%d%s, memory_order_relaxed);\n",
                                tensor->src[0]->index,
                                ccml_index(tensor->src[0], tensor->shape[i] != tensor->src[0]->shape[i]));
                break;
            default: assert(false && "unknown variant of ccml_oper");
        }
    }

    string += snprintf(string, size - (kernel - string), "}");
    return kernel;
}

static void ccml_code_metal(ccml_graph * graph) {
    FILE * file_ptr = fopen("metal.m", "w");
    assert(file_ptr != NULL && "failed to create file for metal source");

    FILE * file_kernel_ptr = fopen("kernel.metal", "w");
    assert(file_kernel_ptr != NULL && "failed to create a file for kernel source");

    fprintf(file_kernel_ptr, "%s", ccml_kernel_metal(graph));

    fprintf(file_ptr, "#import <MetalKit/MetalKit.h>\n"
                    "#import <Foundation/Foundation.h>\n\n"
                    "int main(int argc, const char * argv[]) {\n"
                    "\t@autoreleasepool {\n"
                    "\t\t// setup\n"
                    "\t\tid<MTLDevice> device = MTLCreateSystemDefaultDevice();\n"
                    "\t\tid<MTLCommandQueue> commandQueue = [device newCommandQueue];\n"
                    "\t\tid<MTLLibrary> library = [device newDefaultLibrary];\n"
                    "\t\tid<MTLFunction> kernelFunction = [library newFunctionWithName:@\"my_kernel\"];\n\n"
                    "\t\t// pipeline\n"
                    "\t\tNSError *error = NULL;\n"
                    "\t\t[commandQueue commandBuffer];\n"
                    "\t\tid<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];\n"
                    "\t\tid<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];\n"
                    "\t\t[encoder setComputePipelineState:[device newComputePipelineStateWithFunction:kernelFunction error:&error]];\n\n");

    int buffer_counter = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (tensor->buff == CCML_BUFF_CNST && ccml_size(tensor) != 1) {
            fprintf(file_ptr, "\t\t%sbuff_%d[%d] = {%f",
                    ccml_type_metal(tensor->type), tensor->index, ccml_size(tensor), *((float*)tensor->data));
            for (int j = 1; j < ccml_size(tensor); j++) {
                fprintf(file_ptr, ", %f", *((float*)tensor->data + j));
            }
            fprintf(file_ptr, "};\n");
            fprintf(file_ptr, "\t\t[encoder setBuffer:[device newBufferWithBytes:buff_%d length:%d * sizeof(%s) options:0] offset:0 atIndex:%d];\n\n",
                            tensor->index, ccml_size(tensor), ccml_type_metal(tensor->type), buffer_counter++);

        }
        if (tensor->buff == CCML_BUFF_PERM) {
            fprintf(file_ptr, "\t\tid<MTLBuffer> data_%d = [device newBufferWithLength:%d * sizeof(%s) options:0];\n"
                            "\t\t[encoder setBuffer:data_%d offset:0 atIndex:%d];\n",
                            tensor->index, ccml_size(tensor), ccml_type_metal(tensor->type), tensor->index, buffer_counter++);
        }
    }

    fprintf(file_ptr, "\n\t\tMTLSize numThreadgroups = {%d, %d, %d};\n"
                    "\t\tMTLSize numgroups = {1,1,1};\n"
                    "\t\t[encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:numgroups];\n"
                    "\t\t[encoder endEncoding];\n"
                    "\t\t[commandBuffer commit];\n"
                    "\t\t[commandBuffer waitUntilCompleted];\n\n",
                    graph->nodes[0]->shape[0] * graph->nodes[0]->shape[1],
                    graph->nodes[0]->shape[2], graph->nodes[0]->shape[3]);

    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (tensor->buff == CCML_BUFF_PERM) {
            fprintf(file_ptr, "\t\t%s* buff_%d = [data_%d contents];\n\n", ccml_type_metal(tensor->type), tensor->index, tensor->index);
            fprintf(file_ptr, "\t\tfor (int i = 0; i < %d; i++) {\n"
                            "\t\t\tprintf(\"%%f \", buff_%d[i]);\n\t\t}\n",
                            ccml_size(tensor), tensor->index);
        }
    }

    fprintf(file_ptr, "\n\t}\n\treturn 0;\n}");

    fclose(file_ptr);
    fclose(file_kernel_ptr);
}

//
//  ███████╗██╗  ██╗███████╗ ██████╗██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
//  ██╔════╝╚██╗██╔╝██╔════╝██╔════╝██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
//  █████╗   ╚███╔╝ █████╗  ██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
//  ██╔══╝   ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
//  ███████╗██╔╝ ██╗███████╗╚██████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
//  ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
//

static void ccml_graph_execute(ccml_graph * graph, ccml_backend backend) {
    // currently only macos and linux are supported
    #if !defined(__APPLE__)
        #error "platform not supported"
    #endif

    switch(backend) {
        case CCML_BACKEND_METAL: ccml_code_metal(graph); break;
        default: assert(false && "unknown backend");
    }

    // reading tensors back from their respective files
    /*
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (tensor->buff == CCML_BUFF_PERM) {
            char file_name[CCML_CHAR_MAX];
            snprintf(file_name, CCML_CHAR_MAX, "tensor_%d", tensor->index);
            FILE * file_ptr = fopen(file_name, "rb");
            assert(file_ptr != NULL && "couldn't open file with tensor data");
            fread(tensor->data, ccml_size(tensor), ccml_type_sizes[tensor->type], file_ptr);
            char command[CCML_CHAR_MAX] = "rm ";
            strncat(command, file_name, CCML_CHAR_MAX);
            system(command);
        }
    }
    */
}

#endif