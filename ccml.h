#if !defined(CCML_IMPL)
#define CCML_IMPL

#define __STDC_WANT_IEC_60559_TYPES_EXT__

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdalign.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199409L) && !defined(CCML_API)
    #define CCML_API static inline
#elif !defined(CCML_API)
    #define CCML_API static
#endif

#define CCML_ASSERT(x, ...) do { if (!(x)) {                                               \
    fprintf(stderr, "CCML_ASSERT: %s:%d: %s ", __FILE__, __LINE__, #x);                    \
    __VA_OPT__(fprintf(stderr, __VA_ARGS__);)                                              \
    exit(EXIT_FAILURE);                                                                    \
} } while (0)

#define CCML_SRCS_MAX 2
#define CCML_TYPE_MAX 3
#define CCML_DIMS_MAX 4
#define CCML_KERN_MAX 16
#define CCML_CHAR_MAX 80
#define CCML_NODE_MAX 128

// KNOWN ISSUES
// - including ccml.h in separate compilation units compiles separate/independent symbols
// - a lot of function return statuses aren't checked, mostly snprintf/fread/fwrite
// - openCL backend has no error checking at the moment
// - metal backend has incomplete error checkingat the moment

// TO DO
// - support for dynamic node count
// - more backends

//
//   ██████╗ ██████╗ ███╗   ██╗████████╗███████╗██╗  ██╗████████╗
//  ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
//  ██║     ██║   ██║██╔██╗ ██║   ██║   █████╗   ╚███╔╝    ██║
//  ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══╝   ██╔██╗    ██║
//  ╚██████╗╚██████╔╝██║ ╚████║   ██║   ███████╗██╔╝ ██╗   ██║
//   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝
//

typedef struct ccml_context {
    int capacity;
    int used;
    void * memory;
} ccml_context;

CCML_API ccml_context * ccml_new_context(int capacity) {
    CCML_ASSERT(capacity > 0);
    void * memory = malloc(capacity);
    int max_align = alignof(max_align_t);
    ccml_context * ctx = memory;

    *ctx = (ccml_context) {
        .capacity = capacity,
        .used     = (sizeof(ccml_context) / max_align + 1) * max_align,
        .memory   = memory
    };

    return ctx;
}

CCML_API void * ccml_malloc(ccml_context * ctx, int size) {
    int max_align = alignof(max_align_t);
    int size_aligned = (size / max_align + 1) * max_align;

    CCML_ASSERT(ctx->used + size_aligned < ctx->capacity,
                "needed %d bytes, available %d bytes",
                ctx->used + size_aligned, ctx->capacity);

    void * ptr = ctx->memory + ctx->used;
    ctx->used += size_aligned;
    return ptr;
}

CCML_API void ccml_context_free(ccml_context * ctx) {
    free(ctx->memory);
}

//
//  ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗
//  ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
//     ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
//     ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
//     ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
//     ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
//

typedef enum ccml_type {
    CCML_TYPE_FP32 = -1
} ccml_type;

typedef enum ccml_grad {
    CCML_GRAD_NO  = -2,
    CCML_GRAD_YES = -3
} ccml_grad;

CCML_API bool ccml_is_grad(int value) {
    return value >= CCML_GRAD_YES && value <= CCML_GRAD_NO;
}

CCML_API bool ccml_is_type(int value) {
    return value >= CCML_TYPE_FP32 && value <= CCML_TYPE_FP32;
}

typedef enum ccml_oper {
    CCML_OPER_LOG,
    CCML_OPER_EXP,
    CCML_OPER_SIN,
    CCML_OPER_REC,
    CCML_OPER_SQT,
    CCML_OPER_ADD,
    CCML_OPER_MUL,
    CCML_OPER_SUM,
    CCML_OPER_RES,
    CCML_OPER_PER,
    CCML_OPER_LOAD,
    CCML_OPER_INTR,
    CCML_OPER_SAVE,
} ccml_oper;

typedef struct ccml_tensor {
    enum ccml_type type;
    enum ccml_oper oper;

    int shape[CCML_DIMS_MAX];
    int stride[CCML_DIMS_MAX];

    bool has_gradient;
    int index;
    float * data;

    struct ccml_tensor * grad;
    struct ccml_tensor * src[CCML_SRCS_MAX];
} ccml_tensor;

CCML_API ccml_tensor * ccml_new_tensor_impl(ccml_context * ctx, ccml_type type,
                                            ccml_oper oper, int * shape) {
    ccml_tensor * result = ccml_malloc(ctx, sizeof(ccml_tensor));
    int stride[CCML_DIMS_MAX] = {
        shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1
    };

    *result = (ccml_tensor) {
        .type   = type,
        .oper   = oper,
        .shape  = {shape[0], shape[1], shape[2], shape[3]},
        .stride = {stride[0], stride[1], stride[2], stride[3]},
        .index  = -1
    };

    return result;
}

#define ccml_new_tensor(ctx, ne0, ...) ({                                                  \
    int input[CCML_DIMS_MAX + 2] = {ne0, __VA_ARGS__};                                     \
    int shape[CCML_DIMS_MAX] = {ne0, 1, 1, 1};                                             \
    ccml_type type = CCML_TYPE_FP32;                                                       \
    int has_gradient = false;                                                              \
    int dim_counter = 0;                                                                   \
                                                                                           \
    for (int i = 0; i < CCML_DIMS_MAX; i++) {                                              \
        if (input[i] > 0) shape[dim_counter++] = input[i];                                 \
        if (ccml_is_type(input[i])) type = input[i];                                       \
        if (ccml_is_grad(input[i])) has_gradient = input[i];                               \
    }                                                                                      \
                                                                                           \
    CCML_ASSERT(dim_counter < CCML_DIMS_MAX);                                              \
                                                                                           \
    ccml_tensor * tensor = ccml_new_tensor_impl(ctx, type, CCML_OPER_LOAD, shape);         \
    if (has_gradient == -2) tensor->has_gradient = false;                                  \
    if (has_gradient == -3) tensor->has_gradient = true;                                   \
    if (type         == -1) tensor->type         = CCML_TYPE_FP32;                         \
    tensor->oper == CCML_OPER_LOAD;                                                        \
                                                                                           \
    tensor;                                                                                \
})

CCML_API int ccml_size(ccml_tensor * tensor) {
    int size = 1;
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        size *= tensor->shape[i];
    }

    return size;
}

CCML_API bool ccml_has_buffer(ccml_tensor * tensor) {
    switch (tensor->oper) {
        case CCML_OPER_LOAD:
        case CCML_OPER_INTR:
        case CCML_OPER_SAVE: return true;
        default: return false;
    }
}

CCML_API void ccml_fill(ccml_context * ctx, ccml_tensor * tensor, float value) {
    CCML_ASSERT(ccml_has_buffer(tensor) && tensor->data == NULL);

    int size = ccml_size(tensor);
    tensor->oper = CCML_OPER_LOAD;
    tensor->data = ccml_malloc(ctx, size * sizeof(float));
    for (int i = 0; i < size; i++) {
        tensor->data[i] = value;
    }
}

CCML_API ccml_tensor * ccml_scalar(ccml_context * ctx, float value) {
    ccml_tensor * scalar = ccml_new_tensor_impl(ctx, CCML_TYPE_FP32, CCML_OPER_INTR, (int []){1, 1, 1, 1});
    ccml_fill(ctx, scalar, value);

    return scalar;
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

CCML_API bool ccml_can_broadcast(ccml_tensor * lhs, ccml_tensor * rhs) {
    if (rhs == NULL || lhs == NULL) return true;
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if (lhs->shape[i] != rhs->shape[i] && lhs->shape[i] != 1 && rhs->shape[i] != 1) return false;
    }

    return true;
}

CCML_API bool ccml_is_leaf(ccml_tensor * tensor) {
    return tensor->src[0] == NULL && tensor->src[1] == NULL;
}

CCML_API int ccml_dim(ccml_tensor * tensor) {
    int last_dim = 0;
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if (tensor->shape[i] != 1) last_dim = i;
    }

    return last_dim + 1;
}

CCML_API bool ccml_is_matrix(ccml_tensor * tensor) {
    return tensor->shape[0] != 1 && tensor->shape[1] != 1 && tensor->shape[2] == 1 && tensor->shape[3] == 1;
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

CCML_API ccml_tensor * ccml_log(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_LOG, tensor->shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;

    return result;
}

CCML_API ccml_tensor * ccml_exp(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_EXP, tensor->shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;

    return result;
}

CCML_API ccml_tensor * ccml_sin(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_SIN, tensor->shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;

    return result;
}

CCML_API ccml_tensor * ccml_rec(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_REC, tensor->shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;

    return result;
}

CCML_API ccml_tensor * ccml_sqrt(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_SQT, tensor->shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;

    return result;
}

CCML_API ccml_tensor * ccml_add(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_can_broadcast(lhs, rhs), "incompatible dimensions for broadcasting");
    bool null_input = lhs == NULL || rhs == NULL;
    int shape[CCML_DIMS_MAX] = {0};
    bool has_gradient = lhs->has_gradient || (null_input ? false : rhs->has_gradient);

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[i] :
        (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    ccml_tensor * result = ccml_new_tensor_impl(ctx, lhs->type, CCML_OPER_ADD, shape);
    result->src[0]       = lhs;
    result->src[1]       = rhs;
    result->has_gradient = has_gradient;

    return result;
}

CCML_API ccml_tensor * ccml_mul(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_can_broadcast(lhs, rhs), "incompatible dimensions for broadcasting");
    bool null_input = lhs == NULL || rhs == NULL;
    int shape[CCML_DIMS_MAX] = {0};
    bool has_gradient = lhs->has_gradient || (null_input ? false : rhs->has_gradient);

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[i] :
        (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    ccml_tensor * result = ccml_new_tensor_impl(ctx, lhs->type, CCML_OPER_MUL, shape);
    result->src[0]       = lhs;
    result->src[1]       = rhs;
    result->has_gradient = has_gradient;

    return result;
}

CCML_API ccml_tensor * ccml_reshape(ccml_context * ctx, ccml_tensor * tensor, int * shape) {
    int size = ccml_size(tensor);
    int new_size = shape[0] * shape[1] * shape[2] * shape[3];
    CCML_ASSERT(size == new_size, "reshaped and source tensor must have the same size");

    int stride[CCML_DIMS_MAX] = {
        shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1
    };

    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_RES, shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        result->shape[i]  = shape[i];
        result->stride[i] = stride[i];
    }

    return result;
}

CCML_API ccml_tensor * ccml_permute(ccml_context * ctx, ccml_tensor * tensor, int * perm) {
    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_PER, tensor->shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;


    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        result->shape[i]  = tensor->shape[perm[i]];
        result->stride[i] = tensor->stride[perm[i]];
    }

    return result;
}

CCML_API ccml_tensor * ccml_sum(ccml_context * ctx, ccml_tensor * tensor, int n_axes, int * axes) {
    CCML_ASSERT(n_axes >= 0 && n_axes <= CCML_DIMS_MAX, "invalid number of summed axes");
    int shape[CCML_DIMS_MAX] = {1, 1, 1, 1};

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = tensor->shape[i];
    }

    for (int i = 0; i < n_axes; i++) {
        shape[axes[i]] = 1;
    }

    ccml_tensor * result = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_SUM, shape);
    result->src[0]       = tensor;
    result->has_gradient = tensor->has_gradient;

    ccml_tensor * save = ccml_new_tensor_impl(ctx, result->type, CCML_OPER_INTR, result->shape);
    save->src[0]       = result;
    save->has_gradient = result->has_gradient;

    return save;
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

CCML_API ccml_tensor * ccml_neg(ccml_context * ctx, ccml_tensor * tensor) {
    return ccml_log(ctx, ccml_rec(ctx, ccml_exp(ctx, tensor)));
}

CCML_API ccml_tensor * ccml_sub(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_add(ctx, lhs, ccml_neg(ctx, rhs));
}

CCML_API ccml_tensor * ccml_div(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_mul(ctx, lhs, ccml_rec(ctx, rhs));
}

CCML_API ccml_tensor * ccml_square(ccml_context * ctx, ccml_tensor * tensor) {
    return ccml_mul(ctx, tensor, tensor);
}

CCML_API ccml_tensor * ccml_cos(ccml_context * ctx, ccml_tensor * tensor) {
    return ccml_sin(ctx, ccml_add(ctx, tensor, ccml_scalar(ctx, M_PI_2)));
}

CCML_API ccml_tensor * ccml_tanh(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * exp_neg = ccml_neg(ctx, ccml_exp(ctx, tensor));
    ccml_tensor * exp = ccml_exp(ctx, tensor);

    return ccml_div(ctx, ccml_sub(ctx, exp, exp_neg), ccml_add(ctx, exp, exp_neg));
}

CCML_API ccml_tensor * ccml_matmul(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_is_matrix(lhs) && ccml_is_matrix(rhs));
    CCML_ASSERT(lhs->shape[1] == rhs->shape[0]);
    ccml_tensor * lhs_r = ccml_reshape(ctx, lhs, (int[]){lhs->shape[0], lhs->shape[1], 1, 1});
    ccml_tensor * rhs_r = ccml_reshape(ctx, rhs, (int[]){1, rhs->shape[0], rhs->shape[1], 1});
    ccml_tensor * mul_r = ccml_mul(ctx, lhs_r, rhs_r);
    ccml_tensor * sum_r = ccml_sum(ctx, mul_r, 1, (int[]){1});
    ccml_tensor * res_r = ccml_reshape(ctx, sum_r, (int[]){sum_r->shape[0], sum_r->shape[2], 1, 1});

    return res_r;
}

CCML_API ccml_tensor * ccml_soft_max(ccml_context * ctx, ccml_tensor * tensor) {
    int dims[CCML_DIMS_MAX] = {0, 1, 2, 3};

    return ccml_div(ctx, ccml_exp(ctx, tensor), ccml_sum(ctx, ccml_exp(ctx, tensor), 4, dims));
}

CCML_API ccml_tensor * ccml_cross_entropy_loss(ccml_context * ctx, ccml_tensor * tensor, ccml_tensor * target) {
    CCML_ASSERT(ccml_size(tensor) == ccml_size(target));
    int dims[CCML_DIMS_MAX] = {0, 1, 2, 3};

    return ccml_neg(ctx, ccml_sum(ctx, ccml_mul(ctx, target, ccml_log(ctx, tensor)), 4, dims));
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

CCML_API uint64_t ccml_hash_key(void * key) {
    uint64_t hash = CCML_FNV_OFFSET;
    hash ^= (uint64_t)(uintptr_t)key;
    hash *= CCML_FNV_PRIME;
    return hash;
}

CCML_API ccml_hashmap * ccml_new_hashmap(ccml_context * ctx) {
    int capacity = CCML_NODE_MAX;
    ccml_hashmap * map = ccml_malloc(ctx, sizeof(ccml_hashmap));

    *map = (ccml_hashmap) {
        .used     = 0,
        .entries  = {0},
        .capacity = capacity,
    };

    for (int i = 0; i < capacity; i++) {
        map->entries[i].key = 0;
        map->entries[i].value = -1;
    }

    return map;
}

CCML_API int ccml_hashmap_get(ccml_hashmap * map, void * key) {
    if (key == NULL) return -1;
    uint64_t hash = ccml_hash_key(key);
    int id = (int)(hash & (uint64_t)(map->capacity - 1));

    while (map->entries[id].key != 0) {
        if ((uintptr_t)key == map->entries[id].key) {
            return map->entries[id].value;
        }
        id++;
        if (id >= map->capacity) {
            id = 0;
        }
    }

    return -1;
};

CCML_API void ccml_hashmap_set(ccml_hashmap * map, void * key, int value) {
    if (map->used >= map->capacity) CCML_ASSERT(false, "hashmap size overflow");
    uint64_t hash = ccml_hash_key(key);
    int id = (int)(hash & (int)(map->capacity - 1));
    while (map->entries[id].key != 0) {

        if ((uintptr_t)key == map->entries[id].key) {
            // Found key (it already exists), update value.
            map->entries[id].value = value;
            return;
        }
        id++;
        if (id >= map->capacity) {
            id = 0;
        }
    }

    map->entries[id].key = (uintptr_t)key;
    map->entries[id].value = value;
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
    ccml_context * context;
} ccml_graph;

CCML_API void ccml_graph_forward(ccml_graph * graph, ccml_tensor * tensor, int * node_counter) {
    if (tensor == NULL) return;
    if (ccml_hashmap_get(graph->map, tensor->src[0]) == -1) {
        ccml_graph_forward(graph, tensor->src[0], node_counter);
    }
    if (ccml_hashmap_get(graph->map, tensor->src[1]) == -1) {
        ccml_graph_forward(graph, tensor->src[1], node_counter);
    }
    if (ccml_hashmap_get(graph->map, tensor) == -1) {
        CCML_ASSERT(*node_counter < CCML_NODE_MAX - 1, "more nodes created than CCML_NODE_MAX");
        tensor->index = *node_counter;
        graph->nodes[*node_counter] = tensor;
        ccml_hashmap_set(graph->map, tensor, (*node_counter)++);
    }
}

CCML_API void ccml_graph_backward(ccml_context * ctx, ccml_graph * graph, ccml_tensor * root) {
    if (root->has_gradient == false) return;
    root->grad = ccml_scalar(ctx, 1.0f);

    ccml_tensor * queue[CCML_NODE_MAX] = {NULL};
    int queue_start = 0;
    int queue_end = 0;
    queue[queue_end++] = root;

    while (queue_end != queue_start) {
        ccml_tensor * tensor = queue[queue_start++];
        if (tensor->has_gradient == true) {
            int n_dims_0 = 0;
            int n_dims_1 = 0;
            int dims_0[CCML_DIMS_MAX] = {1, 1, 1, 1};
            int dims_1[CCML_DIMS_MAX] = {1, 1, 1, 1};
            for (int i = 0; i < CCML_DIMS_MAX; i++) {
                if (tensor->src[0] != NULL && tensor->src[0]->shape[i] == 1 && tensor->shape[i] != 1) {
                    dims_0[n_dims_0++] = i;
                }
                if (tensor->src[1] != NULL && tensor->src[1]->shape[i] == 1 && tensor->shape[i] != 1) {
                    dims_1[n_dims_1++] = i;
                }
            }

            // calculating partials
            ccml_tensor * grads[CCML_SRCS_MAX] = {NULL};
            switch (tensor->oper) {
                case CCML_OPER_LOG:
                    grads[0] = ccml_mul(ctx, tensor->grad, ccml_rec(ctx, tensor->src[0])); break;
                case CCML_OPER_EXP:
                    grads[0] = ccml_mul(ctx, tensor->grad, ccml_exp(ctx, tensor->src[0])); break;
                case CCML_OPER_SIN:
                    grads[0] = ccml_mul(ctx, tensor->grad, ccml_cos(ctx, tensor->src[0])); break;
                case CCML_OPER_REC: {
                    ccml_tensor * square = ccml_square(ctx, tensor->src[0]);
                    grads[0] = ccml_mul(ctx, tensor->grad, ccml_neg(ctx, ccml_rec(ctx, square))); break; }
                case CCML_OPER_SQT: {
                    ccml_tensor * fraction = ccml_mul(ctx, ccml_scalar(ctx, 2.0f), ccml_sqrt(ctx, tensor->src[0]));
                    grads[0] = ccml_mul(ctx, tensor->grad, ccml_rec(ctx, fraction)); break; }
                case CCML_OPER_ADD:
                    grads[0] = ccml_sum(ctx, tensor->grad, n_dims_0, dims_0);
                    grads[1] = ccml_sum(ctx, tensor->grad, n_dims_1, dims_1); break;
                case CCML_OPER_MUL:
                    grads[0] = ccml_sum(ctx, ccml_mul(ctx, tensor->grad, tensor->src[1]), n_dims_0, dims_0);
                    grads[1] = ccml_sum(ctx, ccml_mul(ctx, tensor->grad, tensor->src[0]), n_dims_1, dims_1); break;
                case CCML_OPER_SUM:
                case CCML_OPER_PER:
                case CCML_OPER_RES:
                    grads[0] = tensor->grad;
                    break;
                case CCML_OPER_LOAD:
                    break;
                case CCML_OPER_INTR:
                case CCML_OPER_SAVE:
                    grads[0] = tensor->grad;
                    break;
            }

            if (tensor->src[0] != NULL && tensor->src[0]->has_gradient) {
                if (ccml_is_leaf(tensor->src[0])) {
                    tensor->src[0]->grad = ccml_add(ctx, grads[0], tensor->src[0]->grad);
                    ccml_fill(ctx, tensor->src[0]->grad, 0.0f);
                } else {
                    tensor->src[0]->grad = ccml_mul(ctx, grads[0], tensor->src[0]->grad);
                    ccml_fill(ctx, tensor->src[0]->grad, 1.0f);
                }
                ccml_graph_forward(graph, tensor->src[0]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[0];
            }
            if (tensor->src[1] != NULL && tensor->src[1]->has_gradient) {
                if (ccml_is_leaf(tensor->src[1])) {
                    tensor->src[1]->grad = ccml_add(ctx, grads[1], tensor->src[1]->grad);
                    ccml_fill(ctx, tensor->src[1]->grad, 0.0f);
                } else {
                    tensor->src[1]->grad = ccml_mul(ctx, grads[1], tensor->src[1]->grad);
                    ccml_fill(ctx, tensor->src[1]->grad, 1.0f);
                }
                ccml_graph_forward(graph, tensor->src[1]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[1];
            }
        }
    }
}

CCML_API void ccml_graph_allocate(ccml_context * ctx, ccml_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        int size = ccml_size(tensor);
        if (ccml_has_buffer(tensor) && tensor->data == NULL) {
            tensor->data = ccml_malloc(ctx, size * sizeof(float));
        }
        if (ccml_has_buffer(tensor) && tensor->has_gradient == true) {
            tensor->grad = ccml_new_tensor_impl(ctx, tensor->type, CCML_OPER_INTR, tensor->shape);
        }
    }
}

CCML_API ccml_graph * ccml_new_graph(ccml_context * ctx, ccml_tensor * root) {
    ccml_tensor * save = ccml_new_tensor_impl(ctx, root->type, CCML_OPER_SAVE, root->shape);
    save->src[0] = root;
    struct ccml_graph * graph = ccml_malloc(ctx, sizeof(struct ccml_graph));

    *graph = (struct ccml_graph) {
        .n_nodes = 0,
        .nodes   = {NULL},
        .map     = ccml_new_hashmap(ctx),
        .context = ctx
    };

    ccml_graph_forward(graph, save, &graph->n_nodes);
    ccml_graph_backward(ctx, graph, root);

    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i]->oper == CCML_OPER_LOAD && graph->nodes[i]->data == NULL) {
            CCML_ASSERT(false, "data not initialised for source tensors");
        }
    }

    ccml_graph_allocate(ctx, graph);

    return graph;
};

//
//  ██╗███╗   ██╗██████╗ ███████╗██╗  ██╗██╗███╗   ██╗ ██████╗
//  ██║████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝██║████╗  ██║██╔════╝
//  ██║██╔██╗ ██║██║  ██║█████╗   ╚███╔╝ ██║██╔██╗ ██║██║  ███╗
//  ██║██║╚██╗██║██║  ██║██╔══╝   ██╔██╗ ██║██║╚██╗██║██║   ██║
//  ██║██║ ╚████║██████╔╝███████╗██╔╝ ██╗██║██║ ╚████║╚██████╔╝
//  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
//

CCML_API const char * ccml_new_index(ccml_context * ctx, ccml_tensor * parent, ccml_tensor * child) {
    int size = CCML_CHAR_MAX;
    char * index = ccml_malloc(ctx, size * sizeof(char));
    *index = '\0';
    strncat(index, "[", size);

    for (int i = 0; i < ccml_dim(child); i++) {
        // fake dimension is a virtually broadcasted dimension (without actually duplicating/expanding it)
        bool dim_is_fake = false;
        if (parent != NULL) {
            dim_is_fake = parent->shape[i] == 1 && child->shape[i] != 1;
        }

        snprintf(index + strlen(index), size - strlen(index), "%sid%d*%d*%d",
                 i != 0 && i != ccml_dim(child) ? "+" : "", i, child->stride[i],
                 dim_is_fake ? 0 : 1);
    }

    strncat(index + strlen(index), "]", size - strlen(index));

    return index;
}

CCML_API void ccml_new_kernel_slice(ccml_graph * graph, int * n_kernels,
                                    int kernels[CCML_KERN_MAX][2]) {
    kernels[*n_kernels][0] = 0;
    // not sure yet what's the optimal strategy for kernel slicing,
    // therefore for now everything is fused into a single kernel
    kernels[*n_kernels][1] = graph->n_nodes;
    *n_kernels += 1;
}

//
//  ██████╗  █████╗  ██████╗██╗  ██╗███████╗███╗   ██╗██████╗
//  ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝████╗  ██║██╔══██╗
//  ██████╔╝███████║██║     █████╔╝ █████╗  ██╔██╗ ██║██║  ██║
//  ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██║╚██╗██║██║  ██║
//  ██████╔╝██║  ██║╚██████╗██║  ██╗███████╗██║ ╚████║██████╔╝
//  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝
//
//  ███╗   ███╗███████╗████████╗ █████╗ ██╗
//  ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║
//  ██╔████╔██║█████╗     ██║   ███████║██║
//  ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║
//  ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████╗
//  ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
//

#if defined(CCML_BACKEND_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// wrapping in an ifdef block since metal backend is written in objective-C

CCML_API const char * ccml_oper_metal(ccml_tensor * tensor) {
    switch (tensor->oper) {
        case CCML_OPER_LOG: return "log";
        case CCML_OPER_EXP: return "exp";
        case CCML_OPER_SIN: return "sin";
        case CCML_OPER_REC: return "1/";
        case CCML_OPER_SQT: return "sqrt";
        case CCML_OPER_ADD: return "+";
        case CCML_OPER_MUL: return "*";
        default: CCML_ASSERT(false, "no meaningful conversion to string exists");
    }
}

CCML_API const char * ccml_type_metal(ccml_tensor * tensor) {
    switch (tensor->type) {
        case CCML_TYPE_FP32: return "float ";
        default: CCML_ASSERT(false, "unknown variant of ccml_type");
    }
}

CCML_API const char * ccml_new_kernel_metal(ccml_context * ctx, struct ccml_graph * graph,
                                        int n_kernel, int start, int finish) {
    int size = CCML_CHAR_MAX * CCML_NODE_MAX * sizeof(char);
    const char * kernel = ccml_malloc(ctx, size);
    char * string = (char*)kernel;

    // the n_kernel parameter specifies the id of the kernel being generated,
    // and start to finish are ides of graph->nodes that pertain to that kernel
    if (n_kernel == 0) {
        string += snprintf(string, size - (kernel - string), "#include <metal_stdlib>\nusing namespace metal;\n");
    }

    string += snprintf(string, size - (kernel - string), "kernel void my_kernel_%d(", n_kernel);
    // adding kernel input parameters to the kernel string
    int n_kernel_parameters = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];

        if (ccml_has_buffer(tensor)) {
            if (n_kernel_parameters == 0) {
                string += snprintf(string, size - (kernel - string), "device %s* data_%d [[buffer(0)]]",
                                   ccml_type_metal(tensor), i);
                n_kernel_parameters++;
            } else {
                string += snprintf(string, size - (kernel - string),  ", device %s* data_%d [[buffer(%d)]]",
                                   ccml_type_metal(tensor), i, n_kernel_parameters);
                n_kernel_parameters++;
            }
        }
    }

    // setting up thread grid dims
    int grid = 1;
    for (int i = start; i < finish; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        grid = grid > tensor->shape[1] ? grid : tensor->shape[1];
    }

    string += snprintf(string, size - (kernel - string),
                       ", uint3 gid [[thread_position_in_grid]]) {\n"
                       "\tuint id0 = gid.x / %d;\n\tuint id1 = gid.x %% %d;\n"
                       "\tuint id2 = gid.y;\n\tuint id3 = gid.z;\n\n", grid, grid);

    for (int i = start; i < finish; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        switch (tensor->oper) {
            case CCML_OPER_LOG:
            case CCML_OPER_EXP:
            case CCML_OPER_SIN:
            case CCML_OPER_REC:
            case CCML_OPER_SQT:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = %s(temp_%d);\n",
                                   ccml_type_metal(tensor), tensor->index,
                                   ccml_oper_metal(tensor), tensor->src[0]->index);
                break;
            case CCML_OPER_ADD:
            case CCML_OPER_MUL:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = temp_%d %s temp_%d;\n",
                                   ccml_type_metal(tensor), tensor->index, tensor->src[0]->index,
                                   ccml_oper_metal(tensor), tensor->src[1]->index);
                break;
            case CCML_OPER_SUM:
                string += snprintf(string, size - (kernel - string), "\tfor (int i = 0; i < %d; i++) {\n"
                                   "\t\tdata_%d%s += temp_%d;\n"
                                   "\t}\n", ccml_size(tensor->src[0]) / ccml_size(tensor), tensor->index + 1,
                                   ccml_new_index(ctx, tensor, tensor->src[0]), tensor->src[0]->index);
                break;
            case CCML_OPER_RES:
            case CCML_OPER_PER:
                string += snprintf(string, size - (kernel - string), "\tdevice %s* data_%d = data_%d;\n"
                                   "\t%stemp_%d = data_%d%s;\n", ccml_type_metal(tensor), tensor->index,
                                    tensor->src[0]->index, ccml_type_metal(tensor), tensor->index,
                                    tensor->index, ccml_new_index(ctx, NULL, tensor));
                break;
            case CCML_OPER_LOAD:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = data_%d%s;\n",
                                   ccml_type_metal(tensor), tensor->index, tensor->index, ccml_new_index(ctx, NULL, tensor));
                break;
            case CCML_OPER_INTR:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = data_%d%s;\n",
                                   ccml_type_metal(tensor), tensor->index, tensor->index, ccml_new_index(ctx, NULL, tensor));
                break;
            case CCML_OPER_SAVE:
                string += snprintf(string, size - (kernel - string), "\tdata_%d%s = temp_%d;\n",
                                   tensor->index, ccml_new_index(ctx, NULL, tensor), tensor->src[0]->index);
                break;
            default:
                CCML_ASSERT(false, "unknown variant of ccml_oper");
        }
    }

    snprintf(string, size - (kernel - string), "}");

    return kernel;
}

CCML_API void ccml_execute_graph_metal(ccml_context * ctx, ccml_graph * graph) {
    const char * kernel_source = ccml_new_kernel_metal(ctx, graph, 0, 0, graph->n_nodes);
    printf("the kernel is:\n%s\n", kernel_source);

    @autoreleasepool {
        // Errors
        NSError * error = nil;

        // initialise metal
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSString * kernel_source_ = [NSString stringWithUTF8String:kernel_source];
        id<MTLLibrary> library = [device newLibraryWithSource:kernel_source_ options:nil error:&error];
        if (!library) {
            fprintf(stderr, "Failed to create MTLLibrary: %s\n", [[error localizedDescription] UTF8String]);
            return;
        }

        // create compute function and GPU pipeline
        id<MTLFunction> function = [library newFunctionWithName:@"my_kernel_0"];
        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
        id<MTLCommandQueue> command_queue = [device newCommandQueue];

        // data for buffers
        id<MTLBuffer> buffers[CCML_NODE_MAX] = {NULL};
        for (int i = 0; i < graph->n_nodes; i++) {
            ccml_tensor * tensor = graph->nodes[i];
            if (tensor != NULL && ccml_has_buffer(tensor)) {
                buffers[i] = [device newBufferWithBytes:tensor->data length:ccml_size(tensor) options: MTLResourceStorageModeShared];
            }
        }

        // command buffer and compute command encoder
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];

        [compute_encoder setComputePipelineState:pipeline_state];

        int buffer_counter = 0;
        int grid[CCML_DIMS_MAX] = {1, 1, 1, 1};
        for (int i = 0; i < graph->n_nodes; i++) {
            ccml_tensor * tensor = graph->nodes[i];
            if (tensor != NULL && ccml_has_buffer(tensor)) {
                [compute_encoder setBuffer:buffers[i] offset:0 atIndex:buffer_counter++];
            }

            for (int j = 0; j < CCML_DIMS_MAX; j++) {
                if (tensor != NULL && grid[j] < tensor->shape[j]) {
                    grid[j] = tensor->shape[j];
                }
            }
        }

        // dispatch threads
        MTLSize grid_size = MTLSizeMake(grid[0] * grid[1], grid[2], grid[3]);
        MTLSize thread_group_size = MTLSizeMake(1, 1, 1);
        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];

        // end encoding and commit command buffer
        [compute_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        // copy the result back to the C array to check it
        float * result = NULL;
        int result_size = 1;
        for (int i = 0; i < graph->n_nodes; i++) {
            ccml_tensor * tensor = graph->nodes[i];
            if (tensor != NULL && ccml_has_buffer(tensor)) {
                result = [buffers[i] contents];
                result_size = ccml_size(tensor);
            }
        }

        for (int i = 0; i < result_size; i++) {
            printf("%f ", result[i]);
        }
        printf("\n");
    }
}

#else

CCML_API const char * ccml_oper_metal(ccml_tensor *);
CCML_API const char * ccml_type_metal(ccml_tensor *);
CCML_API const char * ccml_new_kernel_metal(ccml_context *, ccml_graph *, int, int, int);
CCML_API void ccml_execute_graph_metal(ccml_context *, ccml_graph *);

#endif /* defined(__APPLE__) */

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

#if defined(CCML_BACKEND_OPENCL)

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

CCML_API const char * ccml_oper_opencl(ccml_tensor * tensor) {
    switch (tensor->oper) {
        case CCML_OPER_LOG: return "log";
        case CCML_OPER_EXP: return "exp";
        case CCML_OPER_SIN: return "sin";
        case CCML_OPER_REC: return "1/";
        case CCML_OPER_SQT: return "sqrt";
        case CCML_OPER_ADD: return "+";
        case CCML_OPER_MUL: return "*";
        default: CCML_ASSERT(false, "no meaningful conversion to string exists");
    }
}

CCML_API const char * ccml_type_opencl(ccml_tensor * tensor) {
    switch (tensor->type) {
        case CCML_TYPE_FP32: return "float ";
        default: CCML_ASSERT(false, "unknown variant of ccml_type");
    }
}

CCML_API const char * ccml_new_kernel_opencl(ccml_context * ctx, struct ccml_graph * graph,
                                         int n_kernel, int start, int finish) {
    int size = CCML_CHAR_MAX * CCML_NODE_MAX * sizeof(char);
    const char * kernel = ccml_malloc(ctx, size);
    char * string = (char*)kernel;

    string += snprintf(string, size - (kernel - string), "__kernel void my_kernel_%d(", n_kernel);
    // adding kernel input parameters to the kernel string
    int n_kernel_parameters = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];

        if (ccml_has_buffer(tensor)) {
            if (n_kernel_parameters == 0) {
                string += snprintf(string, size - (kernel - string), "__global %s* data_%d",
                                   ccml_type_opencl(tensor), i);
                n_kernel_parameters++;
            } else {
                string += snprintf(string, size - (kernel - string),  ", __global %s* data_%d",
                                   ccml_type_opencl(tensor), i);
                n_kernel_parameters++;
            }
        }
    }

    string += snprintf(string, size - (kernel - string), ") {\n");

    // setting up thread grid dims
    int grid = 1;
    for (int i = start; i < finish; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        grid = grid > tensor->shape[1] ? grid : tensor->shape[1];
    }

    string += snprintf(string, size - (kernel - string),
                       "\tint id0 = get_global_id(0) / %d;\n\tint id1 = get_global_id(0) %% %d;\n"
                       "\tint id2 = get_global_id(1);\n\tint id3 = get_global_id(2);\n\n", grid, grid);

    for (int i = start; i < finish; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        switch (tensor->oper) {
            case CCML_OPER_LOG:
            case CCML_OPER_EXP:
            case CCML_OPER_SIN:
            case CCML_OPER_REC:
            case CCML_OPER_SQT:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = %s(temp_%d);\n",
                                   ccml_type_opencl(tensor), tensor->index,
                                   ccml_oper_opencl(tensor), tensor->src[0]->index);
                break;
            case CCML_OPER_ADD:
            case CCML_OPER_MUL:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = temp_%d %s temp_%d;\n",
                                   ccml_type_opencl(tensor), tensor->index, tensor->src[0]->index,
                                   ccml_oper_opencl(tensor), tensor->src[1]->index);
                break;
            case CCML_OPER_SUM:
                string += snprintf(string, size - (kernel - string), "\tfor (int i = 0; i < %d; i++) {\n"
                                   "\t\tdata_%d%s += temp_%d;\n"
                                   "\t}\n", ccml_size(tensor->src[0]) / ccml_size(tensor), tensor->index + 1,
                                   ccml_new_index(ctx, tensor, tensor->src[0]), tensor->src[0]->index);
                break;
            case CCML_OPER_RES:
            case CCML_OPER_PER:
                string += snprintf(string, size - (kernel - string), "\tdevice %s* data_%d = data_%d;\n"
                                   "\t%stemp_%d = data_%d%s;\n", ccml_type_opencl(tensor), tensor->index,
                                    tensor->src[0]->index, ccml_type_opencl(tensor), tensor->index,
                                    tensor->index, ccml_new_index(ctx, NULL, tensor));
                break;
            case CCML_OPER_LOAD:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = data_%d%s;\n",
                                   ccml_type_opencl(tensor), tensor->index, tensor->index, ccml_new_index(ctx, NULL, tensor));
                break;
            case CCML_OPER_INTR:
                string += snprintf(string, size - (kernel - string), "\t%stemp_%d = data_%d%s;\n",
                                   ccml_type_opencl(tensor), tensor->index, tensor->index, ccml_new_index(ctx, NULL, tensor));
                break;
            case CCML_OPER_SAVE:
                string += snprintf(string, size - (kernel - string), "\tdata_%d%s = temp_%d;\n",
                                   tensor->index, ccml_new_index(ctx, NULL, tensor), tensor->src[0]->index);
                break;
            default:
                CCML_ASSERT(false, "unknown variant of ccml_oper");
        }
    }

    snprintf(string, size - (kernel - string), "}");

    return kernel;
}

CCML_API void ccml_check_error_opencl(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(1);
    }
}

CCML_API void ccml_execute_graph_opencl(ccml_context * ctx, ccml_graph * graph) {
    const char * kernel_source = ccml_new_kernel_opencl(ctx, graph, 0, 0, graph->n_nodes);
    printf("kernel is: \n %s \n", kernel_source);

    // get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ccml_check_error_opencl(ret, "clGetPlatformIDs");

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    ccml_check_error_opencl(ret, "clGetDeviceIDs");

    // create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    ccml_check_error_opencl(ret, "clCreateContext");

    // create command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    ccml_check_error_opencl(ret, "clCreateCommandQueue");

    // create memory buffers on the device for each vector and copy data inside buffers
    cl_mem buffers[CCML_NODE_MAX] = {NULL};
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_has_buffer(tensor) && tensor->oper != CCML_OPER_SAVE) {
            buffers[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, ccml_size(tensor) * sizeof(float), NULL, &ret);
            ccml_check_error_opencl(ret, "clCreateBuffer");

            ret = clEnqueueWriteBuffer(command_queue, buffers[i], CL_TRUE, 0, ccml_size(tensor) * sizeof(float), tensor->data, 0, NULL, NULL);
            ccml_check_error_opencl(ret, "clEnqueueWriteBuffer");
        } else if (ccml_has_buffer(tensor) && tensor->oper == CCML_OPER_SAVE) {
            buffers[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ccml_size(tensor) * sizeof(float), NULL, &ret);
            ccml_check_error_opencl(ret, "clCreateBuffer");
        }
    }

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &ret);
    ccml_check_error_opencl(ret, "clCreateProgramWithSource");

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n", buffer);
        exit(1);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "my_kernel_0", &ret);
    ccml_check_error_opencl(ret, "clCreateKernel");

    // Set the arguments of the kernel
    int buffer_index = 0;
    int grid[CCML_DIMS_MAX] = {1, 1, 1, 1};
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_has_buffer(tensor)) {
            ret = clSetKernelArg(kernel, buffer_index++, sizeof(cl_mem), (void *)&buffers[i]);
            ccml_check_error_opencl(ret, "clSetKernelArg");
        }

        for (int j = 0; j < CCML_DIMS_MAX; j++) {
            if (grid[j] < tensor->shape[j]) grid[j] = tensor->shape[j];
        }
    }

    // Execute the OpenCL kernel on the list
    size_t global_item_size[3] = {grid[0] * grid[1], grid[2], grid[3]}; // Adjust according to your problem
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_item_size, NULL, 0, NULL, NULL);
    ccml_check_error_opencl(ret, "clEnqueueNDRangeKernel");

    // Read the memory buffer c on the device to the local variable c
    float * result = NULL;
    int result_size = 1;
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_has_buffer(tensor) && tensor->oper == CCML_OPER_SAVE) {
            ret = clEnqueueReadBuffer(command_queue, buffers[i], CL_TRUE, 0, ccml_size(tensor) * sizeof(float), tensor->data, 0, NULL, NULL);
            ccml_check_error_opencl(ret, "clEnqueueReadBuffer");

            result = tensor->data;
            result_size = ccml_size(tensor);
        }
    }

    for (int i = 0; i < result_size; i++) {
        printf("%f ", result[i]);
    }

    // Clean up
    for (int i = 0; i < graph->n_nodes; i++) {
        if (buffers[i] != NULL) {
            clReleaseMemObject(buffers[i]);
        }
    }


    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}
#else

CCML_API const char * ccml_oper_opencl(ccml_tensor *);
CCML_API const char * ccml_type_opencl(ccml_tensor *);
CCML_API const char * ccml_new_kernel_opencl(ccml_context *, ccml_graph *, int, int, int);
CCML_API void ccml_execute_graph_opencl(ccml_context *, ccml_graph *);

#endif /* defined CCML_BACKEND_OPENCL */

//
//  ███████╗██╗  ██╗███████╗ ██████╗██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
//  ██╔════╝╚██╗██╔╝██╔════╝██╔════╝██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
//  █████╗   ╚███╔╝ █████╗  ██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
//  ██╔══╝   ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
//  ███████╗██╔╝ ██╗███████╗╚██████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
//  ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
//

CCML_API void ccml_graph_execute(ccml_context * ctx, ccml_graph * graph) {
    #if defined(CCML_BACKEND_METAL)
        ccml_execute_graph_metal(ctx, graph);
    #elif defined(CCML_BACKEND_OPENCL)
        ccml_execute_graph_opencl(ctx, graph);
    #else
        #error unknown backend
    #endif
}

#endif /* CCML_IMPL */