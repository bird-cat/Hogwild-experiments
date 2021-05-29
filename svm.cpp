#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include <time.h>
#include <omp.h>
#include "svm.h"

#ifndef min
template <class T>
static inline T min(T x, T y) { return (x < y) ? x : y; }
#endif

#ifndef max
template <class T>
static inline T max(T x, T y)
{
    return (x > y) ? x : y;
}
#endif

template <class T>
static inline void swap(T &x, T &y)
{
    T t = x;
    x = y;
    y = t;
}

static inline double powi(double base, int times)
{
    double tmp = base, ret = 1.0;

    for (int t = times; t > 0; t /= 2)
    {
        if (t % 2 == 1)
            ret *= tmp;
        tmp = tmp * tmp;
    }
    return ret;
}

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

static void print_string_stdout(const char *s)
{
    fputs(s, stdout);
    fflush(stdout);
}
static void (*svm_print_string)(const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt, ...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buf, fmt, ap);
    va_end(ap);
    (*svm_print_string)(buf);
}
#else
static void info(const char *fmt, ...){}
#endif

double inner_product(double *w, const struct svm_node *x)
{
    double v = 0;
    int j;
    for (int i = 0; x[i].index != -1; i++)
    {
        j = x[i].index - 1;
        v += w[j] * x[i].value;
    }
    return v;
}

void weight_cpy(double *w_dst, const double *w_src, const svm_node *x)
{
    int j;
    for (int i = 0; x[i].index != -1; i++)
    {
        j = x[i].index - 1;
        w_dst[j] = w_src[j];
    }
}

void svc_gradient(double *grad, const double lambda, const svm_problem *prob, const int feature_index)
{
    int j;
    svm_node *x = prob->x[feature_index];
    double y = prob->y[feature_index];
    double y_pred = inner_product(grad, x);
    if (y * y_pred < 1) {
        for (int i = 0; x[i].index != -1; i++)
        {
            j = x[i].index - 1;
            grad[j] = lambda * grad[j] / prob->d[j] - y * x[i].value;
        }
    }
    else
    {
        for (int i = 0; x[i].index != -1; i++)
        {
            j = x[i].index - 1;
            grad[j] = lambda * grad[j] / prob->d[j];
        }
    }
}

void svr_gradient(double *grad, const double lambda, const double epsilon, const svm_problem *prob, const int feature_index)
{
    int j;
    svm_node *x = prob->x[feature_index];
    double y = prob->y[feature_index];
    double y_pred = inner_product(grad, x);
    if (y_pred - y > epsilon)
    {
        for (int i = 0; x[i].index != -1; i++)
        {
            j = x[i].index - 1;
            grad[j] = lambda * grad[j] / prob->d[j] + x[i].value;
        }
    }
    else if (y - y_pred > epsilon)
    {
        for (int i = 0; x[i].index != -1; i++)
        {
            j = x[i].index - 1;
            grad[j] = lambda * grad[j] / prob->d[j] - x[i].value;
        }
    }
    else
    {
        for (int i = 0; x[i].index != -1; i++)
        {
            j = x[i].index - 1;
            grad[j] = lambda * grad[j] / prob->d[j];
        }
    }
}

void update_model(svm_model *model, double *grad, const svm_node *x)
{
    int j;
    for (int i = 0; x[i].index != -1; i++)
    {
        j = x[i].index - 1;
        //#pragma omp atomic
        model->w[j] -= grad[j];
    }
}

void binary_svc_Hogwild_solver(const svm_problem *prob, svm_model *model)
{
    svm_parameter *param = &model->param;
    double *grad = Malloc(double, prob->dim);
    //srand((unsigned int)time(NULL));
    omp_set_num_threads(param->n_cores);
#pragma omp parallel for firstprivate(grad) schedule(static)
    for (int t = 0; t < param->T; t++)
    {
        //printf("t = %d, thread = %d\n", t, omp_get_thread_num());
        //int i = rand() % prob->l;
        int i = t % prob->l;
        weight_cpy(grad, model->w, prob->x[i]);
        svc_gradient(grad, param->lambda, prob, i);
        update_model(model, grad, prob->x[i]);
    }
}

void epsilon_svr_Hogwild_solver(const svm_problem *prob, svm_model *model)
{
    svm_parameter *param = &model->param;
    double *grad = Malloc(double, prob->dim);
    //srand((unsigned int)time(NULL));
    omp_set_num_threads(param->n_cores);
#pragma omp parallel for firstprivate(grad)
    for (int t = 0; t < param->T; t++)
    {
        //printf("t = %d, thread = %d\n", t, omp_get_thread_num());
        int i = t % prob->l;
        weight_cpy(grad, model->w, prob->x[i]);
        svr_gradient(grad, param->lambda, param->p, prob, i);
        update_model(model, grad, prob->x[i]);
    }
}

void svc_HogBatch_update(double *grad, double *w, svm_model *model, const svm_problem *prob, const int start_index)
{
    int k;
    int batch_size = model->param.batch_size;
    double lambda = model->param.lambda;
    int m = min(start_index + batch_size, prob->l);
    svm_node *x;
    double y, y_pred;
    memset(grad, 0, prob->dim * sizeof(double));
    memcpy(w, model->w, prob->dim * sizeof(double));

    for (int i = start_index; i < m; i++)
    {
        x = prob->x[i];
        y = prob->y[i];
        y_pred = inner_product(w, x);
        if (y * y_pred < 1)
        {
            for (int j = 0; x[j].index != -1; j++)
            {
                k = x[j].index - 1;
                grad[k] += lambda * w[k] / prob->d[k] - y * x[j].value;
                w[k] -= grad[k];
            }
        }
        else
        {
            for (int j = 0; x[j].index != -1; j++)
            {
                k = x[j].index - 1;
                grad[k] += lambda * w[k] / prob->d[k];
                w[k] -= grad[k];
            }
        }
    }

    // Update model
    for (int j = 0; j < prob->dim; j++)
        model->w[j] -= grad[j];
}

void svr_HogBatch_update(double *grad, double *w, svm_model *model, const svm_problem *prob, const int start_index)
{
    int k;
    int batch_size = model->param.batch_size;
    double lambda = model->param.lambda;
    double epsilon = model->param.p;
    int m = min(start_index + batch_size, prob->l);
    svm_node *x;
    double y, y_pred;
    memset(grad, 0, prob->dim * sizeof(double));
    memcpy(w, model->w, prob->dim * sizeof(double));

    // Calculate gradients
    for (int i = start_index; i < m; i++)
    {
        x = prob->x[i];
        y = prob->y[i];
        y_pred = inner_product(w, x);
        if (y_pred - y > epsilon)
        {
            for (int j = 0; x[j].index != -1; j++)
            {
                k = x[j].index - 1;
                grad[k] += lambda * w[k] / prob->d[k] + x[j].value;
                w[k] -= grad[k];
            }
        }
        else if (y - y_pred > epsilon)
        {
            for (int j = 0; x[j].index != -1; j++)
            {
                k = x[j].index - 1;
                grad[k] += lambda * w[k] / prob->d[k] - x[j].value;
                w[k] -= grad[k];
            }
        }
        else
        {
            for (int j = 0; x[j].index != -1; j++)
            {
                k = x[j].index - 1;
                grad[k] += lambda * w[k] / prob->d[k];
                w[k] -= grad[k];
            }
        }
    }

    // Update model
    for (int j = 0; j < prob->dim; j++)
        model->w[j] -= grad[j];
}

void binary_svc_HogBatch_solver(const svm_problem *prob, svm_model *model)
{
    svm_parameter *param = &model->param;
    double *grad = Malloc(double, prob->dim), *w = Malloc(double, prob->dim);
    omp_set_num_threads(param->n_cores);
#pragma omp parallel for firstprivate(grad, w)
    for (int t = 0; t < param->T; t += param->batch_size)
    {
        //printf("t = %d, thread = %d\n", t, omp_get_thread_num());
        int i = t % prob->l;
        svc_HogBatch_update(grad, w, model, prob, i);
    }
}

void epsilon_svr_HogBatch_solver(const svm_problem *prob, svm_model *model)
{
    svm_parameter *param = &model->param;
    double *grad = Malloc(double, prob->dim), *w = Malloc(double, prob->dim);
    omp_set_num_threads(param->n_cores);
#pragma omp parallel for firstprivate(grad, w)
    for (int t = 0; t < param->T; t += param->batch_size)
    {
        //printf("t = %d, thread = %d\n", t, omp_get_thread_num());
        int i = t % prob->l;
        svr_HogBatch_update(grad, w, model, prob, i);
    }
}

struct svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
    svm_model *model = Malloc(struct svm_model, 1);
    model->param = *param;
    model->dim = prob->dim;
    model->w = Malloc(double, model->dim);
    for (int i = 0; i < prob->dim; i++)
        model->w[i] = 0.0;

    if (param->batch_size == 1)
    {
        if (param->svm_type == BINARY_SVC)
            binary_svc_Hogwild_solver(prob, model);
        else if (param->svm_type == EPSILON_SVR)
            epsilon_svr_Hogwild_solver(prob, model);
    }
    else
    {
        if (param->svm_type == BINARY_SVC)
            binary_svc_HogBatch_solver(prob, model);
        else if (param->svm_type == EPSILON_SVR)
            epsilon_svr_HogBatch_solver(prob, model);
    }
    printf("Finish training\n");
    return model;
}

static const char *svm_type_table[] =
    {
        "b_svc", "epsilon_svr", NULL};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
    FILE *fp = fopen(model_file_name, "w");
    if (fp == NULL)
        return -1;

    char *old_locale = setlocale(LC_ALL, NULL);
    if (old_locale)
    {
        old_locale = strdup(old_locale);
    }
    setlocale(LC_ALL, "C");

    const svm_parameter &param = model->param;

    fprintf(fp, "svm_type %s\n", svm_type_table[param.svm_type]);

    if (model->dim)
        fprintf(fp, "dim %d\n", model->dim);

    if (model->w)
    {
        fprintf(fp, "w");
        for (int i = 0; i < model->dim; i++)
            fprintf(fp, " %.17g", model->w[i]);
        fprintf(fp, "\n");
    }

    setlocale(LC_ALL, old_locale);
    free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0)
        return -1;
    else
        return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)           \
    do                                           \
    {                                            \
        if (fscanf(_stream, _format, _var) != 1) \
            return false;                        \
    } while (0)

bool read_model_header(FILE *fp, svm_model *model)
{
    svm_parameter &param = model->param;
    // parameters for training only won't be assigned, but arrays are assigned as NULL for safety

    char cmd[81];
    while (1)
    {
        FSCANF(fp, "%80s", cmd);

        if (strcmp(cmd, "svm_type") == 0)
        {
            FSCANF(fp, "%80s", cmd);
            int i;
            for (i = 0; svm_type_table[i]; i++)
            {
                if (strcmp(svm_type_table[i], cmd) == 0)
                {
                    param.svm_type = i;
                    break;
                }
            }
            if (svm_type_table[i] == NULL)
            {
                fprintf(stderr, "unknown svm type.\n");
                printf("1\n");
                return false;
            }
        }
        else if (strcmp(cmd, "dim") == 0)
        {
            FSCANF(fp, "%d", &model->dim);
        }
        else if (strcmp(cmd, "w") == 0)
        {
            int dim = model->dim;
            model->w = Malloc(double, dim);
            for (int i = 0; i < dim; i++)
                FSCANF(fp, "%lf", &model->w[i]);
            // Get out off loop
            break;
        }
        else
        {
            fprintf(stderr, "unknown text in model file: [%s]\n", cmd);
            return false;
        }
    }

    return true;
}

svm_model *svm_load_model(const char *model_file_name)
{
    FILE *fp = fopen(model_file_name, "rb");
    if (fp == NULL)
        return NULL;

    char *old_locale = setlocale(LC_ALL, NULL);
    if (old_locale)
    {
        old_locale = strdup(old_locale);
    }
    setlocale(LC_ALL, "C");

    // read parameters

    svm_model *model = Malloc(svm_model, 1);
    //model->rho = NULL;
    model->w = NULL;

    // read header
    if (!read_model_header(fp, model))
    {
        fprintf(stderr, "ERROR: fscanf failed to read model\n");
        setlocale(LC_ALL, old_locale);
        free(old_locale);
        //free(model->rho);
        free(model->w);
        free(model);
        return NULL;
    }

    setlocale(LC_ALL, old_locale);
    free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0)
        return NULL;

    return model;
}

double svm_predict(const svm_model *model, const svm_node *x)
{
    double pred_result;
    if (model->param.svm_type == EPSILON_SVR)
    {
        // TODO
        pred_result = inner_product(model->w, x);
    }
    else
    {
        pred_result = (inner_product(model->w, x) > 0)? 1.0: -1.0;
        //printf("%lf\n", inner_product(model->w, x));
    }
    return pred_result;
}

double svm_predict_probability(
    const svm_model *model, const svm_node *x, double *prob_estimates)
{
    // TODO
    return 0.0;
}

int svm_get_svm_type(const svm_model *model)
{
    return model->param.svm_type;
}

void svm_free_model_content(svm_model *model_ptr)
{
    //free(model_ptr->rho);
    //model_ptr->rho = NULL;

    free(model_ptr->w);
    model_ptr->w = NULL;
}

void svm_free_and_destroy_model(svm_model **model_ptr_ptr)
{
    if (model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
    {
        svm_free_model_content(*model_ptr_ptr);
        free(*model_ptr_ptr);
        *model_ptr_ptr = NULL;
    }
}

void svm_destroy_param(svm_parameter *param)
{
    return;
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
    int svm_type = param->svm_type;
    if (svm_type != BINARY_SVC &&
        svm_type != EPSILON_SVR)
        return "unknown svm type";

    if (param->eps <= 0)
        return "eps <= 0";

    if (param->lambda < 0)
        return "lambda < 0";

    if (svm_type == EPSILON_SVR)
        if (param->p < 0)
            return "p < 0";
    
    if (param->T <= 0)
        return "T <= 0";
    
    if (param->batch_size <= 0)
        return "batch size <= 0";
    
    if (param->n_cores <= 0)
        return "number of cores <= 0";

    return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
    return 0;
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
    if (print_func == NULL)
        svm_print_string = &print_string_stdout;
    else
        svm_print_string = print_func;
}
