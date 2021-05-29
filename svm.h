#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 325

#ifdef __cplusplus
extern "C"
{
#endif

    extern int libsvm_version;

    struct svm_node
    {
        int index;
        double value;
    };

    struct svm_problem
    {
        int l, dim;
        double *y;
        struct svm_node **x;
        int *d;
    };

    enum
    {
        BINARY_SVC,
        EPSILON_SVR
    }; /* svm_type */
    enum
    {
        LINEAR,
        POLY,
        RBF,
        SIGMOID,
        PRECOMPUTED
    }; /* kernel_type */

    struct svm_parameter
    {
        int svm_type;

        /* these are for training only */
        double cache_size; /* in MB */
        double eps;        /* stopping criteria */
        double p;          /* for EPSILON_SVR */

        /* New added */
        double lambda;     /* regularization parameter */
        int T;             /* number of SGD iteration */
        int n_cores;       /* number of cores used to train the model */
        int batch_size;    /* batch size of HogBatch SGD */
    };

    //
    // svm_model
    //
    struct svm_model
    {
        struct svm_parameter param; /* parameter */
        double *rho;                /* constants in decision functions (rho[k*(k-1)/2]) */

        /* new added */
        double *w;
        int dim;
    };

    struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
    void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

    int svm_save_model(const char *model_file_name, const struct svm_model *model);
    struct svm_model *svm_load_model(const char *model_file_name);

    int svm_get_svm_type(const struct svm_model *model);
    double svm_predict(const struct svm_model *model, const struct svm_node *x);

    void svm_free_model_content(struct svm_model *model_ptr);
    void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
    void svm_destroy_param(struct svm_parameter *param);

    const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);

    void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */