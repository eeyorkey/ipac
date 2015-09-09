#include <omp.h>
#include <fstream>
#include <time.h>
#include <map>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "ipac.h"

namespace xgboost {
namespace ipac {

ipac::ipac()
{
    _data_mat_test = NULL;
    _data_mat_train = NULL;
    _d_M = 0;
    _d_D = 0;
    _d_J = 1;
    _data_labels.clear();
    _off_set = 0;
    _stage_one_pi0 = 0;
    _stage_one_pi1 = 0;
    _stage_one_loglik.clear();
    _stage_two_pi0.clear();
    _stage_two_pi1.clear();
    _stage_two_loglik.clear();
    _stage_two_loglikvalid.clear();
    _loglik_eval.clear();
    _iter_num.clear();
    _alpha.clear();
    _Q.clear();
    _model_valid = false;
}

ipac::~ipac()
{
    _stage_one_loglik.clear();
    _stage_two_pi0.clear();
    _stage_two_pi1.clear();
    _stage_two_loglik.clear();
    _stage_two_loglikvalid.clear();
    _loglik_eval.clear();
    _iter_num.clear();
    _solution_path.clear();
    _data_labels.clear();
    _alpha.clear();
    _Q.clear();
}

int ipac::LoadData(const char* f_name, std::string tag) {
    utils::Assert(tag == "train" || tag == "test", "Tag can only be "
        "train or test");
    utils::IIterator<RowBatch>* iter;

    if (tag == "train") {
        _data_mat_train = io::LoadDataMatrix(f_name);
        // initialize the dimension of the weight matrix
        _d_M = _data_mat_train->info.info.num_row;         // num of instances
        _d_D = _data_mat_train->info.info.num_col - _d_J;  // num of features

        iter = _data_mat_train->fmat()->RowIterator();
        
    } else {
        _data_mat_test = io::LoadDataMatrix(f_name);

        _d_M = _data_mat_test->info.info.num_row;
        _d_D = _data_mat_test->info.info.num_col - _d_J;

        iter = _data_mat_test->fmat()->RowIterator();
    }

    iter->BeforeFirst();
    iter->Next();
    const RowBatch& batch = iter ->Value();

    for (int i = 0; i < _d_M; i ++) {
        std::vector<float> data_label_i;
        data_label_i.clear();
        for (int j = 0; j < _d_J; j ++) {
            float p = batch[i][j].fvalue;
            // check data, p_value ranges: p_value >= 0 && p_value <= 1
            utils::Assert(p >= 0 && p <= 1, "p value should be (0, 1],"
                " but it current value is: p=%f", p);
            if (fabs(p) < zero)
                data_label_i.push_back(zero);
            else
                data_label_i.push_back(p);
        }
        _data_labels.push_back(data_label_i);
    }

    return 0;
}

int ipac::EMStageOne()
{
    utils::Assert(_data_mat_train != NULL, "No training data error");

    _stage_one_pi0 = 1 - _options.init_pi;
    _stage_one_pi1 = _options.init_pi;
    real_t beta_alpha_init = _options.init_betamean / (1 - _options.init_betamean);

    // M * 1, when k = 1
    std::vector<real_t> R_1(_d_M, 0.0);
    // M * J, when k = 0
    std::vector<std::vector<real_t> > RZ_0;
    // M * J, when k = 1
    std::vector<std::vector<real_t> > RZ_1;

    for (int i = 0; i < _d_M; i ++) {
        std::vector<real_t> zero_vec(_d_J, 0);
        RZ_0.push_back(zero_vec);
        RZ_1.push_back(zero_vec);
    }

    // init alpha
    _alpha.clear();
    _Q.clear();
    for (int i = 0; i < _d_J; i ++) {
        _alpha.push_back(beta_alpha_init);

        std::vector<real_t> zero_v;
        zero_v.push_back(1 - _options.init_q);
        zero_v.push_back(_options.init_q);
        _Q.push_back(zero_v);
    }

    // init
    _stage_one_loglik.resize(_options.max_iters_init + 1);
    _stage_one_loglik[0] = -inf;

    const int MIN_ITERATOR_NUM = 4;
    int ncore = omp_get_num_procs();
    int max_tn = _d_M / MIN_ITERATOR_NUM;
    int coreNum = max_tn > 2*ncore ? 2*ncore : max_tn;

    // begin to calculate
    int final_iter = 0;
    for (size_t i = 0; i < _options.max_iters_init; i ++) {
        real_t sum_r1 = 0.0;
        real_t sum_weight_total = 0.0;
#pragma omp parallel for reduction(+:sum_r1,sum_weight_total) if( coreNum > 1) num_threads(coreNum)
        for (int j = 0; j < _d_M; j ++) {
            real_t prod_r0 = 0.0;
            real_t prod_r1 = 0.0;
            // prod(x) = exp(sum(log(x))
            for (int k = 0; k < _d_J; k ++) {
                real_t beta_var = beta(_alpha[k], _data_labels[j][k]);
                prod_r0 += log(_Q[k][0] * beta_var + 
                    1 - _Q[k][0]);
                prod_r1 += log(_Q[k][1] * beta_var + 
                    1 - _Q[k][1]);

                RZ_0[j][k] = _Q[k][0] * beta_var / (_Q[k][0] * beta_var +
                    1 - _Q[k][0]);
                RZ_1[j][k] = _Q[k][1] * beta_var / (_Q[k][1] * beta_var +
                    1 - _Q[k][1]);
            }
            prod_r0 = _stage_one_pi0 * exp(prod_r0);
            prod_r1 = _stage_one_pi1 * exp(prod_r1);

            R_1[j] = prod_r1 / (prod_r0 + prod_r1);
            // instance weight
            sum_r1 += _data_mat_train->info.labels[j] * R_1[j];
            sum_weight_total += _data_mat_train->info.labels[j];

            for (int k = 0; k < _d_J; k ++) {
                RZ_0[j][k] *= (1 - R_1[j]);
                RZ_1[j][k] *= R_1[j];
            }
        }

        // The maximization step
        // update alpha and Q
        // alpha_j = -sum_i{sum_k RZ_k[i][j]} / sum_i{sum_k RZ_k[i][j] * log(p_ij)}
        #pragma omp parallel for if( coreNum > 1) num_threads(coreNum)
        for (int j = 0; j < _d_J; j ++) {
            real_t sum_z1 = 0.0;
            real_t sum_z1p = 0.0;
            real_t sum_rz0 = 0.0;
            real_t sum_rz1 = 0.0;
            for (int k = 0; k < _d_M; k++) {
                real_t sum_rz = (RZ_0[k][j] + RZ_1[k][j]) * _data_mat_train->info.labels[k];
                sum_z1 += sum_rz;
                sum_z1p += sum_rz * log(_data_labels[k][j]);
                
                sum_rz0 += RZ_0[k][j] * _data_mat_train->info.labels[k];
                sum_rz1 += RZ_1[k][j] * _data_mat_train->info.labels[k];
            }
            real_t tmp_alpha = - sum_z1 / sum_z1p;
            tmp_alpha = (tmp_alpha<=_options.lb_beta_alpha)?_options.lb_beta_alpha: tmp_alpha;
            tmp_alpha = (tmp_alpha>=(1-_options.lb_beta_alpha))?(1-_options.lb_beta_alpha):tmp_alpha;
            _alpha[j] = tmp_alpha;

            real_t q_zero = sum_rz0 / (sum_weight_total - sum_r1);
            q_zero = (q_zero <= _options.q_lb)?_options.q_lb:q_zero;
            q_zero = (q_zero >= (1 - _options.q_lb))?(1 - _options.q_lb):q_zero;
            _Q[j][0] = q_zero;

            real_t q_one = sum_rz1 / sum_r1;
            q_one = (q_one <= _options.q_lb)?_options.q_lb:q_one;
            q_one = (q_one >= (1 - _options.q_lb))?(1 - _options.q_lb):q_one;
            _Q[j][1] = q_one;
        }

        // update p0_init and beta_var, loglikelihood
        real_t sum_loglik = 0.0;
        _stage_one_pi1 = sum_r1 / sum_weight_total;
        _stage_one_pi1 = (_stage_one_pi1<=_options.lb_pi1)?_options.lb_pi1:_stage_one_pi1;
        _stage_one_pi1 = (_stage_one_pi1 >= (1-_options.lb_pi1)) ? (1-_options.lb_pi1): _stage_one_pi1;
        _stage_one_pi0 = 1 - _stage_one_pi1;
        
#pragma omp parallel for reduction(+:sum_loglik) if( coreNum > 1) num_threads(coreNum)
        for (int j = 0; j < _d_M; j ++) {
            real_t p_x = 0.0;
            real_t prod_k_one = 0.0;
            real_t prod_k_zero = 0.0;
            for (int k = 0; k < _d_J; k ++) {
                real_t beta_var = beta(_alpha[k], _data_labels[j][k]);
                prod_k_one += log(_Q[k][1] * beta_var + 1 - _Q[k][1]);
                prod_k_zero += log(_Q[k][0] * beta_var + 1 - _Q[k][0]);
            }
            p_x = _stage_one_pi1 * exp(prod_k_one) + _stage_one_pi0 * exp(prod_k_zero);
            sum_loglik += log(p_x) * _data_mat_train->info.labels[j];
        }
        _stage_one_loglik[i+1] = sum_loglik;
        if (_options.verbose && i % 10 == 0) {
            fprintf(stdout, "Stage one iteration %d with loglikelihood: %f\n",
                (i+1), sum_loglik);
        }
        
        // Can we stop
        if (_stage_one_loglik[i+1] - _stage_one_loglik[i] < _options.eps_stop_loglik * 
                _stage_one_loglik[i]) {
            fprintf(stdout, "Stage one iteration %d with loglikelihood: %f\n", (i+1), sum_loglik);
            break;
        }
        final_iter ++;
    }
    
    fprintf(stdout, "Stage one FINAL iteration %d with loglikelihood: %f\n", final_iter, 
                _stage_one_loglik[final_iter]);

    printf("_Q:\n");
    for (int i = 0; i < _d_J; i ++) {
        printf("%f,%f\n", _Q[i][0], _Q[i][1]);
    }
    printf("_alpha:\n");
    for(int i = 0; i < _d_J; i++) {
        printf("%f ", _alpha[i]);
    }
    printf("\n");

    return 0;
}

void ipac::init_stage_two() {
    _stage_two_loglik.clear();
    _stage_two_loglik.resize(_options.cv_fold);
    _stage_two_loglikvalid.clear();
    _stage_two_loglikvalid.resize(_options.cv_fold);
    _loglik_eval.clear();
    _loglik_eval.resize(_options.max_iters + 1);
    _iter_num.clear();
    _iter_num.resize(_options.cv_fold);

    for (int i = 0; i < _options.cv_fold; i ++) {
        _stage_two_loglik[i].resize(_options.max_iters + 1);
        _stage_two_loglikvalid[i].resize(_options.max_iters + 1);
        _stage_two_loglik[i][0] = -inf;
        _stage_two_loglikvalid[i][0] = -inf;
    }
}

void ipac::reset_stage_two_model(std::vector<real_t>& model_alpha) {
    if (_options.more_mode && _model_valid) {
        return;
    }

    // init
    _model_weight.clear();
    _model_weight.resize(_d_D, 0.0);
    for (size_t i = 0; i < _d_D; i ++) {
        _model_weight[i] = 0;
    }
    _model_bias = log(_stage_one_pi1 / _stage_one_pi0);
    _alpha.assign(model_alpha.begin(), model_alpha.end());
}

void ipac::init_pi1(std::vector<real_t>& vec) {
    utils::Assert(_data_mat_train != NULL, "Train dataset is not fond");

    utils::IIterator<RowBatch>* iter = _data_mat_train->fmat()->RowIterator();
    iter->BeforeFirst();
    iter->Next();
    const RowBatch& batch = iter ->Value();
    for (size_t i = 0; i < batch.size; i ++) {
        real_t sum_out = 0.0;
        for (size_t k = _d_J; k < batch[i].length; k ++) {
            size_t index = batch[i][k].index - _d_J;
            real_t value = (real_t) batch[i][k].fvalue;
            sum_out += value * _model_weight[index];
        }
        sum_out += _model_bias;
        
        real_t p_out = sigmoid(sum_out);
        p_out = (p_out <=_options.lb_pi1) ? _options.lb_pi1:p_out;
        p_out = (p_out >=(1-_options.lb_pi1)) ? (1-_options.lb_pi1):p_out;
        vec[i] = p_out;
    }
}

int ipac::EMStageTwo() {
    // check status
    utils::Assert(_data_mat_train != NULL, "No train data error");
    if (! _options.more_mode) {
        utils::Assert(_stage_one_pi0 != 0 && _stage_one_pi1 != 0, 
            "pi_0 or pi_1 is zero, stage one run?");
    }

    // M * 1, when k = 1
    std::vector<real_t> R_1(_d_M, 0.0);
    // M * J, when k = 0
    std::vector<std::vector<real_t> > RZ_0;
    // M * J, when k = 1
    std::vector<std::vector<real_t> > RZ_1;

    for (int i = 0; i < _d_M; i ++) {
        std::vector<real_t> zero_vec(_d_J, 0);
        RZ_0.push_back(zero_vec);
        RZ_1.push_back(zero_vec);
    }

    std::vector<real_t> pred_1(_d_M,
        _stage_one_pi1);
    if (_options.more_mode && _model_valid) {
        init_pi1(pred_1);
    }
    std::vector<real_t> grad(_d_D, 0.0);
    init_stage_two();

    utils::IIterator<RowBatch>* iter = _data_mat_train->fmat()->RowIterator();
    iter->BeforeFirst();
    iter->Next();
    const RowBatch& batch = iter ->Value();
    utils::Assert(_options.cv_fold >= 1 && _options.cv_fold < (size_t)(_d_M / 2), 
        "Invalid cross validation folds");

    size_t each_fold_num = (size_t)(_d_M / _options.cv_fold);
    uint16_t train_fold_num = _options.cv_fold != 1? (_options.cv_fold - 1) : 1;
    std::vector<real_t> model_alpha(_alpha);
    
    // structure holding solution path
    _solution_path.clear();

    const int MIN_ITERATOR_NUM = 4;
    int ncore = omp_get_num_procs();
    int max_tn = _d_M / MIN_ITERATOR_NUM;
    int coreNum = max_tn > 2*ncore ? 2*ncore : max_tn;;
    for (uint16_t cv = 0; cv < _options.cv_fold; cv ++) {
        reset_stage_two_model(model_alpha);

        // set buffers for omp
        double** grad_buffer = new double*[coreNum];
        for (int i = 0; i < coreNum; i ++) {
            grad_buffer[i] = new double[_d_D];
            for (int j = 0; j < _d_D; j ++) {
                grad_buffer[i][j] = 0;
            }
        }

        for (size_t i = _off_set; i < _options.max_iters; i ++) {
            real_t sum_r1 = 0.0;
            real_t sum_weight_total = 0.0;
#pragma omp parallel for reduction(+:sum_r1,sum_weight_total) if( coreNum > 1) num_threads(coreNum)
            for (int t = cv * each_fold_num; 
                t < cv * each_fold_num + train_fold_num * each_fold_num; t ++) {
                size_t j = t % _d_M;  // train data index
                real_t prod_r0 = 0.0;
                real_t prod_r1 = 0.0;
                // prod(x) = exp(sum(log(x))
                for (int k = 0; k < _d_J; k ++) {
                    real_t beta_var = beta(_alpha[k], _data_labels[j][k]);
                    prod_r0 += log(_Q[k][0] * beta_var + 
                        1 - _Q[k][0]);
                    prod_r1 += log(_Q[k][1] * beta_var + 
                        1 - _Q[k][1]);

                    RZ_0[j][k] = _Q[k][0] * beta_var / (_Q[k][0] * beta_var +
                        1 - _Q[k][0]);
                    RZ_1[j][k] = _Q[k][1] * beta_var / (_Q[k][1] * beta_var +
                        1 - _Q[k][1]);
                }
                prod_r0 = (1 - pred_1[j]) * exp(prod_r0);
                prod_r1 = pred_1[j] * exp(prod_r1);

                R_1[j] = prod_r1 / (prod_r0 + prod_r1);
                sum_r1 += R_1[j] * _data_mat_train->info.labels[j];
                sum_weight_total += _data_mat_train->info.labels[j];

                for (int k = 0; k < _d_J; k ++) {
                    RZ_0[j][k] *= (1 - R_1[j]);
                    RZ_1[j][k] *= R_1[j];
                }
            }
            // end expectation

            // The maximization step
            // update alpha and Q
            // alpha_j = -sum_i{sum_k RZ_k[i][j]} / sum_i{sum_k RZ_k[i][j] * log(p_ij)}
            #pragma omp parallel for if( coreNum > 1) num_threads(coreNum)
            for (int j = 0; j < _d_J; j ++) {
                real_t sum_z1 = 0.0;
                real_t sum_z1p = 0.0;
                real_t sum_rz0 = 0.0;
                real_t sum_rz1 = 0.0;
                for (int k = 0; k < _d_M; k++) {
                    real_t sum_rz = (RZ_0[k][j] + RZ_1[k][j]) * _data_mat_train->info.labels[k];
                    sum_z1 += sum_rz;
                    sum_z1p += sum_rz * log(_data_labels[k][j]);
                
                    sum_rz0 += RZ_0[k][j] * _data_mat_train->info.labels[k];
                    sum_rz1 += RZ_1[k][j] * _data_mat_train->info.labels[k];
                }
                real_t tmp_alpha = - sum_z1 / sum_z1p;
                tmp_alpha = (tmp_alpha<=_options.lb_beta_alpha)?_options.lb_beta_alpha: tmp_alpha;
                tmp_alpha = (tmp_alpha>=(1-_options.lb_beta_alpha))?(1-_options.lb_beta_alpha):tmp_alpha;
                _alpha[j] = tmp_alpha;

                real_t q_zero = sum_rz0 / (sum_weight_total - sum_r1);
                q_zero = (q_zero <= _options.q_lb)?_options.q_lb:q_zero;
                q_zero = (q_zero >= (1 - _options.q_lb))?(1 - _options.q_lb):q_zero;
                _Q[j][0] = q_zero;

                real_t q_one = sum_rz1 / sum_r1;
                q_one = (q_one <= _options.q_lb)?_options.q_lb:q_one;
                q_one = (q_one >= (1 - _options.q_lb))?(1 - _options.q_lb):q_one;
                _Q[j][1] = q_one;
            }
            real_t sum_error = 0.0;
            grad.assign(_d_D, 0.0);

#pragma omp parallel for reduction(+:sum_error) if( coreNum > 1) num_threads(coreNum)
            for (int t = cv * each_fold_num; 
                t < cv * each_fold_num + train_fold_num * each_fold_num; t ++) {
                size_t j = t % _d_M;
                real_t error = (R_1[j] - pred_1[j]) * _data_mat_train->info.labels[j];
                sum_error += error;
                int thread_id = omp_get_thread_num();
                // update gradient information
                for (int k = _d_J; k < batch[j].length; k ++) {
                    // grad = (label - predict) * x, the gradient of the
                    // maximal likelihood
                    
                    size_t index = batch[j][k].index - _d_J;  // update grad for non-zero features
                    real_t value = (real_t) batch[j][k].fvalue;
                    grad_buffer[thread_id][index] += value * error;
                }
            }

            for (int l_i = 0; l_i < _d_D; l_i ++) {
                double grad_sum = 0.0;
                for (int l_j = 0; l_j < coreNum; l_j ++) {
                    grad_sum += grad_buffer[l_j][l_i];
                    grad_buffer[l_j][l_i] = 0.0;
                }
                grad[l_i] = grad_sum;
            }

            update_model(i, grad, sum_error);

            // update pred_1
            real_t sum_loglik = 0.0;
#pragma omp parallel for reduction(+:sum_loglik) if( coreNum > 1) num_threads(coreNum)
            for (int t = cv * each_fold_num; 
                t < cv * each_fold_num + train_fold_num * each_fold_num; t ++) {
                size_t j = t % _d_M;
                real_t sum_out = 0.0;
                for (size_t k = _d_J; k < batch[j].length; k ++) {
                    size_t index = batch[j][k].index - _d_J;
                    real_t value = (real_t) batch[j][k].fvalue;
                    sum_out += value * _model_weight[index];
                }
                sum_out += _model_bias;
                real_t pred_var = sigmoid(sum_out);
                pred_var = (pred_var <=_options.lb_pi1) ? _options.lb_pi1:pred_var;
                pred_var = (pred_var >=(1-_options.lb_pi1)) ? (1-_options.lb_pi1):pred_var;
                pred_1[j] = pred_var;

                // compute loglik
                real_t p_x = 0.0;
                real_t prod_k_one = 0.0;
                real_t prod_k_zero = 0.0;
                for (int k = 0; k < _d_J; k ++) {
                    real_t beta_var = beta(_alpha[k], _data_labels[j][k]);
                    prod_k_one += log(_Q[k][1] * beta_var + 1 - _Q[k][1]);
                    prod_k_zero += log(_Q[k][0] * beta_var + 1 - _Q[k][0]);
                }
                p_x = pred_1[j] * exp(prod_k_one) + (1-pred_1[j]) * exp(prod_k_zero);
                sum_loglik += log(p_x) * _data_mat_train->info.labels[j];
            }
            _stage_two_loglik[cv][i+1] = sum_loglik;  // mean of errors
            if (_options.verbose && i%1 == 0) {
                fprintf(stdout, "cv-train %d: Stage two iteration %d with loglikelihood: %f\n",
                    cv, (i+1), sum_loglik);
            }

            // validation only
            if (_options.cv_fold > 1) {
                real_t sum_loglik_valid = 0.0;
#pragma omp parallel for reduction(+:sum_loglik_valid) if( coreNum > 1) num_threads(coreNum)
                for (int t = cv * each_fold_num + train_fold_num * each_fold_num; 
                    t < cv * each_fold_num + _options.cv_fold * each_fold_num; t ++) {
                    size_t j = t % _d_M;

                    real_t sum_out = 0.0;
                    for (size_t k = _d_J; k < batch[j].length; k ++) {
                        size_t index = batch[j][k].index - _d_J;
                        real_t value = (real_t) batch[j][k].fvalue;
                        sum_out += value * _model_weight[index];
                    }
                    sum_out += _model_bias;
                    
                    real_t pred_var = sigmoid(sum_out);
                    pred_var = (pred_var <=_options.lb_pi1) ? _options.lb_pi1:pred_var;
                    pred_var = (pred_var >=(1-_options.lb_pi1)) ? (1-_options.lb_pi1):pred_var;

                    // compute loglik
                    real_t p_x = 0.0;
                    real_t prod_k_one = 0.0;
                    real_t prod_k_zero = 0.0;
                    for (int k = 0; k < _d_J; k ++) {
                        real_t beta_var = beta(_alpha[k], _data_labels[j][k]);
                        prod_k_one += log(_Q[k][1] * beta_var + 1 - _Q[k][1]);
                        prod_k_zero += log(_Q[k][0] * beta_var + 1 - _Q[k][0]);
                    }
                    p_x = pred_1[j] * exp(prod_k_one) + (1-pred_1[j]) * exp(prod_k_zero);
                    sum_loglik_valid += log(p_x) * _data_mat_train->info.labels[j];
                }
                _stage_two_loglikvalid[cv][i+1] = sum_loglik_valid;
                if (_options.verbose && i%1 == 0) {
                    fprintf(stdout, "cv-valid %d: Stage two iteration %d with loglikelihood: %f\n",
                        cv, (i+1), sum_loglik_valid);
                }
            }

            _iter_num[cv] = i + 1;
            // Can we stop
            if (_stage_two_loglik[cv][i+1] - _stage_two_loglik[cv][i] < _options.eps_stop_loglik * 
                    _stage_two_loglik[cv][i]) {
                fprintf(stdout, "cv %d: Iteration=%d,loglik=%f; Iteration=%d,loklik=%f\n",
                    cv, i, _stage_two_loglik[cv][i], (i+1), sum_loglik);
                break;
            }

            if (_options.eval_at_train) {
                real_t loglik = 0.0;
                Eval(loglik);
                _loglik_eval[i + 1] += loglik;
                if (_options.verbose && i%10 == 0) {
                    fprintf(stdout, "cv %d: Stage two iteration %d with eval loglik: %f\n",
                    cv, (i+1), loglik);
                }
            }

        }  // end i

        // free buffer
        for (int i = 0; i < coreNum; i ++) {
            delete[] grad_buffer[i];
        }
        delete[] grad_buffer;
        grad_buffer = NULL;
    }  // end cross validation cv

    if (_options.cv_fold > 1) {
        // reset alpha
        _alpha = model_alpha;
    } else {
        printf("_Q:\n");
        for (int i = 0; i < _d_J; i ++) {
            printf("%f,%f\n", _Q[i][0], _Q[i][1]);
        }
        printf("_alpha:\n");
        for(int i = 0; i < _d_J; i++) {
            printf("%f ", _alpha[i]);
        }
        printf("\n");
    }

    _model_valid = true;

    return get_best_iter_via_cv();
}

void ipac::update_model(int iter, std::vector<real_t>& grad, real_t sum_error) {
    if (grad.size() == 0) {
        return;
    }

    switch(_options.mode) {
    case L1: {
            // get the index with the maximal index
            size_t max_index = get_max_index(grad);
            // gradient ascent for the maximal likelihood
            _model_weight[max_index] += sign(grad[max_index]) * _options.eps;
            _model_bias += sign(sum_error) * _options.eps;
            // dump solution path, this may take extra time
            if (_options.cv_fold == 1 && _options.dump_solution_path) {
                std::pair<int, real_t> path_info = std::make_pair(iter, _model_weight[max_index]);
                _solution_path[max_index].push_back(path_info);
            }
        }
        break;
    case GRAD_DESCENT: {
            for (int i = 0; i < _d_D; i ++) {
                _model_weight[i] += grad[i] * _options.eps;
                _model_bias += sum_error * _options.eps;

                if (_options.cv_fold == 1 && _options.dump_solution_path) {
                    std::pair<int, real_t> path_info = std::make_pair(iter, _model_weight[i]);
                    _solution_path[i].push_back(path_info);
                }
            }
        }
        break;
    default:
        break;
    }
}

int ipac::TrainMore(int more_n)
{
    if (_options.init_train_only) {
        return 0;
    }

    _off_set = _options.max_iters;
    _options.max_iters += more_n;
    _options.more_mode = true;

    int ret = EMStageTwo();

    _options.more_mode = false;
    return ret;
}

int ipac::get_best_iter_via_cv()
{
    if (_options.cv_fold > 1) {
        // cross validation
        int best_iter = 0;
        real_t max_loglik = -inf;

        // diff CVs have diff iter number
        int common_iter_num = inf;
        for (size_t i = 0; i < _options.cv_fold; i ++) {
            common_iter_num = (_iter_num[i] < common_iter_num)? _iter_num[i]: common_iter_num;
        }

        for (size_t i = 0; i < common_iter_num; i ++) {
            real_t sum_loglik = 0.0;
            for (size_t j = 0; j < _options.cv_fold; j ++) {
                sum_loglik += _stage_two_loglikvalid[j][i];
            }
            sum_loglik /= _options.cv_fold;
            bool tag = sum_loglik > max_loglik;

            max_loglik = tag? sum_loglik: max_loglik;
            best_iter = tag? i: best_iter;
        }

        return best_iter + 1;
    }

    return 0;
}

int ipac::PreTrain(ipac_option_t option)
{
    set_option(option);
    EMStageOne();

    return 0;
}

int ipac::Train(ipac_option_t option)
{
    set_option(option);
    if (option.init_train_only) {
        return 0;
    }

    return EMStageTwo();
}

int ipac::Eval(real_t& loglik) {
    utils::Assert(_data_mat_test != NULL, "Test dataset is not fond");

    utils::IIterator<RowBatch>* iter = _data_mat_test->fmat()->RowIterator();
    iter->BeforeFirst();
    iter->Next();
    const RowBatch& batch = iter ->Value();
    loglik = 0.0;

    for (size_t i = 0; i < batch.size; i ++) {
        real_t sum_out = 0.0;
        real_t pred_var = _stage_one_pi1;

        if (_options.init_train_only == false) {
            for (size_t k = _d_J; k < batch[i].length; k ++) {
                size_t index = batch[i][k].index - _d_J;
                real_t value = (real_t) batch[i][k].fvalue;
                sum_out += value * _model_weight[index];
            }
            sum_out += _model_bias;
            pred_var = sigmoid(sum_out);
                    
            pred_var = (pred_var <=_options.lb_pi1) ? _options.lb_pi1:pred_var;
            pred_var = (pred_var >=(1-_options.lb_pi1)) ? (1-_options.lb_pi1):pred_var;
        }

        // compute loglik
        real_t p_x = 0.0;
        real_t prod_k_one = 0.0;
        real_t prod_k_zero = 0.0;
        for (int k = 0; k < _d_J; k ++) {
            real_t beta_var = beta(_alpha[k], _data_labels[i][k]);
            prod_k_one += log(_Q[k][1] * beta_var + 1 - _Q[k][1]);
            prod_k_zero += log(_Q[k][0] * beta_var + 1 - _Q[k][0]);
        }
        p_x = pred_var * exp(prod_k_one) + (1-pred_var) * exp(prod_k_zero);
        loglik += log(p_x) * _data_mat_test->info.labels[i];
    }

    return 0;
}

int ipac::dump_model(const char* f_name)
{
    fprintf(stdout, "Dumping the model ...\n");

    std::fstream f_writer(f_name, std::ios::out | std::ios::binary);
    if (f_writer.good() == false) {  // fail to open the file
        return -1;
    }

    // dump options
    f_writer.write((const char*)&_options, sizeof(ipac_option_t));

    // dump stage one pi
    f_writer.write((const char*)&_stage_one_pi1, sizeof(real_t));
    f_writer.write((const char*)&_model_valid, sizeof(bool));

    // dump the model weight
    f_writer.write((const char*)&_d_D, sizeof(size_t));
    if (_options.init_train_only == false) {
        size_t weight_dim = _model_weight.size();
        xgboost::utils::Assert(weight_dim == _d_D, "model dimension exception");
        for (size_t i = 0; i < weight_dim; i ++) {
            f_writer.write((const char*)&_model_weight[i], sizeof(real_t));
        }
        f_writer.write((const char*)&_model_bias, sizeof(real_t));
    }

    // dump model alpha
    f_writer.write((const char*)&_d_J, sizeof(size_t));
    for (size_t i = 0; i < _alpha.size(); i ++) {
        f_writer.write((const char*)&_alpha[i], sizeof(real_t));
    }

    // dump Q
    xgboost::utils::Assert(_Q.size() == _d_J, "Invalid Q");
    for (int i = 0; i < _d_J; i ++) {
        f_writer.write((const char*)&_Q[i][0], sizeof(real_t));
        f_writer.write((const char*)&_Q[i][1], sizeof(real_t));
    }

    f_writer.close();
    return 0;
}

int ipac::load_solution_path(const char* f_name, 
        std::map<int, std::vector<std::pair<int,real_t> > >& path)
{
    std::fstream f_reader(f_name, std::ios::in);
    if (f_reader.good() == false) {
        return -1;
    }
    
    path.clear();
    
    // begin to load the file
    char read_buffer[READ_BUFFER_LEN];
    while (f_reader.eof() == false) {
        f_reader.getline(read_buffer, READ_BUFFER_LEN);
        if (read_buffer == NULL || read_buffer[0] == '\0') {
            break;
        }
        
        std::vector<std::string> tokens;
        boost::split(tokens, read_buffer, boost::is_any_of(" "));
        int fea_index = boost::lexical_cast<int>(tokens[0]);
        for (size_t i = 1; i < tokens.size(); i ++) {
            std::vector<std::string> sub_tokens;
            boost::split(sub_tokens, tokens[i], boost::is_any_of(":"));
            if (sub_tokens.size() != 2) {
                f_reader.close();
                return -1;
            }
            
            std::pair<int, real_t> path_info = 
                std::make_pair(
                    boost::lexical_cast<int>(sub_tokens[0]),
                    boost::lexical_cast<real_t>(sub_tokens[1])
                );
                
            path[fea_index].push_back(path_info);
        }
    }
    
    f_reader.close();
    
    return 0;
}

int ipac::dump_solution_path(const char* f_name, bool append_mode)
{
    fprintf(stdout, "Dump the solution path.....\n");
    
    std::map<int, std::vector<std::pair<int,real_t> > > path;
    if (append_mode) {
        if (load_solution_path(f_name, path) != 0) {
            fprintf(stderr, "Error in loading the original path");
            return -1;
        }
    }
    
    std::fstream f_writer(f_name, std::ios::out);
    if (f_writer.good() == false) {  // fail to open the file
        return -1;
    }
    
    for (size_t i = 0; i < _d_D; i ++) {
        bool fea_exist = (path.find(i) != path.end()) || 
            (_solution_path.find(i) != _solution_path.end());
        if (fea_exist == false) {
            continue;
        }
        
        f_writer << i;
        if (path.find(i) != path.end()) {
            for (size_t j = 0; j < path[i].size(); j ++) {
                f_writer << " " << path[i][j].first << ":" <<
                    path[i][j].second;
            }
        }
        
        if (_solution_path.find(i) != _solution_path.end()) {
            for (size_t j = 0; j < _solution_path[i].size(); j ++) {
                f_writer << " " << _solution_path[i][j].first << ":" <<
                    _solution_path[i][j].second;
            }
        }
        
        f_writer << std::endl;
    }
    
    f_writer.close();
    
    return 0;
}

int ipac::load_model(const char* f_name)
{
    fprintf(stdout, "Restoring the model ...\n");

    std::fstream f_reader(f_name, std::ios::in | std::ios::binary);
    if (f_reader.good() == false) {
        return -1;
    }

    // recover options
    f_reader.read((char*)&_options, sizeof(ipac_option_t));

    // load stage one pi
    f_reader.read((char*)&_stage_one_pi1, sizeof(size_t));
    _stage_one_pi0 = 1 - _stage_one_pi1;

    f_reader.read((char*)&_model_valid, sizeof(bool));

    // recover model weight
    size_t weight_dim = 0;
    f_reader.read((char*)&weight_dim, sizeof(size_t));

    if (_options.init_train_only == false) {
        _model_weight.resize(weight_dim);
        for (size_t i = 0; i < weight_dim; i ++) {
            real_t value = 0;
            f_reader.read((char *)&value, sizeof(real_t));
            _model_weight[i] = value;
        }
        f_reader.read((char*)&_model_bias, sizeof(real_t));
    }

    size_t label_num = 0;
    f_reader.read((char*)&label_num, sizeof(size_t));
    _alpha.resize(label_num);
    xgboost::utils::Assert(label_num == _d_J, "p value numbers are different between model and data");
    for (int i = 0; i < label_num; i ++) {
        // recover model alpha
        f_reader.read((char*)&_alpha[i], sizeof(real_t));
    }
    // recover Q
    _Q.clear();
    for (int i = 0; i < label_num; i ++) {
        real_t q_0 = 0;
        real_t q_1 = 0;
        f_reader.read((char*)&q_0, sizeof(real_t));
        f_reader.read((char*)&q_1, sizeof(real_t));

        std::vector<real_t> q;
        q.push_back(q_0);
        q.push_back(q_1);

        _Q.push_back(q);
    }

    f_reader.close();

    return 0;
}

void ipac::dump_info(const char* info_file)
{

    std::fstream f_writer(info_file, std::ios::out);
    if (f_writer.good() == false) {  // fail to open the file
        return;
    }

    // dump Q
    f_writer << "Q";
    for (int i = 0; i < _d_J; i ++) {
        f_writer << " " << _Q[i][0] << "," << _Q[i][1];
    }
    f_writer << std::endl;

    // dump alpha
    f_writer << "alpha";
    for (int i = 0; i < _d_J; i ++) {
        f_writer << " " << _alpha[i];
    }
    f_writer << std::endl;
    
    // model_weight
    f_writer << "model";
    if (_options.init_train_only == false) {
        f_writer << " " << _model_bias;  // bias
        for (int i = 0; i < _d_D; i ++) {
            f_writer << " " << _model_weight[i];
        }
    }
    f_writer << std::endl;
    
    // stage one pi
    f_writer << "stageone_pi1 " << _stage_one_pi1 << std::endl;
    
    // loglik stage one
    f_writer << "loglik_stag_1";
    for (int i = 1; i < _stage_one_loglik.size(); i ++) {
        if (_stage_one_loglik[i] == 0) {
            continue;
        }

        f_writer << " " << i << ":" << _stage_one_loglik[i];
    }
    f_writer << std::endl;
    
    // loglik stage two
    f_writer << "loglik_stag_2";
    if (_options.init_train_only == false)  {
        for (int i = 1; i < _stage_two_loglik[0].size(); i ++) {
            if (_stage_two_loglik[0][i] == 0) {
                continue;
            }

            f_writer << " " << _stage_two_loglik[0][i];
        }
    }
    f_writer << std::endl;
    
    f_writer.close();
}

}  // namespace ipac
}  // namespace xgboost
