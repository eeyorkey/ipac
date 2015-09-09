#include <omp.h>
#include <fstream>
#include <time.h>
#include <map>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "ipac.h"

#define BOUNDS(p_out) do {\
    p_out = (p_out <=_options.lb_pi1) ? _options.lb_pi1:p_out;\
    p_out = (p_out >=(1-_options.lb_pi1)) ? (1-_options.lb_pi1):p_out;\
} while (0)

namespace xgboost {
namespace ipac {

ipac::ipac()
{
    _data_mat_test = NULL;
    _data_mat_train = NULL;
    _d_M = 0;
    _d_D = 0;
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
    _data_labels.clear();
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
}

int ipac::LoadData(const char* f_name, std::string tag) {
    utils::Assert(tag == "train" || tag == "test", "Tag can only be "
        "train or test");
    utils::IIterator<RowBatch>* iter;

    if (tag == "train") {
        _data_mat_train = io::LoadDataMatrix(f_name);
        // initialize the dimension of the weight matrix
        _d_M = _data_mat_train->info.info.num_row;         // num of instances
        _d_D = _data_mat_train->info.info.num_col - 1;  // num of features

        iter = _data_mat_train->fmat()->RowIterator();
        
    } else {
        _data_mat_test = io::LoadDataMatrix(f_name);

        _d_M = _data_mat_test->info.info.num_row;
        _d_D = _data_mat_test->info.info.num_col - 1;

        iter = _data_mat_test->fmat()->RowIterator();
    }

    iter->BeforeFirst();
    iter->Next();
    const RowBatch& batch = iter->Value();

    for (int i = 0; i < _d_M; i ++) {
        real_t data_label_i = 0.0;
        
        float p = batch[i][0].fvalue;
        // check data, p_value ranges: p_value >= 0 && p_value <= 1
        utils::Assert(p >= 0 && p <= 1, "p value should be (0, 1],"
            " but it current value is: p=%f", p);
        if (fabs(p) < zero)
            data_label_i = zero;
        else
            data_label_i = p;
        
        _data_labels.push_back(data_label_i);
    }

    return 0;
}

int ipac::EMStageOne()
{
    utils::Assert(_data_mat_train != NULL, "No training data error");

    _stage_one_pi0 = 1 - _options.init_pi;
    real_t beta_alpha_init = _options.init_betamean / (1 - _options.init_betamean);
    std::vector<real_t> Z0(_d_M, 0);
    std::vector<real_t> Z1(_d_M, 0);
    std::vector<real_t> beta_var(_d_M, 0.0);

    // init
    _stage_one_loglik.resize(_options.max_iters_init + 1);
    _stage_one_loglik[0] = -inf;

    for (int i = 0; i < _d_M; i ++) {
        beta_var[i] = beta(beta_alpha_init, 
            _data_labels[i]);
    }

    const int MIN_ITERATOR_NUM = 4;
    int ncore = omp_get_num_procs();
    int max_tn = _d_M / MIN_ITERATOR_NUM;
    int coreNum = max_tn > 2*ncore ? 2*ncore : max_tn;
    
    real_t weight_sum = 0.0;
    for (int j = 0; j < _d_M; j ++) {
        weight_sum += _data_mat_train->info.labels[j];
    }

    // begin to calculate
    for (size_t i = 0; i < _options.max_iters_init; i ++) {
        // The expectation
        real_t sum_z1 = 0.0;
        real_t sum_z1p = 0.0;

#pragma omp parallel for reduction(+:sum_z1,sum_z1p) if( coreNum > 1) num_threads(coreNum)
        for (int j = 0; j < _d_M; j ++) {
            Z0[j] = _stage_one_pi0;
            Z1[j] = (1-Z0[j]) * beta_var[j];
            real_t sum_var = Z0[j] + Z1[j];

            utils::Assert(sum_var != 0,
                "z0+z1=0?");

            Z0[j] = Z0[j] / sum_var;
            Z1[j] = 1 - Z0[j];
            sum_z1 += Z1[j] * _data_mat_train->info.labels[j];
            sum_z1p += Z1[j] * (-log(_data_labels[j])) * _data_mat_train->info.labels[j];
            
        }

        // The maximization step
        _alpha = sum_z1 / sum_z1p;
        _alpha = (_alpha<=_options.lb_beta_alpha)?_options.lb_beta_alpha: _alpha;
        _alpha = (_alpha>=(1-_options.lb_beta_alpha))?(1-_options.lb_beta_alpha):_alpha;

        // update p0_init and beta_var, loglikelihood
        real_t sum_loglik = 0.0;
        _stage_one_pi1 = sum_z1 / weight_sum;
        _stage_one_pi1 = (_stage_one_pi1<=_options.lb_pi1)?_options.lb_pi1:_stage_one_pi1;
        _stage_one_pi1 = (_stage_one_pi1 >= (1-_options.lb_pi1)) ? (1-_options.lb_pi1): _stage_one_pi1;
        _stage_one_pi0 = 1 - _stage_one_pi1;
        
#pragma omp parallel for reduction(+:sum_loglik) if( coreNum > 1) num_threads(coreNum)
        for (int j = 0; j < _d_M; j ++) {
            beta_var[j] = beta(_alpha, _data_labels[j]);
            // p(X|theta)=P(X|Z=0;theta)p(Z=0)+P(X|Z=1;theta)p(Z=1)
            real_t p_x = _stage_one_pi0 + 
                _stage_one_pi1 * beta_var[j];
            sum_loglik += log(p_x) * _data_mat_train->info.labels[j];
        }
        _stage_one_loglik[i+1] = sum_loglik;
        if (_options.verbose && i % 10 == 0) {
            fprintf(stdout, "Stage one iteration %d with loglikelihood: %f\n",
                (i+1), sum_loglik);
        }
        
        // Can we stop
        if (_stage_one_loglik[i+1] - _stage_one_loglik[i] < _options.eps_stop_loglik * _stage_one_loglik[i]) {
            break;
        }
    }

    printf("alpha=%f\n", _alpha);
    printf("stage_one_pi1=%f\n", _stage_one_pi1);

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

void ipac::reset_stage_two_model(real_t model_alpha) {
    if (_options.more_mode && _model_valid)
        return;

    // init
    _model_weight.clear();
    _stage_two_pi0.clear();
    _stage_two_pi1.clear();
    _model_weight.resize(_d_D, 0.0);
    _stage_two_pi0.resize(_d_M, 0.0);
    _stage_two_pi1.resize(_d_M, 0.0);
    for (size_t i = 0; i < _d_D; i ++) {
        _model_weight[i] = 0;
    }
    _model_bias = log(_stage_one_pi1 / _stage_one_pi0);
    _alpha = model_alpha;
}

void ipac::init_pi1(std::vector<real_t>& vec) {
    utils::Assert(_data_mat_train != NULL, "Train dataset is not fond");

    utils::IIterator<RowBatch>* iter = _data_mat_train->fmat()->RowIterator();
    iter->BeforeFirst();
    iter->Next();
    const RowBatch& batch = iter->Value();
    for (size_t i = 0; i < batch.size; i ++) {
        real_t sum_out = 0.0;
        for (size_t k = 1; k < batch[i].length; k ++) {
            size_t index = batch[i][k].index - 1;
            real_t value = (real_t) batch[i][k].fvalue;
            sum_out += value * _model_weight[index];
        }
        sum_out += _model_bias;
        
        real_t var = sigmoid(sum_out);
        BOUNDS(var);
        vec[i] = var;
    }
}

int ipac::EMStageTwo() {
    // check status
    utils::Assert(_data_mat_train != NULL, "No train data error");
    if (! _options.more_mode) {
        utils::Assert(_stage_one_pi0 != 0 && _stage_one_pi1 != 0, 
            "pi_0 or pi_1 is zero, stage one run?");
    }

    std::vector<real_t> Z0(_d_M, 0);
    std::vector<real_t> Z1(_d_M, 0);
    std::vector<real_t> beta_var(_d_M, 0.0);
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
    real_t model_alpha = _alpha;
    
    // structure holding solution path
    _solution_path.clear();

    const int MIN_ITERATOR_NUM = 4;
    int ncore = omp_get_num_procs();
    int max_tn = _d_M / MIN_ITERATOR_NUM;
    int coreNum = max_tn > 2*ncore ? 2*ncore : max_tn;;
    for (uint16_t cv = 0; cv < _options.cv_fold; cv ++) {
        reset_stage_two_model(model_alpha);

        for (int i = 0; i < _d_M; i ++) {
            beta_var[i] = beta(_alpha, _data_labels[i]);
        }

        // set buffers for omp
        double** grad_buffer = new double*[coreNum];
        for (int i = 0; i < coreNum; i ++) {
            grad_buffer[i] = new double[_d_D];
            for (int j = 0; j < _d_D; j ++) {
                grad_buffer[i][j] = 0;
            }
        }

        for (size_t i = _off_set; i < _options.max_iters; i ++) {
            // Expectation
            real_t sum_z1 = 0.0;
            real_t sum_z1p = 0.0;

#pragma omp parallel for reduction(+:sum_z1,sum_z1p) if( coreNum > 1) num_threads(coreNum)
            for (int t = cv * each_fold_num; 
                t < cv * each_fold_num + train_fold_num * each_fold_num; t ++) {
                size_t j = t % _d_M;  // train data index
                Z0[j] = 1 - pred_1[j];
                Z1[j] = pred_1[j] * beta_var[j];
                real_t sum_z = Z0[j] + Z1[j];

                utils::Assert(sum_z != 0,
                    "z0 + z1 = 0?");

                Z0[j] = Z0[j] / sum_z;
                Z1[j] = 1 - Z0[j];

                sum_z1 += Z1[j] * _data_mat_train->info.labels[j];
                sum_z1p += Z1[j] * (-log(_data_labels[j])) * _data_mat_train->info.labels[j];
            }
            // end expectation

            // Max
            _alpha = sum_z1 / sum_z1p;
            _alpha = (_alpha<=_options.lb_beta_alpha)?_options.lb_beta_alpha: _alpha;
            _alpha = (_alpha>=(1-_options.lb_beta_alpha))?(1-_options.lb_beta_alpha):_alpha;

            real_t sum_error = 0.0;
            grad.assign(_d_D, 0.0);

#pragma omp parallel for reduction(+:sum_error) if( coreNum > 1) num_threads(coreNum)
            for (int t = cv * each_fold_num; 
                t < cv * each_fold_num + train_fold_num * each_fold_num; t ++) {
                size_t j = t % _d_M;
                // X_j * beta
                real_t sum_out = 0.0;
                // sparse matrix multiplication
                for (int k = 1; k < batch[j].length; k ++) {
                    size_t index = batch[j][k].index - 1;
                    real_t value = (real_t) batch[j][k].fvalue;
                    sum_out += value * _model_weight[index];
                }
                sum_out += _model_bias;

                real_t pred = sigmoid(sum_out);
                BOUNDS(pred);
                real_t error = (Z1[j] - pred) * _data_mat_train->info.labels[j];
                sum_error += error;
                int thread_id = omp_get_thread_num();
                // update gradient information
                for (int k = 1; k < batch[j].length; k ++) {
                    // grad = (label - predict) * x, the gradient of the
                    // maximal likelihood
                    
                    size_t index = batch[j][k].index - 1;  // update grad for non-zero features
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

            // get the index with the maximal index
            size_t max_index = get_max_index(grad);
            // gradient ascent for the maximal likelihood
            _model_weight[max_index] += sign(grad[max_index]) * _options.eps;
            _model_bias += sign(sum_error) * _options.eps;
            
            // dump solution path, this may take extra time
            if (_options.cv_fold == 1 && _options.dump_solution_path) {
                std::pair<int, real_t> path_info = std::make_pair(i, _model_weight[max_index]);
                _solution_path[max_index].push_back(path_info);
            }

            // update pred_1 and beta_var
            real_t sum_loglik = 0.0;
#pragma omp parallel for reduction(+:sum_loglik) if( coreNum > 1) num_threads(coreNum)
            for (int t = cv * each_fold_num; 
                t < cv * each_fold_num + train_fold_num * each_fold_num; t ++) {
                size_t j = t % _d_M;
                real_t sum_out = 0.0;
                for (size_t k = 1; k < batch[j].length; k ++) {
                    size_t index = batch[j][k].index - 1;
                    real_t value = (real_t) batch[j][k].fvalue;
                    sum_out += value * _model_weight[index];
                }
                sum_out += _model_bias;
                real_t pred_var = sigmoid(sum_out);
                BOUNDS(pred_var);
                pred_1[j] = pred_var;

                beta_var[j] = beta(_alpha, _data_labels[j]);

                // p(X|theta)=P(X|Z=0;theta)p(Z=0)+P(X|Z=1;theta)p(Z=1)
                sum_loglik += log((1-pred_1[j]) + pred_1[j] * beta_var[j]) * _data_mat_train->info.labels[j];
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
                    for (size_t k = 1; k < batch[j].length; k ++) {
                        size_t index = batch[j][k].index - 1;
                        real_t value = (real_t) batch[j][k].fvalue;
                        sum_out += value * _model_weight[index];
                    }
                    sum_out += _model_bias;
                    real_t pred_var = sigmoid(sum_out);
                    BOUNDS(pred_var);

                    sum_loglik_valid += log( (1-pred_var) + 
                        pred_var * beta(_alpha, _data_labels[j]) ) * _data_mat_train->info.labels[j];
                }
                _stage_two_loglikvalid[cv][i+1] = sum_loglik_valid;
                if (_options.verbose && i%1 == 0) {
                    fprintf(stdout, "cv-valid %d: Stage two iteration %d with loglikelihood: %f\n",
                        cv, (i+1), sum_loglik_valid);
                }
            }

            _iter_num[cv] = i + 1;
            // Can we stop
            if (_stage_two_loglik[cv][i+1] - _stage_two_loglik[cv][i] < _options.eps_stop_loglik * _stage_two_loglik[cv][i]) {
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
    }

    printf("alpha=%f\n", _alpha);
    printf("bias: %f\n", _model_bias);
    printf("weight:");
    for (int i = 0; i < _model_weight.size(); i ++) {
      printf(" %f", _model_weight[i]);
    }
    printf("\n");

    _model_valid = true;

    return get_best_iter_via_cv();
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
    const RowBatch& batch = iter->Value();
    loglik = 0.0;
    
    for (size_t i = 0; i < batch.size; i ++) {
        real_t sum_out = 0.0;
        real_t pred_var = _stage_one_pi1;
        
        if (_options.init_train_only == false) {
            for (size_t k = 1; k < batch[i].length; k ++) {
                size_t index = batch[i][k].index - 1;
                real_t value = (real_t) batch[i][k].fvalue;
                sum_out += value * _model_weight[index];
            }
            sum_out += _model_bias;
            pred_var = sigmoid(sum_out);
            pred_var = (pred_var <=_options.lb_pi1) ? _options.lb_pi1:pred_var;
            pred_var = (pred_var >=(1-_options.lb_pi1)) ? (1-_options.lb_pi1):pred_var;
        }

        loglik += log( (1-pred_var) + 
            pred_var * beta(_alpha, _data_labels[i]) ) * _data_mat_test->info.labels[i];
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
    
    // dump model alpha
    f_writer.write((const char*)&_alpha, sizeof(real_t));

    // weight dimension
    f_writer.write((const char*)&_d_D, sizeof(size_t));
    
    // dump the model weight
    if (_options.init_train_only == false) {
        size_t weight_dim = _d_D;
        for (size_t i = 0; i < weight_dim; i ++) {
            f_writer.write((const char*)&_model_weight[i], sizeof(real_t));
        }
        f_writer.write((const char*)&_model_bias, sizeof(real_t));
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

    // recover model alpha
    f_reader.read((char*)&_alpha, sizeof(real_t));

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

    f_reader.close();

    return 0;
}

void ipac::dump_info(const char* info_file)
{

    std::fstream f_writer(info_file, std::ios::out);
    if (f_writer.good() == false) {  // fail to open the file
        return;
    }

    // dump alpha
    f_writer << "alpha" << _alpha << std::endl;
    
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

            f_writer << " " << i << ":" << _stage_two_loglik[0][i];
        }
    }
    f_writer << std::endl;
}

}  // namespace ipac
}  // namespace xgboost
