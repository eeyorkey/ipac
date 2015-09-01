#ifndef __IPAC_H__
#define __IPAC_H__

#include <math.h>
#include <map>
#include <string>

#include "data.h"
#include "io.h"

namespace xgboost {
namespace ipac {

#define real_t double
#define inf 1e20
#define zero 1e-30
#define READ_BUFFER_LEN 10240

struct ipac_option_t {
    size_t max_iters;
    size_t max_iters_init;
    real_t eps_stop_loglik;
    real_t init_betamean;
    real_t init_pi;
    real_t lb_pi1;
    real_t lb_beta_alpha;
    real_t eps;
    uint16_t cv_fold;
    bool eval_at_train;
    bool verbose;
    bool more_mode;
    bool dump_solution_path;
    bool init_train_only;
    
    ipac_option_t() : max_iters_init(10000), max_iters(2000), eps_stop_loglik(1e-4), init_betamean(0.1),
        init_pi(0.1), lb_pi1(0.001), lb_beta_alpha(0.001), eps(0.005), cv_fold(1), 
        eval_at_train(false), verbose(true), more_mode(false), dump_solution_path(false),
        init_train_only(false) {
    }
};

class ipac
{
public:
    ipac();
    ~ipac();

    int LoadData(const char* f_name, std::string tag="train");
    int PreTrain(ipac_option_t option);
    int Train(ipac_option_t option);
    int TrainMore(int more_n);
    int Eval(real_t& loglik);
    int dump_model(const char* f_name);
    int load_model(const char* f_name);
    int dump_solution_path(const char* f_name, bool append_mode);

    void set_option(ipac_option_t& option){_options = option;}
    void get_option(ipac_option_t& option) {option = _options;}

    void dump_info(const char* info_file);

protected:
    real_t beta(real_t alpha, real_t p){return alpha * pow(p, (alpha-1.0));}
    int EMStageOne();
    int EMStageTwo();
    int get_best_iter_via_cv();
    real_t sigmoid(real_t out) {return 1.0 / (1.0 + exp(-out));}
    size_t get_max_index(std::vector<real_t>& data) {
        real_t value = fabs(data[0]);
        size_t index = 0;
        for(size_t i = 1; i < data.size(); i ++) {
            if (fabs(data[i]) > value) {
                index = i;
                value = fabs(data[i]);
            }
        }

        return index;
    }
    real_t sign(real_t value) {
        return value > 0.0? 1.0:-1.0;
    }
    void init_stage_two();
    void reset_stage_two_model(real_t model_alpha);
    void init_pi1(std::vector<real_t>& vec);
    
    int load_solution_path(const char* f_name, 
        std::map<int, std::vector<std::pair<int,real_t> > >& path);

private:
    // model cnfigurations
    ipac_option_t _options;
    size_t _off_set;  // offset of iterations for the more mode

    io::DataMatrix* _data_mat_train;
    io::DataMatrix* _data_mat_test;

    // data dimension
    size_t _d_M;
    size_t _d_D;

    // stage one parameters
    real_t _stage_one_pi0;
    real_t _stage_one_pi1;
    std::vector<real_t> _stage_one_loglik;
    
    // stage two parametrs
    std::vector<real_t> _stage_two_pi0;
    std::vector<real_t> _stage_two_pi1;
    std::vector<std::vector<real_t> > _stage_two_loglik;
    std::vector<std::vector<real_t> > _stage_two_loglikvalid;
    std::vector<real_t> _loglik_eval;
    std::vector<size_t> _iter_num;
    
    std::vector<real_t> _data_labels;

    // model weight
    bool _model_valid;
    real_t _alpha;
    std::vector<real_t> _model_weight;  // beta
    real_t _model_bias;  // beta_0
    
    // data structure for solution path
    std::map<int, std::vector<std::pair<int,real_t> > > _solution_path;

};
}

}

#endif
