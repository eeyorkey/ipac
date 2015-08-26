#include <iostream>
#include <stdlib.h>
#include <string>
#include <omp.h>
#include <boost/lexical_cast.hpp>
using namespace std;

#define XGU xgboost::utils
#define XPAC xgboost::ipac

#include "ipac.h"

#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#endif

const char* prog_short_options = "k:i:l:t:e:m:n:s:r:c:j:x:o:p:h";
struct option long_options[] = {
    {"init_train_n", 1, NULL, 'k'},
    {"init_train_only", 1, NULL, 'i'},
	{"label_num", 1, NULL, 'l'},
	{"train_path", 1, NULL, 't'},
	{"test_path", 1, NULL, 'e'},
	{"more_n", 1, NULL, 'm'},
	{"iter_n", 1, NULL, 'n'},
	{"stop_eps", 1, NULL, 's'},
	{"learning_rate", 1, NULL, 'r'},
	{"cv_folds", 1, NULL, 'c'},
	{"eval_when_train", 1, NULL, 'j'},
	{"exe_mode", 1, NULL, 'x'},
	{"model_path", 1, NULL, 'o'},
    {"figure_path", 1, NULL, 'p'},
	{"help", 0, NULL, 'h'},
	{NULL, 0, NULL, 0}
};
void print_help(int argc, char** argv);

int main(int argc, char** argv)
{
	if (argc == 1) {
		print_help(argc, argv);
	}

	// parameters
	string training_path = "";
	string testing_path = "";
	int more_n = 0;
	int iter_n = 2000;
	double eps = 1e-4;
	double learning_rate = 0.005;
	int cv_folds = 1;
	bool eval_when_train = false;
	string output_model_path = "";
	string exe_mode = "train";
	int label_num = 0;
    bool init_train_only = false;
    string figure_path = "";
    int init_train_n = 1000;

	int ret_val;
	try {
		while ((ret_val = getopt_long(argc, argv, prog_short_options, long_options,
				NULL)) != -1) {
			switch(ret_val) {
			case 'h':
				print_help(argc, argv);
				break;
			case 't':
				training_path = optarg;
				break;
			case 'e':
				testing_path = optarg;
				break;
			case 'm':
				more_n = boost::lexical_cast<int>(optarg);
				break;
            case 'k':
                init_train_n = boost::lexical_cast<int>(optarg);
                break;
			case 'n':
				iter_n = boost::lexical_cast<int>(optarg);
				break;
			case 's':
				eps = boost::lexical_cast<double>(optarg);
				break;
			case 'r':
				learning_rate = boost::lexical_cast<double>(optarg);
				break;
			case 'c':
				cv_folds = boost::lexical_cast<int>(optarg);
				break;
			case 'j':
				eval_when_train = boost::lexical_cast<bool>(optarg);
				break;
            case 'i':
                init_train_only = boost::lexical_cast<bool>(optarg);
                break;
			case 'x':
				exe_mode = optarg;
				break;
			case 'l':
				label_num = boost::lexical_cast<int>(optarg);
				break;
			case 'o':
				output_model_path = optarg;
				break;
            case 'p':
                figure_path = optarg;
                break;
			default:
				cerr << "Get option exception" << endl;
				return -1;
				break;
			}
		}
	} catch (boost::bad_lexical_cast& err) {
		cout << err.what() << endl;
		return -1;
	}

	if (label_num == 0) {
		cerr << "Please specify the number of classes or labels" << endl;
		return -1;
	}

	XPAC::ipac ipac;
	XPAC::ipac_option_t options;
	options.cv_fold = cv_folds;
	options.max_iters = iter_n;
    options.max_iters_init = init_train_n;
	options.eps_stop_loglik = eps;
	options.eps = learning_rate;
	options.eval_at_train = eval_when_train;
    options.init_train_only = init_train_only;
    if (figure_path != "") {
        options.dump_solution_path = true;
    }

	ipac.set_soft_label_num(label_num);

	if (exe_mode == "train") {
		XGU::Check(training_path != "", "Training data is missing");
		if (eval_when_train) {
			XGU::Check(testing_path != "", "Eval during training need testing data");
			ipac.LoadData(testing_path.c_str(), "test");
		}
		XGU::Check(output_model_path != "", "Model path is missing");

		XGU::Assert(ipac.LoadData(training_path.c_str()) == 0, "Fail to load training data");
		
		double start_time = omp_get_wtime();
		ipac.PreTrain(options);

		if (cv_folds > 1) {
            if (init_train_only == false) {
			    int best_iter = ipac.Train(options);
			    options.cv_fold = 1;
			    options.max_iters = best_iter;

			    cout << "The number of iteration chosen by cv is: " << best_iter << endl;
			    ipac.Train(options);
            }
		} else {
			// diretly train and exit
			ipac.Train(options);
		}
		double end_time = omp_get_wtime();
		cout << "Elapse time: " << end_time-start_time << " seconds"<< endl; 

		XGU::Assert(ipac.dump_model(output_model_path.c_str()) == 0, "Fail to dump the model");

        if (figure_path != "") {
            ipac.dump_solution_path(figure_path.c_str(), false);
        }

	} else if (exe_mode == "test") {
		XGU::Check(testing_path != "" && output_model_path != "", "Either tesing data path or "
			"model path is missing");
		XGU::Assert(ipac.LoadData(testing_path.c_str(), "test") == 0, "Fail to load testing data");

		// load model
		XGU::Assert(ipac.load_model(output_model_path.c_str()) == 0, "Fail to load the model");

		real_t eval_loglik = -inf;
		ipac.Eval(eval_loglik);
		cout << "The log likelihood value is: " << eval_loglik << endl;

	} else if (exe_mode == "more") {
		XGU::Check(training_path != "" && output_model_path != "", "Either training data "
			"the model path is missing");
		XGU::Check(more_n >= 1, "More iteration number should be >=1. Currently, more_n=%d",
			more_n);

		XGU::Assert(ipac.LoadData(training_path.c_str()) == 0, "Fail to load training data");
		XGU::Assert(ipac.load_model(output_model_path.c_str()) == 0, "Fail to load the model");

		double start_time = omp_get_wtime();
		ipac.TrainMore(more_n);
        double end_time = omp_get_wtime();
		cout << "Elapse time: " << end_time-start_time << " seconds" << endl; 

		string final_model_name = output_model_path + "_more";
		XGU::Assert(ipac.dump_model(final_model_name.c_str()) == 0, "Fail to dump the model");

        if (figure_path != "") {
            ipac.dump_solution_path(figure_path.c_str(), true);
        }
	} else {
		cerr << "Unknown execution mode: " << exe_mode << endl;
		return -1;
	}

	return 0;
}

void print_help(int argc, char** argv)
{
	cout << argv[0] << "  options(values). Options are as follows:" << endl;
	cout << "\t-h|--help\t\tThis help" << endl;
	cout << "\t-l|--init_train_n\tIteration number for pre-training" << endl;
	cout << "\t-i|--init_train_only\tOnly run pre-training" << endl;
	cout << "\t-l|--label_num\t\tThe number of classes" << endl;
	cout << "\t-x|--exe_mode\t\tExecution mode(train|test|more)" << endl;
	cout << "\t-t|--train_path\t\tTraining data path(file name)" << endl;
	cout << "\t-e|--test_path\t\tTesting data path(file name)" << endl;
	cout << "\t-m|--more_n\t\tMore iteratons numers(Integer >= 1)" << endl;
	cout << "\t-n|--iter_n\t\tMax iteration (Integer >=1)" << endl;
	cout << "\t-s|--stop_eps\t\tLikelihood delta for stop (Float)" << endl;
	cout << "\t-r|--learning_rate\tLearning rate for boosting (Float)" << endl;
	cout << "\t-c|--cv_folds\t\tCV folds for cross-validation (Integer >=1)" << endl;
	cout << "\t-j|--eval_when_train\tTesting during training (0|1)" << endl;
	cout << "\t-o|--model_file\t\tIpac model file name (file name)" << endl;

	exit(0);
}
