#include <stdio.h>
#include <math.h> // cosh
#include <stdlib.h>
#include <iostream>
#include <tuple> // for tie() and tuple
#include <argparse.hpp>
#include <rng.hpp>

std::tuple <double*, unsigned int> load_numpy_array(std::string isf_file) {
    FILE * input_file;
    long file_size_bytes;
    double * buffer;
    size_t result;
  
    input_file = fopen( isf_file.c_str(), "rb" );
    if (input_file==NULL) {fputs("File error",stderr); exit(1);}
  
    // obtain file size:
    fseek(input_file , 0 , SEEK_END);
    file_size_bytes = ftell(input_file);
    rewind(input_file);
    
    unsigned int number_of_elements = static_cast<unsigned int> (file_size_bytes/sizeof(double));
    std::cout << "number of doubles: " << number_of_elements << std::endl;
  
    // allocate memory to contain the whole file:
    buffer = (double*) malloc(sizeof(char)*file_size_bytes);
    if (buffer == NULL) {fputs("Memory error",stderr); exit(2);}
  
    // copy the file into the buffer:
    result = fread(buffer,1,file_size_bytes,input_file);
    if (result != file_size_bytes) {fputs("Reading error",stderr); exit(3);}
  
    /* the whole file is now loaded in the memory buffer. */
    fclose (input_file);

    std::tuple <double*,unsigned int> numpy_data_tuple(buffer, number_of_elements);
    return numpy_data_tuple;
}

double* deac(struct xoshiro256p_state * rng, double * const imaginary_time,
        double * const isf, double * const isf_error, double * frequency,
        double temperature, int number_of_generations, int number_of_timeslices, int population_size,
        int genome_size, bool normalize, bool use_inverse_first_moment, 
        double first_moment, double third_moment, double third_moment_error,
        double crossover_probability,
        double self_adapting_crossover_probability,
        double differential_weight, 
        double self_adapting_differential_weight_probability,
        double self_adapting_differential_weight_shift,
        double self_adapting_differential_weight, bool track_stats) {

    double beta = 1.0/temperature;
    double zeroth_moment = isf[0];
    bool use_first_moment = first_moment >= 0.0;
    bool use_third_moment = third_moment >= 0.0;

    double * isf_term;
    isf_term = (double*) malloc(sizeof(double)*genome_size*number_of_timeslices);
    for (int i=0; i<number_of_timeslices; i++) {
        double t = imaginary_time[i];
        double bo2mt = 0.5*beta - t;
        for (int j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            int isf_term_idx = i*genome_size + j;
            isf_term[isf_term_idx] = df*cosh(bo2mt*f);
        }
    }

    //Generate population and set initial fitness
    double * population_old;
    double * population_new;
    population_old = (double*) malloc(sizeof(double)*genome_size*population_size);
    population_new = (double*) malloc(sizeof(double)*genome_size*population_size);
    for (int i=0; i<genome_size*population_size; i++) {
        population_old[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53; // to_double2
    }

    // Normalize population
    double * normalization_term;
    double * normalization;
    if (normalize) {
        normalization_term = (double*) malloc(sizeof(double)*genome_size);
        for (int j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            normalization_term[j] = df*cosh(0.5*beta*f);
        }
        normalization = (double*) malloc(sizeof(double)*population_size);
        //FIXME need to set normalization here (used matrix multiplication in Julia code)
        //mul!(normalization,P',normalization_term);
        for (int i=0; i<population_size; i++) {
            double _norm = normalization[i];
            for (int j=0; j<genome_size; j++) {
                population_old[i*genome_size + j] *= zeroth_moment/_norm;
            }
        }
    }

    double * fitness_old;
    double * fitness_new;
    fitness_old = (double*) malloc(sizeof(double)*population_size);
    fitness_new = (double*) malloc(sizeof(double)*population_size);

    double * best_dsf;
    best_dsf = (double*) malloc(sizeof(double)*genome_size);
    for (int i=0; i<genome_size; i++) {
        best_dsf[i] = static_cast<double>(i);
    }

    free(isf_term);
    free(population_old);
    free(population_new);
    free(fitness_old);
    free(fitness_new);
    if (normalize) {
        free(normalization_term);
        free(normalization);
    }
    return best_dsf;
}

int main (int argc, char *argv[]) {
    argparse::ArgumentParser program("DEAC");
    program.add_argument("-T", "--temperature")
        .help("Temperature of system.")
        .default_value(0.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-N", "--number_of_generations")
        .help("Number of generations before genetic algorithm quits.")
        .default_value(100000)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-P","--population_size")
        .help("Size of initial population")
        .default_value(512)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-M","--genome_size")
        .help("Size of genome.")
        .default_value(512)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("--omega_max")
        .help("Maximum frequency to explore.")
        .default_value(60.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--normalize")
        .help("Normalize spectrum to the zeroeth moment.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--use_inverse_first_moment")
        .help("Calculate inverse first moment from ISF data and use it in fitness.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--first_moment")
        .help("FIXME First moment.")
        .default_value(-1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--third_moment")
        .help("FIXME Third moment.")
        .default_value(-1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--third_moment_error")
        .help("FIXME Third moment error.")
        .default_value(0.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-r","--crossover_probability")
        .help("Initial probability for parent gene to become mutant vector gene.")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-u","--self_adapting_crossover_probability")
        .help("Probability for `crossover_probability` to mutate.")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-F","--differential_weight")
        .help("Initial weight factor when creating mutant vector.")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-v","--self_adapting_differential_weight_probability")
        .help("Probability for `differential_weight` to mutate.")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-l","--self_adapting_differential_weight_shift")
        .help("If `self_adapting_differential_weight_probability` mutate, new value is `l + m*rand()`.")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-m","--self_adapting_differential_weight")
        .help("If `self_adapting_differential_weight_probability` mutate, new value is `l + m*rand()`.")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--stop_minimum_fitness")
        .help("Stopping criteria, if minimum fitness is below `stop_minimum_fitness` stop evolving.")
        .default_value(1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--seed")
        .help("Seed to pass to random number generator.")
        .default_value(0)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("--save_state")
        .help("Save state of DEAC algorithm. Saves the random number generator, population, and population fitness.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--save_file_dir")
        .help("Directory to save results in.")
        .default_value("./deacresults");
    program.add_argument("--track_stats")
        .help("Track minimum fitness and other stats.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("isf_file")
        .help("binary file containing isf data (tau, isf, error)");
    try {
      program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(0);
    }

    unsigned int number_of_elements;
    double* numpy_data;
    auto isf_file = program.get<std::string>("isf_file");
    std::tie(numpy_data,number_of_elements) = load_numpy_array(isf_file);
    int number_of_timeslices = number_of_elements/3;

    double * const imaginary_time = numpy_data;
    double * const isf = numpy_data + number_of_timeslices;
    double * const isf_error = numpy_data + 2*number_of_timeslices;
    std::cout << "imaginary_time: ";
    for (int i = 0; i < number_of_timeslices; i++) {
        std::cout << imaginary_time[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "isf: ";
    for (int i = 0; i < number_of_timeslices; i++) {
        std::cout << isf[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "isf_error: ";
    for (int i = 0; i < number_of_timeslices; i++) {
        std::cout << isf_error[i] << " ";
    }
    std::cout << std::endl;

    uint64_t seed = 1407513600 + static_cast<uint64_t>(program.get<int>("--seed"));
    struct xoshiro256p_state rng = xoshiro256p_init(seed);
    std::cout << "generate 10 random unsigned ints:" << std::endl;
    for (int i=0; i < 10; i++) {
        uint64_t _randint = xoshiro256p(&rng);
        std::cout << _randint << std::endl;
    }

    std::cout << "generate 1000 random 64-bit floats [0,1):" << std::endl;
    for (int i=0; i < 1000; i++) {
        uint64_t _randint = xoshiro256p(&rng);
        double _randdouble = (_randint >> 11) * 0x1.0p-53;
        std::cout << _randdouble << std::endl;
    }

    double temperature = program.get<double>("--temperature");
    int number_of_generations = program.get<int>("--number_of_generations");
    int population_size = program.get<int>("--population_size");
    int genome_size = program.get<int>("--genome_size");
    double max_frequency = program.get<double>("--omega_max");
    
    double * frequency;
    frequency = (double*) malloc(sizeof(double)*genome_size);
    double dfrequency = max_frequency/(genome_size - 1);
    for (int i=0; i<genome_size; i++) {
        frequency[i] = i*dfrequency;
    }

    bool normalize = program.get<bool>("--normalize");
    bool use_inverse_first_moment = program.get<bool>("--use_inverse_first_moment");
    double first_moment = program.get<double>("--first_moment");
    double third_moment = program.get<double>("--third_moment");
    double third_moment_error = program.get<double>("--third_moment_error");

    double crossover_probability = program.get<double>("--crossover_probability");
    double self_adapting_crossover_probability = program.get<double>("--self_adapting_crossover_probability");
    double differential_weight = program.get<double>("--differential_weight");
    double self_adapting_differential_weight_probability = program.get<double>("--self_adapting_differential_weight_probability");
    double self_adapting_differential_weight_shift = program.get<double>("--self_adapting_differential_weight_shift");
    double self_adapting_differential_weight = program.get<double>("--self_adapting_differential_weight");

    bool track_stats = program.get<bool>("--track_stats");

    double *best_dsf = deac( &rng, imaginary_time, isf, isf_error, frequency,
            temperature, number_of_generations, number_of_timeslices, population_size, genome_size,
            normalize, use_inverse_first_moment, first_moment, third_moment,
            third_moment_error, crossover_probability,
            self_adapting_crossover_probability, differential_weight,
            self_adapting_differential_weight_probability,
            self_adapting_differential_weight_shift,
            self_adapting_differential_weight, track_stats);
    std::cout << "best_dsf: " << std::endl;
    for (int i=0; i<genome_size; i++) {
        std::cout << best_dsf[i] << std::endl;
    }

    free(numpy_data);
    free(frequency);
    free(best_dsf);
    return 0;
}
