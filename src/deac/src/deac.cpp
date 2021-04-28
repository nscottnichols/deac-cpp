#include <stdio.h>
#include <math.h> // cosh
#include <stdlib.h>
#include <iostream>
#include <tuple> // for tie() and tuple
#include <argparse.hpp>
#include <rng.hpp>
#include <memory> // string_format
#include <string> // string_format
#include <stdexcept> // string_format
#include <fstream> // std::ofstream
#include <filesystem>
namespace fs = std::filesystem;

void write_to_logfile(fs::path filename, std::string log_message ) {
    std::ofstream ofs(filename.c_str(), std::ios_base::out | std::ios_base::app );
    ofs << log_message << std::endl;
    ofs.close();
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args ) {
    //See https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
    int size_s = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

void write_array(fs::path filename, double * buffer, int length) {
    FILE * output_file;
    output_file = fopen (filename.c_str(), "wb");
    fwrite (buffer , sizeof(double), static_cast<size_t>(length), output_file);
    fclose (output_file);
}

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

void matrix_multiply_MxN_by_Nx1(double * C, double * A, double * B, int M, int N) {
    for (int i=0; i<M; i++) {
        C[i] = 0.0;
        for (int j=0; j<N; j++) {
            C[i] += A[i*N + j]*B[j];
        }
    }
}

void matrix_multiply_LxM_by_MxN(double * C, double * A, double * B, int L,
        int M, int N) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<L; j++) {
            C[i*L + j] = 0.0;
            for (int k=0; k<M; k++) {
                C[i*L + j] += A[j*M + k]*B[i*M + k];
            }
        }
    }
}

double reduced_chi_square_statistic(double * observed, double * calculated,
        double * error, int length) {
    double chi_squared = 0.0;
    for (int i=0; i<length; i++) {
        chi_squared += pow((observed[i] - calculated[i])/error[i],2);
    }
    return chi_squared;
}

double minimum(double * A, int length) {
    double _minimum = A[0];
    for (int i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
        }
    }
    return _minimum;
}

int argmin(double * A, int length) {
    int _argmin=0;
    double _minimum = A[0];
    for (int i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
            _argmin = i;
        }
    }
    return _argmin;
}

std::tuple <int, double> argmin_and_min(double * A, int length) {
    int _argmin=0;
    double _minimum = A[0];
    for (int i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
            _argmin = i;
        }
    }
    std::tuple <int, double> argmin_tuple(_argmin, _minimum);
    return argmin_tuple;
}

double mean(double * A, int length) {
    double _mean = 0.0;
    for (int i=0; i<length; i++) {
        _mean += A[i];
    }
    return _mean/length;
}

double standard_deviation(double * A, double mean, int length) {
    double _std = 0.0;
    for (int i=0; i<length; i++) {
        _std += pow(A[i] - mean,2);
    }
    return sqrt(_std/length);
}

std::tuple <int, int, int> get_mutant_indices(struct xoshiro256p_state * rng,
        int mutant_index0, int length) {
    int mutant_index1 = mutant_index0;
    int mutant_index2 = mutant_index0;
    int mutant_index3 = mutant_index0;
    while (mutant_index1 == mutant_index0) {
        mutant_index1 = xoshiro256p(rng) % length;
    }

    while ((mutant_index2 == mutant_index0) || (mutant_index2 == mutant_index1)) {
        mutant_index2 = xoshiro256p(rng) % length;
    }

    while ((mutant_index3 == mutant_index0) || (mutant_index3 == mutant_index1)
            || (mutant_index3 == mutant_index2)) {
        mutant_index3 = xoshiro256p(rng) % length;
    }

    std::tuple <int, int, int> _mutant_indices(mutant_index1, mutant_index2, mutant_index3);
    return _mutant_indices;
}

void set_mutant_indices(struct xoshiro256p_state * rng, int * mutant_indices,
        int mutant_index0, int length) {
    mutant_indices[0] = mutant_index0;
    mutant_indices[1] = mutant_index0;
    mutant_indices[2] = mutant_index0;
    while (mutant_indices[0] == mutant_index0) {
        mutant_indices[0] = xoshiro256p(rng) % length;
    }

    while ((mutant_indices[1] == mutant_index0) || (mutant_indices[1] == mutant_indices[0])) {
        mutant_indices[1] = xoshiro256p(rng) % length;
    }

    while ((mutant_indices[2] == mutant_index0) || (mutant_indices[2] == mutant_indices[0])
            || (mutant_indices[2] == mutant_indices[1])) {
        mutant_indices[2] = xoshiro256p(rng) % length;
    }
}

void deac(struct xoshiro256p_state * rng, double * const imaginary_time,
        double * const isf, double * const isf_error, double * frequency,
        double temperature, int number_of_generations, int number_of_timeslices, int population_size,
        int genome_size, bool normalize, bool use_inverse_first_moment, 
        double first_moment, double third_moment, double third_moment_error,
        double crossover_probability,
        double self_adapting_crossover_probability,
        double differential_weight, 
        double self_adapting_differential_weight_probability,
        double self_adapting_differential_weight_shift,
        double self_adapting_differential_weight, double stop_minimum_fitness,
        bool track_stats, int seed, fs::path save_directory) {

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
        matrix_multiply_MxN_by_Nx1(normalization, population_old,
                normalization_term, population_size, genome_size);
        for (int i=0; i<population_size; i++) {
            double _norm = normalization[i];
            for (int j=0; j<genome_size; j++) {
                population_old[i*genome_size + j] *= zeroth_moment/_norm;
            }
        }
    }

    double * first_moments_term;
    double * first_moments;
    if (use_first_moment) {
        first_moments_term = (double*) malloc(sizeof(double)*genome_size);
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
            first_moments_term[j] = df*f*sinh(0.5*beta*f);
        }

        first_moments = (double*) malloc(sizeof(double)*population_size);
        matrix_multiply_MxN_by_Nx1(first_moments, population_old,
                first_moments_term, population_size, genome_size);
    }

    double * third_moments_term;
    double * third_moments;
    if (use_third_moment) {
        third_moments_term = (double*) malloc(sizeof(double)*genome_size);
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
            third_moments_term[j] = df*pow(f,3)*sinh(0.5*beta*f);
        }

        third_moments = (double*) malloc(sizeof(double)*population_size);
        matrix_multiply_MxN_by_Nx1(third_moments, population_old,
                third_moments_term, population_size, genome_size);
    }

    //set isf_model and calculate fitness
    double * isf_model;
    isf_model = (double*) malloc(sizeof(double)*number_of_timeslices*population_size);

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
    matrix_multiply_LxM_by_MxN(isf_model, isf_term, population_old,
            number_of_timeslices, genome_size, population_size);

    double * inverse_first_moments_term;
    double * inverse_first_moments;
    double inverse_first_moment = 0.0;
    double inverse_first_moment_error = 0.0;
    if (use_inverse_first_moment) {
        inverse_first_moments_term = (double*) malloc(sizeof(double)*number_of_timeslices);
        for (int j=0; j<number_of_timeslices; j++) {
            double dt;
            if (j==0) {
                dt = 0.5*(imaginary_time[j+1] - imaginary_time[j]);
            } else if (j == number_of_timeslices - 1) {
                dt = 0.5*(imaginary_time[j] - imaginary_time[j-1]);
            } else {
                dt = 0.5*(imaginary_time[j+1] - imaginary_time[j-1]);
            }
            inverse_first_moments_term[j] = dt;
            inverse_first_moment += isf[j]*dt;
            inverse_first_moment_error = pow(isf_error[j],2) * pow(dt,2);
        }
        inverse_first_moment_error = sqrt(inverse_first_moment_error);

        inverse_first_moments = (double*) malloc(sizeof(double)*population_size);
        matrix_multiply_MxN_by_Nx1(inverse_first_moments, isf_model,
                inverse_first_moments_term, population_size, number_of_timeslices);
    }

    double * fitness_old;
    fitness_old = (double*) malloc(sizeof(double)*population_size);

    for (int i=0; i<population_size; i++) {
        double _fitness = reduced_chi_square_statistic(isf,
                isf_model + i*number_of_timeslices, isf_error,
                number_of_timeslices)/number_of_timeslices;
        if (use_inverse_first_moment) {
            _fitness += pow((inverse_first_moment - inverse_first_moments[i])/inverse_first_moment_error,2);
        }
        if (use_first_moment) {
            _fitness += pow(first_moments[i] - first_moment,2)/first_moment;
        }
        if (use_third_moment) {
            _fitness += pow((third_moment - third_moments[i])/third_moment_error,2);
        }
        fitness_old[i] = _fitness;
    }

    double * crossover_probabilities_old;
    double * crossover_probabilities_new;
    crossover_probabilities_old = (double*) malloc(sizeof(double)*population_size);
    crossover_probabilities_new = (double*) malloc(sizeof(double)*population_size);
    for (int i=0; i<population_size; i++) {
        crossover_probabilities_old[i] = crossover_probability;
    }

    double * differential_weights_old;
    double * differential_weights_new;
    differential_weights_old = (double*) malloc(sizeof(double)*population_size);
    differential_weights_new = (double*) malloc(sizeof(double)*population_size);
    for (int i=0; i<population_size; i++) {
        differential_weights_old[i] = differential_weight;
    }

    //Initialize statistics arrays
    double * fitness_mean;
    double * fitness_minimum;
    double * fitness_standard_deviation;
    if (track_stats) {
        fitness_mean = (double*) malloc(sizeof(double)*number_of_generations);
        fitness_minimum = (double*) malloc(sizeof(double)*number_of_generations);
        fitness_standard_deviation = (double*) malloc(sizeof(double)*number_of_generations);
    }
    
    bool * mutate_indices;
    mutate_indices = (bool*) malloc(sizeof(bool)*genome_size*population_size);

    int * mutant_indices;
    mutant_indices = (int*) malloc(sizeof(int)*3*population_size);

    double minimum_fitness;
    int minimum_fitness_idx;
    
    int generation;
    for (int ii=0; ii < number_of_generations - 1; ii++) {
        generation = ii;
        minimum_fitness = minimum(fitness_old,population_size);

        //Get Statistics
        if (track_stats) {
            fitness_mean[ii] = mean(fitness_old, population_size);
            fitness_minimum[ii] = minimum_fitness;
            fitness_standard_deviation[ii] = standard_deviation(fitness_old,
                    fitness_mean[ii], population_size);
        }
        
        //Stopping criteria
        if (minimum_fitness <= stop_minimum_fitness) {
            break;
        }


        //Set crossover probabilities
        for (int i=0; i<population_size; i++) {
            if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                crossover_probabilities_new[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53;
            } else {
                crossover_probabilities_new[i] = crossover_probabilities_old[i];
            }

            if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                //differential_weights_new[i] = 
                //    self_adapting_differential_weight_shift + 
                //    self_adapting_differential_weight*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
                differential_weights_new[i] = 2.0*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
            } else {
                differential_weights_new[i] = differential_weights_old[i];
            }
        }

        //Set mutant population and indices 
        for (int i=0; i<population_size; i++) {
            double crossover_rate = crossover_probabilities_new[i];
            set_mutant_indices(rng, mutant_indices + 3*i, i, population_size);
            for (int j=0; j<genome_size; j++) {
                mutate_indices[i*genome_size + j] = (xoshiro256p(rng) >> 11) * 0x1.0p-53 < crossover_rate;
            }
        }

        for (int i=0; i<population_size; i++) {
            double F = differential_weights_new[i];
            int mutant_index1 = mutant_indices[3*i];
            int mutant_index2 = mutant_indices[3*i + 1];
            int mutant_index3 = mutant_indices[3*i + 2];
            for (int j=0; j<genome_size; j++) {
                bool mutate = mutate_indices[i*genome_size + j];
                if (mutate) {
                    population_new[i*genome_size + j] = fabs( 
                        population_old[mutant_index1*genome_size + j] + F*(
                                population_old[mutant_index2*genome_size + j] -
                                population_old[mutant_index3*genome_size + j]));
                } else {
                    population_new[i*genome_size + j] = population_old[i*genome_size + j];
                }
            }
        }

        // Normalization
        if (normalize) {
            matrix_multiply_MxN_by_Nx1(normalization, population_new,
                    normalization_term, population_size, genome_size);
            for (int i=0; i<population_size; i++) {
                double _norm = normalization[i];
                for (int j=0; j<genome_size; j++) {
                    population_new[i*genome_size + j] *= zeroth_moment/_norm;
                }
            }
        }

        //Rejection
        //Set model isf for new population
        matrix_multiply_LxM_by_MxN(isf_model, isf_term, population_new,
                number_of_timeslices, genome_size, population_size);

        //Set moments
        if (use_inverse_first_moment) {
            matrix_multiply_MxN_by_Nx1(inverse_first_moments, isf_model,
                    inverse_first_moments_term, population_size,
                    number_of_timeslices);
        }
        if (use_first_moment) {
            matrix_multiply_MxN_by_Nx1(first_moments, population_new,
                    first_moments_term, population_size, genome_size);
        }
        if (use_third_moment) {
            matrix_multiply_MxN_by_Nx1(third_moments, population_new,
                    third_moments_term, population_size, genome_size);
        }

        //Set fitness for new population
        for (int i=0; i<population_size; i++) {
            double _fitness = reduced_chi_square_statistic(isf,
                    isf_model + i*number_of_timeslices, isf_error,
                    number_of_timeslices)/number_of_timeslices;
            if (use_inverse_first_moment) {
                _fitness += pow((inverse_first_moment - inverse_first_moments[i])/inverse_first_moment_error,2);
            }
            if (use_first_moment) {
                _fitness += pow(first_moments[i] - first_moment,2)/first_moment;
            }
            if (use_third_moment) {
                _fitness += pow((third_moment - third_moments[i])/third_moment_error,2);
            }
            // Rejection step
            if (_fitness <= fitness_old[i]) {
                fitness_old[i] = _fitness;
                crossover_probabilities_old[i] = crossover_probabilities_new[i];
                differential_weights_old[i] = differential_weights_new[i];
                for (int j=0; j<genome_size; j++) {
                    population_old[i*genome_size + j] = population_new[i*genome_size + j];
                }
            }
        }
    }

    std::tie(minimum_fitness_idx,minimum_fitness) = argmin_and_min(fitness_old,population_size);

    double * best_dsf;
    best_dsf = (double*) malloc(sizeof(double)*genome_size);
    for (int i=0; i<genome_size; i++) {
        double f = frequency[i];
        best_dsf[i] = 0.5*population_old[genome_size*minimum_fitness_idx + i]*exp(0.5*beta*f);
    }

    //Get Statistics
    if ((track_stats) && (generation == number_of_generations - 2)) {
        generation += 1;
        fitness_mean[generation] = mean(fitness_old, population_size);
        fitness_minimum[generation] = minimum_fitness;
        fitness_standard_deviation[generation] = standard_deviation(fitness_old,
                fitness_mean[generation], population_size);
    }

    //Save data
    std::string best_dsf_filename_str = string_format("deac_dsf_%06d.bin",seed);
    fs::path best_dsf_filename = save_directory / best_dsf_filename_str;
    write_array(best_dsf_filename, best_dsf, genome_size);
    std::string frequency_filename_str = string_format("deac_frequency_%06d.bin",seed);
    fs::path frequency_filename = save_directory / frequency_filename_str;
    write_array(frequency_filename, frequency, genome_size);
    fs::path fitness_mean_filename;
    fs::path fitness_minimum_filename;
    fs::path fitness_standard_deviation_filename;
    if (track_stats) {
        std::string fitness_mean_filename_str = string_format("deac_stats_fitness-mean_%06d.bin",seed);
        std::string fitness_minimum_filename_str = string_format("deac_stats_fitness-minimum_%06d.bin",seed);
        std::string fitness_standard_deviation_filename_str = string_format("deac_stats_fitness-standard-deviation_%06d.bin",seed);
        fs::path fitness_mean_filename = save_directory / fitness_mean_filename_str;
        fs::path fitness_minimum_filename = save_directory / fitness_minimum_filename_str;
        fs::path fitness_standard_deviation_filename = save_directory / fitness_standard_deviation_filename_str;
        write_array(fitness_mean_filename, fitness_mean, generation + 1);
        write_array(fitness_minimum_filename, fitness_minimum, generation + 1);
        write_array(fitness_standard_deviation_filename, fitness_standard_deviation, generation + 1);
    }

    //Write to log file
    std::string log_filename_str = string_format("deac_log_%06d.bin",seed);
    fs::path log_filename = save_directory / log_filename_str;
    std::ofstream log_ofs(log_filename.c_str(), std::ios_base::out | std::ios_base::app );

    //Input parameters
    log_ofs << "temperature: " << temperature << std::endl;
    log_ofs << "number_of_generations: " << number_of_generations << std::endl;
    log_ofs << "number_of_timeslices: " << number_of_timeslices << std::endl;
    log_ofs << "population_size: " << population_size << std::endl;
    log_ofs << "genome_size: " << genome_size << std::endl;
    log_ofs << "normalize: " << normalize << std::endl;
    log_ofs << "use_inverse_first_moment: " << use_inverse_first_moment << std::endl;
    log_ofs << "first_moment: " << first_moment << std::endl;
    log_ofs << "third_moment: " << third_moment << std::endl;
    log_ofs << "third_moment_error: " << third_moment_error << std::endl;
    log_ofs << "crossover_probability: " << crossover_probability << std::endl;
    log_ofs << "self_adapting_crossover_probability: " << self_adapting_crossover_probability << std::endl;
    log_ofs << "differential_weight: " << differential_weight << std::endl;
    log_ofs << "self_adapting_differential_weight_probability: " << self_adapting_differential_weight_probability << std::endl;
    log_ofs << "self_adapting_differential_weight_shift: " << self_adapting_differential_weight_shift << std::endl;
    log_ofs << "self_adapting_differential_weight: " << self_adapting_differential_weight << std::endl;
    log_ofs << "stop_minimum_fitness: " << stop_minimum_fitness << std::endl;
    log_ofs << "track_stats: " << track_stats << std::endl;
    log_ofs << "seed: " << seed << std::endl;

    //Generated variables
    log_ofs << "best_dsf_filename: " << best_dsf_filename << std::endl;
    log_ofs << "frequency_filename: " << frequency_filename << std::endl;
    log_ofs << "generation: " << generation << std::endl;
    if (track_stats) {
        log_ofs << "fitness_mean_filename: " << fitness_mean_filename << std::endl;
        log_ofs << "fitness_minimum_filename: " << fitness_minimum_filename << std::endl;
        log_ofs << "fitness_standard_deviation_filename: " << fitness_standard_deviation_filename << std::endl;
    }
    log_ofs.close();

    //Free memory
    free(isf_term);
    free(population_old);
    free(population_new);
    free(fitness_old);
    if (normalize) {
        free(normalization_term);
        free(normalization);
    }
    if (use_first_moment) {
        free(first_moments_term);
        free(first_moments);
    }
    if (use_third_moment) {
        free(third_moments_term);
        free(third_moments);
    }
    free(isf_model);
    if (use_inverse_first_moment) {
        free(inverse_first_moments_term);
        free(inverse_first_moments);
    }
    free(crossover_probabilities_old);
    free(crossover_probabilities_new);
    free(differential_weights_old);
    free(differential_weights_new);
    if (track_stats) {
        free(fitness_mean);
        free(fitness_minimum);
        free(fitness_standard_deviation);
    }
    free(mutate_indices);
    free(mutant_indices);
    free(best_dsf);
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
    program.add_argument("--save_directory")
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
    std::string isf_file = program.get<std::string>("isf_file");
    std::tie(numpy_data,number_of_elements) = load_numpy_array(isf_file);
    int number_of_timeslices = number_of_elements/3;

    double * const imaginary_time = numpy_data;
    double * const isf = numpy_data + number_of_timeslices;
    double * const isf_error = numpy_data + 2*number_of_timeslices;

    uint64_t seed = 1407513600 + static_cast<uint64_t>(program.get<int>("--seed"));
    struct xoshiro256p_state rng = xoshiro256p_init(seed);

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
    
    double stop_minimum_fitness = program.get<double>("--stop_minimum_fitness");

    bool track_stats = program.get<bool>("--track_stats");
    std::string save_directory_str = program.get<std::string>("--save_directory");
    fs::path save_directory(save_directory_str);
    fs::create_directory(save_directory);

    //Write to log file
    std::string log_filename_str = string_format("deac_log_%06d.bin",seed);
    fs::path log_filename = save_directory / log_filename_str;
    std::ofstream log_ofs(log_filename.c_str(), std::ios_base::out | std::ios_base::app );

    //Input parameters
    log_ofs << "isf_file: " << isf_file << std::endl;
    log_ofs.close();

    deac( &rng, imaginary_time, isf, isf_error, frequency, temperature,
            number_of_generations, number_of_timeslices, population_size, genome_size,
            normalize, use_inverse_first_moment, first_moment, third_moment,
            third_moment_error, crossover_probability,
            self_adapting_crossover_probability, differential_weight,
            self_adapting_differential_weight_probability,
            self_adapting_differential_weight_shift,
            self_adapting_differential_weight, stop_minimum_fitness,
            track_stats, seed, save_directory);

    free(numpy_data);
    free(frequency);
    return 0;
}
