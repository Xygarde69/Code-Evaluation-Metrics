#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <ctype.h>
#include <json-c/json.h> // For JSON parsing
#include <sys/time.h>    // For gettimeofday

// --- Configuration & Constants ---
#define MAX_TESTS 20
#define TIMEOUT_SECONDS 5
#define MAX_OUTPUT_SIZE 4096
#define MEMORY_LIMIT_MB 64
#define CPU_TIME_LIMIT_S 2
#define MAX_INPUT_SIZE 1024
#define MAX_EXPECTED_OUTPUT_SIZE 1024
#define MAX_DESCRIPTION_SIZE 256

#define RESULTS_JSON_PATH "/tmp/eval_results.json"
#define VALGRIND_LOG_PATH "/tmp/valgrind_log.txt"
#define EXEC_FAILURE_EXIT_CODE 127
#define ROBUSTNESS_SIGINT_WAIT_US 200000 // 200ms

// --- Enhanced Structs ---
typedef struct {
    char input[MAX_INPUT_SIZE];
    char expected_output[MAX_EXPECTED_OUTPUT_SIZE];
    char description[MAX_DESCRIPTION_SIZE];
    char category[32]; // normal, edge, error, corner
    float weight;      // Test importance weight
} DynamicTestCase;

typedef struct {
    DynamicTestCase tests[MAX_TESTS];
    int num_tests;
    char program_description[512];
    char program_type[64];
    char difficulty_level[32];
    char potential_edge_cases[MAX_TESTS][256];
    int num_edge_cases;
} TestSuite;

typedef struct {
    float passrate;
    float memory_score;
    float robustness_score;
    float weighted_score; // New: weight-based scoring
    long execution_time_ms;
    int tests_passed;
    int tests_failed;
    char failed_tests[MAX_TESTS][512]; // Details of failed tests
    int num_failed_details;
} EnhancedEvalMetrics;

// --- Global State ---
char executable_path[256];
char temp_dir_path[256];
TestSuite test_suite;

// --- Function Prototypes ---
void cleanup(void);
void handle_signal(int sig);
long current_time_ms(void);
void set_child_resource_limits(void);
int compile_source(const char *source_filename);
int run_test_process(const char *input, char *output_buffer, size_t buffer_size);
int load_test_cases_from_json(const char *json_file);
float calculate_dynamic_passrate(EnhancedEvalMetrics *metrics);
float analyze_memory(void);
float check_robustness(void);
void write_enhanced_results_to_json(const EnhancedEvalMetrics *metrics);
void trim_trailing_whitespace(char *str);
void print_test_suite_info(void);

// --- JSON Loading Functions ---

/**
 * @brief Loads test cases from LLM-generated JSON file
 */
int load_test_cases_from_json(const char *json_file) {
    FILE *file = fopen(json_file, "r");
    if (!file) {
        fprintf(stderr, "❌ Cannot open test cases file: %s\n", json_file);
        return -1;
    }

    // Read entire file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char *json_string = malloc(file_size + 1);
    if (!json_string) {
        perror("malloc for json_string failed");
        fclose(file);
        return -1;
    }
    fread(json_string, 1, file_size, file);
    json_string[file_size] = '\0';
    fclose(file);

    // Parse JSON
    json_object *root = json_tokener_parse(json_string);
    if (!root) {
        fprintf(stderr, "❌ Invalid JSON in test cases file\n");
        free(json_string);
        return -1;
    }

    // Extract program metadata
    json_object *desc_obj, *type_obj, *diff_obj, *tests_obj;
    
    if (json_object_object_get_ex(root, "program_description", &desc_obj)) {
        strncpy(test_suite.program_description, json_object_get_string(desc_obj), 
                sizeof(test_suite.program_description) - 1);
    }
    
    if (json_object_object_get_ex(root, "program_type", &type_obj)) {
        strncpy(test_suite.program_type, json_object_get_string(type_obj), 
                sizeof(test_suite.program_type) - 1);
    }
    
    if (json_object_object_get_ex(root, "difficulty_level", &diff_obj)) {
        strncpy(test_suite.difficulty_level, json_object_get_string(diff_obj), 
                sizeof(test_suite.difficulty_level) - 1);
    }

    // Extract test cases
    if (!json_object_object_get_ex(root, "test_cases", &tests_obj)) {
        fprintf(stderr, "❌ No test_cases found in JSON\n");
        json_object_put(root);
        free(json_string);
        return -1;
    }

    int array_len = json_object_array_length(tests_obj);
    test_suite.num_tests = (array_len > MAX_TESTS) ? MAX_TESTS : array_len;

    for (int i = 0; i < test_suite.num_tests; i++) {
        json_object *test_obj = json_object_array_get_idx(tests_obj, i);
        json_object *input_obj, *output_obj, *desc_obj_tc, *cat_obj, *weight_obj;

        if (json_object_object_get_ex(test_obj, "input", &input_obj)) {
            strncpy(test_suite.tests[i].input, json_object_get_string(input_obj), 
                    sizeof(test_suite.tests[i].input) - 1);
        }

        if (json_object_object_get_ex(test_obj, "expected_output", &output_obj)) {
            strncpy(test_suite.tests[i].expected_output, json_object_get_string(output_obj), 
                    sizeof(test_suite.tests[i].expected_output) - 1);
        }

        if (json_object_object_get_ex(test_obj, "description", &desc_obj_tc)) {
            strncpy(test_suite.tests[i].description, json_object_get_string(desc_obj_tc), 
                    sizeof(test_suite.tests[i].description) - 1);
        }

        if (json_object_object_get_ex(test_obj, "category", &cat_obj)) {
            strncpy(test_suite.tests[i].category, json_object_get_string(cat_obj), 
                    sizeof(test_suite.tests[i].category) - 1);
        }

        if (json_object_object_get_ex(test_obj, "weight", &weight_obj)) {
            test_suite.tests[i].weight = json_object_get_double(weight_obj);
        } else {
            test_suite.tests[i].weight = 1.0; // Default weight
        }
    }

    // Extract potential edge cases
    json_object *edge_cases_obj;
    if (json_object_object_get_ex(root, "potential_edge_cases", &edge_cases_obj)) {
        int edge_array_len = json_object_array_length(edge_cases_obj);
        test_suite.num_edge_cases = (edge_array_len > MAX_TESTS) ? MAX_TESTS : edge_array_len;
        
        for (int i = 0; i < test_suite.num_edge_cases; i++) {
            json_object *edge_obj = json_object_array_get_idx(edge_cases_obj, i);
            strncpy(test_suite.potential_edge_cases[i], json_object_get_string(edge_obj), 
                    sizeof(test_suite.potential_edge_cases[i]) - 1);
        }
    }

    json_object_put(root);
    free(json_string);
    return 0;
}

/**
 * @brief Enhanced passrate calculation with weighted scoring
 */
float calculate_dynamic_passrate(EnhancedEvalMetrics *metrics) {
    metrics->tests_passed = 0;
    metrics->tests_failed = 0;
    metrics->num_failed_details = 0;
    
    float total_weight = 0.0f;
    float passed_weight = 0.0f;
    
    printf("    Running %d LLM-generated test cases:\n", test_suite.num_tests);
    
    for (int i = 0; i < test_suite.num_tests; i++) {
        char output_buf[MAX_OUTPUT_SIZE] = {0};
        total_weight += test_suite.tests[i].weight;
        
        printf("    Test %d [%s]: %s\n", i + 1, test_suite.tests[i].category, 
               test_suite.tests[i].description);
        
        if (run_test_process(test_suite.tests[i].input, output_buf, sizeof(output_buf)) == 0) {
            trim_trailing_whitespace(output_buf);
            
            if (strcmp(output_buf, test_suite.tests[i].expected_output) == 0) {
                printf("      ✅ PASS\n");
                metrics->tests_passed++;
                passed_weight += test_suite.tests[i].weight;
            } else {
                printf("      ❌ FAIL - Expected: '%s', Got: '%s'\n", 
                       test_suite.tests[i].expected_output, output_buf);
                metrics->tests_failed++;
                
                // Record failure details
                if (metrics->num_failed_details < MAX_TESTS) {
                    snprintf(metrics->failed_tests[metrics->num_failed_details], 512,
                             "Test %d (%s): Expected '%s', Got '%s'", 
                             i + 1, test_suite.tests[i].description,
                             test_suite.tests[i].expected_output, output_buf);
                    metrics->num_failed_details++;
                }
            }
        } else {
            printf("      ❌ FAIL - Timeout or execution error\n");
            metrics->tests_failed++;
            
            if (metrics->num_failed_details < MAX_TESTS) {
                snprintf(metrics->failed_tests[metrics->num_failed_details], 512,
                         "Test %d (%s): Execution timeout or error", 
                         i + 1, test_suite.tests[i].description);
                metrics->num_failed_details++;
            }
        }
    }
    
    // Calculate both simple and weighted scores
    float simple_passrate = (test_suite.num_tests > 0) ? (float)metrics->tests_passed / test_suite.num_tests * 100.0f : 0.0f;
    metrics->weighted_score = (total_weight > 0) ? (passed_weight / total_weight * 100.0f) : 0.0f;
    
    return simple_passrate;
}

/**
 * @brief Prints information about the loaded test suite
 */
void print_test_suite_info(void) {
    printf("📋 Test Suite Information:\n");
    printf("    Program: %s\n", test_suite.program_description);
    printf("    Type: %s\n", test_suite.program_type);
    printf("    Difficulty: %s\n", test_suite.difficulty_level);
    printf("    Tests: %d test cases loaded\n", test_suite.num_tests);
    
    if (test_suite.num_edge_cases > 0) {
        printf("    Edge Cases to Consider:\n");
        for (int i = 0; i < test_suite.num_edge_cases; i++) {
            printf("      • %s\n", test_suite.potential_edge_cases[i]);
        }
    }
    printf("\n");
}

/**
 * @brief Enhanced results output with detailed failure information
 */
void write_enhanced_results_to_json(const EnhancedEvalMetrics *metrics) {
    FILE *f = fopen(RESULTS_JSON_PATH, "w");
    if (!f) {
        perror("fopen (results.json)");
        return;
    }
    
    fprintf(f, "{\n");
    fprintf(f, "  \"program_description\": \"%s\",\n", test_suite.program_description);
    fprintf(f, "  \"program_type\": \"%s\",\n", test_suite.program_type);
    fprintf(f, "  \"difficulty_level\": \"%s\",\n", test_suite.difficulty_level);
    fprintf(f, "  \"passrate\": %.1f,\n", metrics->passrate);
    fprintf(f, "  \"weighted_score\": %.1f,\n", metrics->weighted_score);
    fprintf(f, "  \"memory_score\": %.1f,\n", metrics->memory_score);
    fprintf(f, "  \"robustness_score\": %.1f,\n", metrics->robustness_score);
    fprintf(f, "  \"tests_passed\": %d,\n", metrics->tests_passed);
    fprintf(f, "  \"tests_failed\": %d,\n", metrics->tests_failed);
    fprintf(f, "  \"total_tests\": %d,\n", test_suite.num_tests);
    fprintf(f, "  \"execution_time_ms\": %ld,\n", metrics->execution_time_ms);
    
    // Include failed test details
    fprintf(f, "  \"failed_test_details\": [\n");
    for (int i = 0; i < metrics->num_failed_details; i++) {
        fprintf(f, "    \"%s\"", metrics->failed_tests[i]);
        if (i < metrics->num_failed_details - 1) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "  ],\n");
    
    // Include potential edge cases for further analysis
    fprintf(f, "  \"potential_edge_cases\": [\n");
    for (int i = 0; i < test_suite.num_edge_cases; i++) {
        fprintf(f, "    \"%s\"", test_suite.potential_edge_cases[i]);
        if (i < test_suite.num_edge_cases - 1) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

// --- Main Logic (Modified) ---

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <source.c> <test_cases.json>\n", argv[0]);
        return 1;
    }

    // Set up signal handlers and cleanup routine
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    atexit(cleanup);

    // Load LLM-generated test cases
    printf("🔍 Loading LLM-generated test cases...\n");
    if (load_test_cases_from_json(argv[2]) != 0) {
        fprintf(stderr, "❌ Failed to load test cases from %s\n", argv[2]);
        return 1;
    }
    
    print_test_suite_info();

    // Create secure temporary directory
    char temp_dir_template[] = "/tmp/safe_eval_XXXXXX";
    if (mkdtemp(temp_dir_template) == NULL) {
        perror("mkdtemp failed");
        return 1;
    }
    snprintf(temp_dir_path, sizeof(temp_dir_path), "%s", temp_dir_template);

    long start_time = current_time_ms();

    printf("1. Compiling source file: %s\n", argv[1]);
    if (compile_source(argv[1]) != 0) {
        fprintf(stderr, "❌ Compilation failed.\n");
        EnhancedEvalMetrics metrics = {0};
        write_enhanced_results_to_json(&metrics);
        return 1;
    }
    printf("    ✅ Compilation successful.\n\n");

    EnhancedEvalMetrics metrics = {0};
    
    printf("2. Running LLM-generated correctness tests...\n");
    metrics.passrate = calculate_dynamic_passrate(&metrics);
    printf("    ✅ Simple Passrate: %.1f%% (%d/%d tests passed)\n", 
           metrics.passrate, metrics.tests_passed, test_suite.num_tests);
    printf("    ✅ Weighted Score: %.1f%%\n\n", metrics.weighted_score);

    printf("3. Analyzing memory usage with Valgrind...\n");
    metrics.memory_score = analyze_memory();
    printf("    ✅ Memory Score: %.1f\n\n", metrics.memory_score);

    printf("4. Checking robustness...\n");
    metrics.robustness_score = check_robustness();
    printf("    ✅ Robustness Score: %.1f\n\n", metrics.robustness_score);

    metrics.execution_time_ms = current_time_ms() - start_time;

    write_enhanced_results_to_json(&metrics);
    printf("🎉 Enhanced evaluation complete. Results written to %s\n", RESULTS_JSON_PATH);
    printf("📊 Ready for Stage 3 analysis...\n");

    return 0;
}

// --- Utility Function Implementations ---

/**
 * @brief Cleans up temporary files and directories.
 */
void cleanup(void) {
    if (strlen(temp_dir_path) > 0) {
        char command[512];
        snprintf(command, sizeof(command), "rm -rf %s", temp_dir_path);
        system(command);
    }
    remove(RESULTS_JSON_PATH);
    remove(VALGRIND_LOG_PATH);
}

/**
 * @brief Handles termination signals to ensure cleanup is called.
 */
void handle_signal(int sig) {
    printf("\nCaught signal %d, cleaning up and exiting.\n", sig);
    exit(1); // atexit will call cleanup
}

/**
 * @brief Gets the current time in milliseconds.
 */
long current_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long)(tv.tv_sec) * 1000 + (long)(tv.tv_usec) / 1000;
}

/**
 * @brief Sets resource limits for the child process.
 */
void set_child_resource_limits(void) {
    struct rlimit mem_limit;
    mem_limit.rlim_cur = MEMORY_LIMIT_MB * 1024 * 1024;
    mem_limit.rlim_max = MEMORY_LIMIT_MB * 1024 * 1024;
    if (setrlimit(RLIMIT_AS, &mem_limit) != 0) {
        perror("setrlimit(RLIMIT_AS) failed");
    }

    struct rlimit cpu_limit;
    cpu_limit.rlim_cur = CPU_TIME_LIMIT_S;
    cpu_limit.rlim_max = CPU_TIME_LIMIT_S;
    if (setrlimit(RLIMIT_CPU, &cpu_limit) != 0) {
        perror("setrlimit(RLIMIT_CPU) failed");
    }
}

/**
 * @brief Compiles the given C source file into the temp directory.
 * @return 0 on success, -1 on failure.
 */
int compile_source(const char *source_filename) {
    snprintf(executable_path, sizeof(executable_path), "%s/user_program", temp_dir_path);
    
    char command[1024];
    snprintf(command, sizeof(command), "gcc -o %s %s -lm", executable_path, source_filename);

    int ret = system(command);
    return (WIFEXITED(ret) && WEXITSTATUS(ret) == 0) ? 0 : -1;
}

/**
 * @brief Runs a single test case in a sandboxed child process.
 * @return 0 on success, -1 on timeout or execution error.
 */
int run_test_process(const char *input, char *output_buffer, size_t buffer_size) {
    int stdin_pipe[2], stdout_pipe[2];
    pid_t pid;

    if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1) {
        perror("pipe failed");
        return -1;
    }

    pid = fork();
    if (pid == -1) {
        perror("fork failed");
        return -1;
    }

    if (pid == 0) { // Child process
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stdout_pipe[1], STDERR_FILENO); // Redirect stderr to stdout pipe
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        set_child_resource_limits();
        
        execl(executable_path, executable_path, (char *)NULL);
        // If execl returns, it must have failed
        perror("execl failed");
        exit(EXEC_FAILURE_EXIT_CODE);
    } else { // Parent process
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        // Write input to child's stdin
        write(stdin_pipe[1], input, strlen(input));
        close(stdin_pipe[1]);

        long start = current_time_ms();
        int status;
        ssize_t bytes_read = 0;

        // Non-blocking wait with timeout
        while (current_time_ms() - start < TIMEOUT_SECONDS * 1000) {
            if (waitpid(pid, &status, WNOHANG) == pid) {
                // Child terminated
                if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                    bytes_read = read(stdout_pipe[0], output_buffer, buffer_size - 1);
                    if(bytes_read >= 0) output_buffer[bytes_read] = '\0';
                    close(stdout_pipe[0]);
                    return 0;
                }
                close(stdout_pipe[0]);
                return -1; // Child crashed or exited with error
            }
            usleep(10000); // Sleep for 10ms
        }

        // Timeout occurred
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        close(stdout_pipe[0]);
        return -1;
    }
}

/**
 * @brief Analyzes memory usage by running the program with Valgrind.
 * @return A score from 0 to 100.
 */
float analyze_memory(void) {
    if (test_suite.num_tests == 0) return 100.0f;

    char command[1024];
    // Use the first test case for memory analysis
    snprintf(command, sizeof(command), "echo \"%s\" | valgrind --tool=memcheck --leak-check=full --log-file=%s %s",
             test_suite.tests[0].input, VALGRIND_LOG_PATH, executable_path);
    
    system(command);

    FILE *log_file = fopen(VALGRIND_LOG_PATH, "r");
    if (!log_file) {
        fprintf(stderr, "Could not open valgrind log file.\n");
        return 0.0f;
    }

    char line[512];
    int definitely_lost = 0;
    while (fgets(line, sizeof(line), log_file)) {
        if (strstr(line, "definitely lost:")) {
            sscanf(line, "==%*d== definitely lost: %d bytes in %*d blocks", &definitely_lost);
            break;
        }
    }
    fclose(log_file);
    remove(VALGRIND_LOG_PATH);

    if (definitely_lost == 0) {
        return 100.0f;
    } else if (definitely_lost < 100) {
        return 75.0f;
    } else if (definitely_lost < 1024) {
        return 25.0f;
    }
    return 0.0f;
}

/**
 * @brief Checks if the program handles signals gracefully.
 * @return A score from 0 to 100.
 */
float check_robustness(void) {
    pid_t pid = fork();
    if (pid == -1) {
        perror("fork for robustness check failed");
        return 0.0f;
    }

    if (pid == 0) { // Child process
        // Run the program with no input, it should just wait or exit
        execl(executable_path, executable_path, (char *)NULL);
        exit(EXEC_FAILURE_EXIT_CODE);
    } else { // Parent process
        int status;
        usleep(ROBUSTNESS_SIGINT_WAIT_US); // Wait a bit before sending signal

        kill(pid, SIGINT); // Send interrupt signal

        // Check if the process terminates quickly after the signal
        long start = current_time_ms();
        while (current_time_ms() - start < 1000) { // 1 second timeout for graceful exit
            if (waitpid(pid, &status, WNOHANG) == pid) {
                return 100.0f; // Terminated gracefully
            }
            usleep(10000);
        }

        // If it's still running, it didn't handle the signal well
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        return 0.0f; // Hung or did not terminate
    }
}

/**
 * @brief Removes trailing whitespace (space, tab, newline) from a string.
 */
void trim_trailing_whitespace(char *str) {
    if (str == NULL) return;
    int len = strlen(str);
    while (len > 0 && isspace((unsigned char)str[len - 1])) {
        len--;
    }
    str[len] = '\0';
}
