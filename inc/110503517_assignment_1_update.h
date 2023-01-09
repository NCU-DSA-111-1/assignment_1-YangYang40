#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define SIZE_L1 2 //how many nuerons in L1(input layer)
#define SIZE_L2 2 //how many nuerons in L2(hudden layer)
#define SIZE_L3 1 //how many nuerons in L3(output layer)
#define INPUT_SIZE_MAX 101 //how many test set being input(+1space for'\0')
#define INPUT_SIZE_MIN 2 //minimum bit to be input
#define SIZE_OF_TRAIN 4 //how many train set being input
#define L1_WEIGHTS 4 //how many weights L1 have
#define L2_WEIGHTS 2 //how many weights L2 have
#define TRAIN_INPUT_NUM 8 //how many train inputs
#define TRAIN_ANSWER_NUM 4 // how many train numbers
#define TRAIN_SHIFT_UNIT 2 //how many bits should be process when training
#define TEST_SHIFT_UNIT 1 // shift 1 bit a time when testing
#define CLEAR 0 //to clear the vars
#define TOTAL_EPOCHS 20000 //how many epochs should be processed
#define PLOT_EPOCHS_UNIT 200 //for how many epochs should we compute and show the loss

//functions' prototype
void fill_hyperparams_with_rand(void);
void L1_compute_weight_change(void);
void L2_compute_weight_change(void);
void weight_update(void);
void feed_forward(double *feed_ptr);
double sigmoid(double num);
double dsigmoid(double num);
void backprop(void);
void user_input(void);
void reset_nn(void);
void clear_vars(void);
void show_loss(int num_epochs);
void show_train_result(void);