#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "110503517_assignment_1_update.h"



//NOTE: L1 is the input layer , L2 is the hidden layer , L3 is the output layer
int check_code = 0; //use for loop status
char *test_input; //space for user iniput storage
double *train_input; //input is first layer of training
double *train_answr_key;//storage for train solution
double *L2; //space for second layer results
double *L3; //space for third layer results
double *sigL2; //space for second layer results
double *sigL3; //space for third layer results
double *L1_to_L2_weights; //input weights
double *L2_to_L3_weights; //weights from first hidden layer (second overall) to next hidden layer
char *test_ptr = NULL; //pointer to test set
double *for_feed_forward; //to store the 2 bit that are going to be process
double *input_ptr = NULL; //pointer to set in input or test array
double *show_ptr = NULL; //pointer to show the loss
double xor_result = 0; //variable for two bit result
double loss = 0;  // the total loss (sum of the four testing sets' absolute errors )
double learning_rate = 0.8; //learning rate

//vars used temporarily during backpropagation
double *L3_der_err_der_y; //output value derivatives
double *L3_der_err_der_x; //derivatives of the input to the final layer
double *L2_der_err_der_w; //derivatives of the weights from L2 to L3
double *L2_suggested_weight_changes; // space for L2 suggested weight changes
double *L2_der_err_der_y; //derivative of the output of second layer
double *L2_der_err_der_x; //derivative of the input to the second layer
double *L1_der_err_der_w; //derivatives of the weights from L1 to L2
double *L1_suggested_weight_changes;// space for L1 suggested weight changes



int main(void) 
{
    //create space for Dynamic memories
    test_input = malloc(100 * sizeof(char)); //space for test_input pointer
    L2 = malloc(SIZE_L2 * sizeof(double));
    L3 = malloc(SIZE_L3 * sizeof(double));
    sigL2 = malloc(SIZE_L2 * sizeof(double));
    sigL3 = malloc(SIZE_L3 * sizeof(double));
    L1_to_L2_weights = malloc(L1_WEIGHTS * sizeof(double));
    L2_to_L3_weights = malloc(L2_WEIGHTS * sizeof(double));
    for_feed_forward = malloc(SIZE_L1 * sizeof(double));
    L3_der_err_der_y = malloc(SIZE_L3 * sizeof(double));
    L3_der_err_der_x = malloc(SIZE_L3 * sizeof(double));
    L2_der_err_der_w = malloc(L2_WEIGHTS * sizeof(double));
    L2_suggested_weight_changes = malloc(L2_WEIGHTS * sizeof(double));
    L2_der_err_der_y = malloc(L2_WEIGHTS * sizeof(double));
    L2_der_err_der_x = malloc(L2_WEIGHTS * sizeof(double));
    L1_der_err_der_w = malloc(L1_WEIGHTS * sizeof(double));
    L1_suggested_weight_changes = malloc(L1_WEIGHTS * sizeof(double));
    train_input = malloc(TRAIN_INPUT_NUM * sizeof(double));
    train_answr_key = malloc(TRAIN_ANSWER_NUM * sizeof(double));

    //initialize train input and train answer key
    for(int count = 0; count < TRAIN_INPUT_NUM; count++)
    {
        switch(count)
        {
            case 0:
            *(train_input+count) = 0;
            *(train_input+count+1) = 0;
            break;

            case 2:
            *(train_input+count) = 0;
            *(train_input+count+1) = 1;
            break;

            case 4:
            *(train_input+count) = 1;
            *(train_input+count+1) = 0;
            break;

            case 6:
            *(train_input+count) = 1;
            *(train_input+count+1) = 1;
            break;

        }
    }

    for(int count = 0; count < TRAIN_ANSWER_NUM; count++)
    {
        switch(count)
        {
            case 0:
            *(train_answr_key+count) = 0;
            break;

            case 1:
             *(train_answr_key+count) = 1;
            break;

            case 2:
             *(train_answr_key+count) = 1;
            break;

            case 3:
             *(train_answr_key+count) = 0;
            break;

        }
    }   

    srand((unsigned) 0); //seed with 0 for consistency
    //train session
    fill_hyperparams_with_rand();
    backprop();
    printf("-------NN training complete with %d epochs------- \n\n",TOTAL_EPOCHS);
    //train result and user input
    show_train_result();
    user_input();

    return 0;
}

void user_input(void) //to test the NN after learning
{ 

    //input number
    while(check_code == 0) //check code 0 could enter loop
    {
       
        printf("Please enter testing set with binary numbers(2-100bit):");
        
        scanf("%s",test_input); //user input test set
        if (*(test_input + 0) == '$')
        {
            break; //type $ to exit
        }
        if(strlen(test_input)<INPUT_SIZE_MIN) //if input < 2 bit, input again
        {
            printf("Input error,try again\n\n");
            continue;
        }
        for(int bit = 0; bit<INPUT_SIZE_MAX;bit++) //to check if the string is binary number
        {
            if (*(test_input + bit) == '\0') //if end, stop checking
            {
                check_code = 0;
                break;
            }
            else if(*(test_input + bit) != '1' && *(test_input + bit)!= '0') 
            {
                check_code = 1; //code 1 for input again
                printf("Input error,try again\n\n");
                break;
            }
            
        }
        if(check_code == 1)
        {
            check_code = 0;
            continue; //back to begining of loop,input again
        }
    
        test_ptr = test_input ;

        //feed in the first two bit first
        for(int count = 0;count<TRAIN_SHIFT_UNIT;count++)
        {   // store the first to bit (after changing into double)
            *(for_feed_forward+count) = (double)(*(test_ptr+count)-48); //change char to double
        }
        input_ptr = for_feed_forward;
        feed_forward(input_ptr); //feed the input to the network and run through
        xor_result = round(*sigL3+0); //store the result inorder to feed it with the next bit later
        reset_nn(); //reset neurons for next feed
        test_ptr += TRAIN_SHIFT_UNIT; //move the pointer to the 3rd bit
    
        while(*test_ptr != '\0')
        {   //now we use the xor result from the previous and feed with the next bit until the end
            *(for_feed_forward+0) = xor_result;
            *(for_feed_forward+1) = (double)(*(test_ptr)-48);
            input_ptr = for_feed_forward;
            feed_forward(input_ptr);
            xor_result = round((*sigL3+0));
            reset_nn();
            test_ptr += TEST_SHIFT_UNIT; //shift 1 bit (to the new bit that will be feed with the xor result)
        }
        printf("The output is: %1.0lf \n",xor_result); //print the final result
        printf("type $ if you want to exit the program\n\n"); //the way to stop entering and exit
        
        
    }
    
}

void reset_nn(void) //reset_neurons
{
    for(int neurons_num = 0; neurons_num < L1_WEIGHTS; neurons_num++)
    {
        if(neurons_num < L2_WEIGHTS)
        {
            *(L2+neurons_num) = CLEAR;
        }
    }
    *(L3+0) = CLEAR;
}

void backprop(void) //the backpropagration process
{ 
    for (int epochs = 1; epochs <= TOTAL_EPOCHS; epochs++)
    {
        input_ptr = train_input; // set the pointer to rhe start of train input

        for (int train_number = 0; train_number < SIZE_OF_TRAIN; train_number++)
        {
            feed_forward(input_ptr); //run through the network

            //find derivatives of final output,using loss function (Y_output - Y_expected)
            *(L3_der_err_der_y+0) = *(sigL3+0) - *(train_answr_key+train_number);

            //find derivatives of the input to the third layer
            *(L3_der_err_der_x+0) = dsigmoid(*(sigL3+0)) * (*(L3_der_err_der_y+0)); 

            //derivative of weights from L2 to L3
            for (int count = 0; count < SIZE_L2; count++)
            {
                *(L2_der_err_der_w+count) = (*(sigL2+count)) * (*(L3_der_err_der_x+0));
            }

            //after knowing the derivative of weights from L2 to L3 we can copute the weight change of L2
            L2_compute_weight_change(); 

            //derivative of the output of second layer
            for (int count = 0; count < SIZE_L2; count++)
            {
                *(L2_der_err_der_y+count) = (*(L2_to_L3_weights+count)) * (*(L3_der_err_der_x+0));
            }
                

            //derivative of the input to the second layer
            for (int count = 0; count < SIZE_L2; count++)
            {
                 *(L2_der_err_der_x+count) = dsigmoid(*(sigL2+count)) * (*(L2_der_err_der_y+count));
            }
               

            //derivative of the weights from L1 to L2
            for (int num_L1 = 0; num_L1 < SIZE_L1; num_L1++)
            {
                 for (int num_L2 = 0; num_L2 < SIZE_L2; num_L2++)
                 {
                    *(L1_der_err_der_w+(num_L1 * SIZE_L2 + num_L2)) = (*(input_ptr+num_L1)) * (*(L2_der_err_der_x+num_L2));
                 }
            }
               
            //after knowing the derivative of weights from L2 to L3 we can copute the weight change of L2
            L1_compute_weight_change();

            input_ptr += TRAIN_SHIFT_UNIT; //move the pointer to the next set
            
            clear_vars(); //clear the variables that need to use in the next epoch
        }

        //update the weights after 1 epochs of learning
        weight_update();
        
        // show the total loss value (sum of the absolute error of all the test sets) after a certain epochs
        if(epochs % PLOT_EPOCHS_UNIT == 0)
        {
            show_loss(epochs);
        }
    }

}

void clear_vars(void) //function to clear the variables for the next epcho 
{
    for (int count = 0; count < L1_WEIGHTS; count++)
    {
        *(L1_der_err_der_w+count) = CLEAR;

        if (count < L2_WEIGHTS)
        {
            *(L2_der_err_der_y+count) = CLEAR;
            *(L2_der_err_der_x+count) = CLEAR;
            *(L2_der_err_der_w+count) = CLEAR;
        }
    }
    *(L3_der_err_der_y+0) = CLEAR;
    *(L3_der_err_der_x+0) = CLEAR;

    reset_nn();//to clear the neurons
}

void L1_compute_weight_change(void) //function to compute the weight change for each weight of L1
{
    for (int L1_weight_num = 0;  L1_weight_num< L1_WEIGHTS; L1_weight_num++)
    {
        //the (Derivative * learning rate) part from the delta rule
        *(L1_suggested_weight_changes+L1_weight_num) +=  learning_rate * (*(L1_der_err_der_w+L1_weight_num));
    }
}

void L2_compute_weight_change(void)//function to compute the weight change for each weight of L1
{
    for (int L2_weight_num = 0; L2_weight_num < L2_WEIGHTS; L2_weight_num++)
    {
        //the (Derivative * learning rate) part from the delta rule
        *(L2_suggested_weight_changes+L2_weight_num) += learning_rate * (*(L2_der_err_der_w+L2_weight_num));
    }      
}
void weight_update(void) //complete the delta rule (New weight = old weight — Derivative * learning rate)
{
    for (int weight_num = 0; weight_num < L1_WEIGHTS; weight_num++) 
    {
        *(L1_to_L2_weights+weight_num) -= *(L1_suggested_weight_changes+weight_num);
        *(L1_suggested_weight_changes+weight_num) = CLEAR; //clear for the next epcho

        if (weight_num < L2_WEIGHTS) 
        {
            *(L2_to_L3_weights+weight_num) -= *(L2_suggested_weight_changes+weight_num);
            *(L2_suggested_weight_changes+weight_num) = 0; //clear for the next epcho
        }
    }
}

double sigmoid(double num) //use sigmoid as activation function 
{ 
    return 1 / (1 + exp(-1.0 * num));
}

double dsigmoid(double num)//num*(1-num)) is derivative of the sigmoid
{
    return num * (1 - num);
}
   
void feed_forward(double *feed_ptr) //matrix multiplication (run through the network)
{ 

    for (int L1_neuron_num = 0; L1_neuron_num < SIZE_L1; L1_neuron_num++) //process from L1(input layer) to L2
    {

        for (int L2_neuron_num = 0; L2_neuron_num  < SIZE_L2; L2_neuron_num++)
        {
            *(L2+L1_neuron_num) += (*(L1_to_L2_weights+((L2_neuron_num * SIZE_L2) + L1_neuron_num))) * (*(feed_ptr+L2_neuron_num));
        }
            
        *(sigL2+L1_neuron_num) = sigmoid(*(L2+L1_neuron_num));
    }

    for (int neuron_num = 0; neuron_num < SIZE_L2; neuron_num ++) // process from L2 to L3(output layer)
    {
        *(L3+0) += (*(L2_to_L3_weights+neuron_num)) * (*(sigL2+neuron_num));
    }
        
    *(sigL3+0) = sigmoid(*(L3+0));
}

double init_weights(void) //generate a random double from 0 to 1
{
    return ((double) rand() / (double)RAND_MAX);
}

void fill_hyperparams_with_rand(void) //fill initial weights at the begining
{
    for (int num_weights = 0; num_weights < L1_WEIGHTS; num_weights++)
    {
        *(L1_to_L2_weights+num_weights) = init_weights();
    }
        
    for (int num_weights = 0; num_weights < L2_WEIGHTS; num_weights++)
    {
        *(L2_to_L3_weights+num_weights) = init_weights();
    }
        
}

void show_loss(int num_epochs) //function to show the loss at a certain epoch
{
    show_ptr = train_input; //set the pointer to test input
    for(int count = 0; count < SIZE_OF_TRAIN; count++ )
    {
        feed_forward(show_ptr); //run through the network
        loss += (fabs( (*(sigL3+0)) - (*(train_answr_key+count)) )); //add up the absolute error (loss value) of each set
        reset_nn(); //reset neurons for next set
        show_ptr += TRAIN_SHIFT_UNIT; //shift the pointer to the next set
    }
    printf("at %6d epochs the total loss is %lf \n",num_epochs,loss); //print out the results
    loss = CLEAR; //clear the variable for next time of compute
}

void show_train_result(void) //show output and tje error
{
    input_ptr = train_input;

    for (int train_num = 0; train_num < SIZE_OF_TRAIN; train_num++)
    {
        feed_forward(input_ptr);
        printf("for train input [%1.0lf,%1.0lf]the  compute output is: %1.0lf\n",*input_ptr,*(input_ptr+1),round(*(sigL3+0)));
        printf("the abosolute error of this set is %lf\n\n",fabs( (*(sigL3+0)) - (*(train_answr_key+train_num)) ));
        reset_nn();
        input_ptr += TRAIN_SHIFT_UNIT;
    }
}



