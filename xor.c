/***************************************************************************
;                Commodore 64 XOR Neural Network
;
;           Copyright (C) 2026 By Ulrik Hørlyk Hjort
;
; Permission is hereby granted, free of charge, to any person obtaining
; a copy of this software and associated documentation files (the
; "Software"), to deal in the Software without restriction, including
; without limitation the rights to use, copy, modify, merge, publish,
; distribute, sublicense, and/or sell copies of the Software, and to
; permit persons to whom the Software is furnished to do so, subject to
; the following conditions:
;
; The above copyright notice and this permission notice shall be
; included in all copies or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
; EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
; MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
; NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
; LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
; OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
; WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
; ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>

/* Network: 2 inputs, 4 hidden, 1 output */
#define INPUTS 2
#define HIDDEN 4
#define SCALE 100

/* XOR training data */
const char train_x[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
const char train_y[4] = {0, 1, 1, 0};

/* Weights */
int w1[INPUTS][HIDDEN];  /* input->hidden */
int w2[HIDDEN];          /* hidden->output */
int b1[HIDDEN];
int b2;

/* Activations */
int h[HIDDEN];
int out;

unsigned long seed = 1;

int rnd(void) {
    seed = seed * 1103515245UL + 12345UL;
    return ((int)(seed >> 16) % 200) - 100;
}

/* Simple activation: if x > 0 return SCALE, else return 0 */
/* This is like a step function but we'll use tanh approximation */
int activate(int x) {
    /* Tanh approximation scaled to 0-SCALE */
    if (x > 200) return SCALE;
    if (x < -200) return 0;
    /* Linear region: 50 + x/4 */
    return 50 + (x / 4);
}

/* Derivative (approximate): constant for middle region */
int deriv(int y) {
    if (y < 10 || y > 90) return 5;
    return 25;
}

void init(void) {
    unsigned char i, j;
    for (i = 0; i < INPUTS; i++)
        for (j = 0; j < HIDDEN; j++)
            w1[i][j] = rnd();
    for (i = 0; i < HIDDEN; i++) {
        w2[i] = rnd();
        b1[i] = rnd();
    }
    b2 = rnd();
}

void forward(const char *x) {
    unsigned char i, j;
    int sum;
    
    /* Hidden layer */
    for (j = 0; j < HIDDEN; j++) {
        sum = b1[j];
        for (i = 0; i < INPUTS; i++)
            sum += x[i] * w1[i][j];
        h[j] = activate(sum);
    }
    
    /* Output */
    sum = b2;
    for (i = 0; i < HIDDEN; i++)
        sum += (h[i] * w2[i]) / SCALE;
    out = activate(sum);
}

void train(void) {
    unsigned int epoch;
    unsigned char s, i, j;
    int target, err;
    int d_out, d_h[HIDDEN];
    int dw;
    
    for (epoch = 0; epoch < 8000; epoch++) {
        for (s = 0; s < 4; s++) {
            /* Forward */
            forward(train_x[s]);
            
            /* Error */
            target = train_y[s] * SCALE;
            err = target - out;
            
            /* Output gradient */
            d_out = (err * deriv(out)) / 100;
            
            /* Hidden gradients */
            for (i = 0; i < HIDDEN; i++)
                d_h[i] = (d_out * w2[i] * deriv(h[i])) / (SCALE * 100);
            
            /* Update output layer - learning rate ~1.0 */
            for (i = 0; i < HIDDEN; i++) {
                dw = (d_out * h[i]) / SCALE;
                w2[i] += dw;
            }
            b2 += d_out / 100;
            
            /* Update hidden layer */
            for (i = 0; i < INPUTS; i++) {
                for (j = 0; j < HIDDEN; j++) {
                    dw = (d_h[j] * train_x[s][i]);
                    w1[i][j] += dw;
                }
            }
            for (j = 0; j < HIDDEN; j++)
                b1[j] += d_h[j] / 100;
        }
        
        if (epoch % 2000 == 0)
            printf("Epoch %u\n", epoch);
    }
}

int main(void) {
    unsigned char i;
    
    printf("XOR Neural Network\n\n");
    
    init();
    
    printf("Before training:\n");
    for (i = 0; i < 4; i++) {
        forward(train_x[i]);
        printf("%d^%d=%d ", train_x[i][0], train_x[i][1], out/50);
    }
    
    printf("\n\nTraining...\n");
    train();
    
    printf("\nAfter training:\n");
    for (i = 0; i < 4; i++) {
        forward(train_x[i]);
        printf("%d XOR %d = %d.%02d\n", 
               train_x[i][0], train_x[i][1],
               out / SCALE, out % SCALE);
    }    
    return 0;
}
