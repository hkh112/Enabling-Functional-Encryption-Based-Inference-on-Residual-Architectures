#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include "rlwe_sife.h"

#define SEC_LEVEL 1
#define UNKNOWN                16
// #define PERF 1
#define DEC_GPU 16 // 4,16

uint64_t CLOCK1, CLOCK2;
uint64_t SUM_POLY_TIME = 0;

// void polynomial(uint32_t terms[TERMS][2], double number) {
//         int integ = (int)number;
//         double decim = number - integ;

//     if (number >= 0) {
//                 terms[0][0] = integ;
//                 terms[0][1] = 0;
//         } else {
//                 terms[0][0] = 0;
//                 terms[0][1] = integ * -1;
//         }

//     for (int t = 1; t < TERMS; t++) {
//         decim = (decim - (int)decim) * UNKNOWN;

//         if (number >= 0) {
//             terms[t][0] = (int)fabs(decim);
//             terms[t][1] = 0;
//         } else {
//             terms[t][0] = 0;
//             terms[t][1] = (int)fabs(decim);
//         }
//     }
// }

void polynomial(uint32_t terms[TERMS][2], double number) {
        #ifdef PERF
                uint64_t start, end;
                start = cpucycles();
        #endif
    double current_val = fabs(number);
    int is_negative = (number < 0);

    for (int t = 0; t < TERMS; t++) {
        current_val *= UNKNOWN;

        uint32_t digit = (uint32_t)current_val;

        if (!is_negative) {
            terms[t][0] = digit;
            terms[t][1] = 0;
        } else {
            terms[t][0] = 0;
            terms[t][1] = digit;
        }

        current_val -= (double)digit;

        if (current_val < 0) current_val = 0;
    }
    #ifdef PERF
        end = cpucycles();
        SUM_POLY_TIME += (end - start);
    #endif
}


void makeKeys(uint32_t* mpk, uint32_t* msk) {
        uint32_t (*mpk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])mpk;
        uint32_t (*msk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])msk;
        rlwe_sife_setup(mpk_t, msk_t);
}

void loadSecInput1x1(uint32_t* encryptedImage, double* image, int* imageSize, int stride, uint32_t* mpk) {
        
        #ifdef PERF
                uint64_t CLOCK1, CLOCK2, AVG_CLOCK;
                AVG_CLOCK = 0;
        #endif
        
        uint32_t (*mpk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])mpk;

        int inputBatch = imageSize[0];
        int inputChannel = imageSize[1];
        int inputWidth = imageSize[2];
        int inputHeight = imageSize[3];

        int channelSplit = ceil((double)inputChannel / (double)SIFE_L);
        printf("channelSplit: %d / inputChannel: %d / sifel: %d\n", channelSplit, inputChannel, SIFE_L);
        
        int outputWidth = floor((inputWidth - 1) / stride) + 1;
        int outputHeight = floor((inputHeight - 1) / stride) + 1;

        printf("outputWidth: %d / outputHeight: %d / inputBatch: %d\n", outputWidth, outputHeight, inputBatch);

        uint32_t m[TERMS][2][SIFE_L] = {0};

        uint32_t (*encryptedImage_t)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])encryptedImage;

        int input_W_dot_H = inputWidth * inputHeight;

        uint32_t polyInput[TERMS][2] = {0};

        for (int b = 0; b < inputBatch; b++) {
                for (int h = 0; h < outputHeight; h++) {
                        for (int w = 0; w < outputWidth; w++) {
                                for (int cs = 0; cs < channelSplit; cs++) {
                                        
                                        memset(m, 0, sizeof(m)); // (채널 스플릿 수정 1)

                                        int channels_in_this_split = (cs == channelSplit - 1) ? (inputChannel - cs * SIFE_L) : SIFE_L; // (채널 스플릿 수정 2)

                                        for (int ich = 0; ich < channels_in_this_split; ich++) { // (채널 스플릿 수정 3)
                                                polynomial(polyInput, image[b * inputChannel * input_W_dot_H + (cs * SIFE_L + ich) * input_W_dot_H + inputWidth * h * stride + w * stride]);

                                                for (int poly = 0; poly < TERMS; poly++) {
                                                        for (int s = 0; s < 2; s++) {
                                                                m[poly][s][ich] = polyInput[poly][s];
                                                        }
                                                }
                                        }
                                        #ifdef PERF
                                                CLOCK1 = cpucycles();
                                        #endif
                                        rlwe_sife_encrypt_gui((uint32_t*)m, mpk_t, (uint32_t*)encryptedImage_t[b][h][w][cs], 2*TERMS);
                                        #ifdef PERF
                                                CLOCK2 = cpucycles();
                                                AVG_CLOCK += (CLOCK2 - CLOCK1);
                                        #endif
                                };
                        }
                }
        }
        #ifdef PERF
                printf("avg encrypt_gui: %ld\n", AVG_CLOCK / (inputBatch * outputHeight * outputWidth * channelSplit));
        #endif
        // printf("loadsecinput1x1 ended!\n");
}


void convolution1x1_dec16(double* output, uint32_t* secImage, int* imageSize, double* filter, int* filterSize, int stride, uint32_t* msk) {

        #ifdef PERF
                uint64_t AVG_CLOCK_KEYGEN = 0;
                uint64_t AVG_CLOCK_DEC = 0;
                uint64_t AVG_CLOCK_EXT = 0;
        #endif

        uint64_t TOTAL_CLOCK_PIXEL = 0;

	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

	int channelSplit = ceil((double)inputChannel / (double)SIFE_L);
	int splitted = SIFE_L;

	int filterCount = filterSize[0];
	int filterLength = filterSize[1];

	int outputWidth = floor((inputWidth - 1) / stride) + 1;
	int outputHeight = floor((inputHeight - 1) / stride) + 1;

	uint32_t (*msk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])msk;
	uint32_t (*secImage_t)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])secImage;

	for (int b = 0; b < inputBatch; b++) {
                #pragma omp parallel for schedule(dynamic)
		for (int fc = 0; fc < filterCount; fc++) {
                        #ifdef PERF
                                uint64_t CLOCK_KEYGEN_1, CLOCK_KEYGEN_2;
                                uint64_t CLOCK_DEC_1, CLOCK_DEC_2;
                                uint64_t CLOCK_EXT_1, CLOCK_EXT_2;
                        #endif
                        uint32_t dy2[SIFE_NMODULI][SIFE_N];
                        uint32_t* d_y = (uint32_t*)malloc(TERMS*2*TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
                        uint32_t y[TERMS][2][SIFE_L] = {0};
                        uint32_t* slicedFilter = (uint32_t*)malloc(1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));
                        uint32_t sk_y[TERMS][2][SIFE_NMODULI][SIFE_N] = {0};
                        uint32_t polyFilter[TERMS][2] = {0};

                        if (omp_get_thread_num() == 0) {
                                printf("batch:%d/%d filter:%d/%d (Thread 0)  \r", b+1, inputBatch, fc+1, filterCount);
                                fflush(stdout);
                        }

			for (int j = 0; j < outputHeight; j++) {
				for (int i = 0; i < outputWidth; i++) {
                                        uint64_t START_PIXEL_CLOCK = cpucycles();
                    
					double outPix = 0;

					for (int cs = 0; cs < channelSplit; cs++) {
						int channels_in_this_split = (cs == channelSplit - 1) ? (inputChannel - cs * SIFE_L) : SIFE_L;

						memset(slicedFilter, 0, 1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));
						memset(y, 0, sizeof(y));

						for (int ich = 0; ich < channels_in_this_split; ich++) {
							polynomial(polyFilter, filter[fc * filterLength + (cs * SIFE_L + ich)]);
							for (int poly = 0; poly < TERMS; poly++) {
								slicedFilter[poly * splitted + ich] = polyFilter[poly][0];
								slicedFilter[(poly + TERMS) * splitted + ich] = polyFilter[poly][1];
							}
						}

						for (int ft = 0; ft < TERMS * 2; ft++) {
							int fs = (int)floor((double)ft/(double)TERMS);
							for (int ich = 0; ich < splitted; ich++) {
								y[ft%TERMS][fs][ich] = slicedFilter[ft * splitted + ich];
							}
						}

                                                #ifdef PERF
                                                        CLOCK_KEYGEN_1 = cpucycles();
                                                #endif
						rlwe_sife_keygen_gui((uint32_t*)y, msk_t, (uint32_t*)sk_y, TERMS*2);
                                                #ifdef PERF
                                                        CLOCK_KEYGEN_2 = cpucycles();
                                                        #pragma omp atomic
                                                        AVG_CLOCK_KEYGEN += (CLOCK_KEYGEN_2 - CLOCK_KEYGEN_1);
                                                #endif
						
						memset(d_y, 0, (size_t)TERMS*2*TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
                                                #ifdef PERF
                                                        CLOCK_DEC_1 = cpucycles();
                                                #endif
						rlwe_sife_decrypt_gmp_gui3_x16((uint32_t*)secImage_t[b][j][i][cs], (uint32_t*)y, (uint32_t*)sk_y, (uint32_t*)d_y, TERMS*2, TERMS*2);
                                                #ifdef PERF
                                                        CLOCK_DEC_2 = cpucycles();
                                                        #pragma omp atomic
                                                        AVG_CLOCK_DEC += (CLOCK_DEC_2 - CLOCK_DEC_1);
                                                #endif

						for (int ft = 0; ft < TERMS; ft++) {
							for (int fs = 0; fs < 2; fs++) {
								for (int it = 0; it < TERMS; it++) {
									for (int is = 0; is < 2; is++) {
                                        memcpy(dy2, d_y + (it * 2 * TERMS * 2 + is * TERMS * 2 + ft * 2 + fs) * SIFE_NMODULI * SIFE_N, SIFE_NMODULI * SIFE_N * sizeof(uint32_t));
                                                #ifdef PERF
                                                        CLOCK_EXT_1 = cpucycles();
                                                #endif
                                        double extracted_val = round_extract_gmp2(dy2);
                                                #ifdef PERF
                                                        CLOCK_EXT_2 = cpucycles();
                                                        #pragma omp atomic
                                                        AVG_CLOCK_EXT += (CLOCK_EXT_2 - CLOCK_EXT_1);
                                                #endif
                                        int sign = (fs == is) ? 1 : -1;
                                        if (extracted_val != 50241) {
                                            double scale = pow(UNKNOWN, (double)(it + ft + 2));
                                            outPix += (extracted_val / scale) * sign;
                                        }
									}
								}
							}
						}
					}
					int idx = b * (filterCount * outputHeight * outputWidth) + fc * (outputHeight * outputWidth) + j * outputWidth + i;
					output[idx] = outPix;
                                        TOTAL_CLOCK_PIXEL += (cpucycles() - START_PIXEL_CLOCK);
				}
			}
            free(slicedFilter);
            free(d_y);
		}
	}
    printf("\n");

        #ifdef PERF
                printf("Clocks\n");
                printf("Keygen: %ld\n", AVG_CLOCK_KEYGEN / (inputBatch * filterCount * outputHeight * outputWidth * channelSplit));
                printf("Decryption: %ld\n", AVG_CLOCK_DEC / (inputBatch * filterCount * outputHeight * outputWidth * channelSplit));
                printf("Extract: %ld\n", AVG_CLOCK_EXT / (inputBatch * filterCount * outputHeight * outputWidth * channelSplit * TERMS * 2 * 2 * TERMS));
        #endif

        printf("Pixel Avg Clocks\n");
        printf("Total Pixel Avg Clocks: %ld\n", TOTAL_CLOCK_PIXEL / (inputBatch * outputHeight * outputWidth * filterCount));
}
void convolution1x1_dec4(double* output, uint32_t* secImage, int* imageSize, double* filter, int* filterSize, int stride, uint32_t* msk) {

	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

	int channelSplit = ceil((double)inputChannel / (double)SIFE_L);
	int splitted = SIFE_L;

	int filterCount = filterSize[0];
	int filterLength = filterSize[1];

	int outputWidth = floor((inputWidth - 1) / stride) + 1;
	int outputHeight = floor((inputHeight - 1) / stride) + 1;

	uint32_t (*msk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])msk;
	uint32_t (*secImage_t)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])secImage;

	for (int b = 0; b < inputBatch; b++) {
        #pragma omp parallel for schedule(dynamic)
		for (int fc = 0; fc < filterCount; fc++) {
            uint32_t dy2[SIFE_NMODULI][SIFE_N];
            uint32_t* d_y = (uint32_t*)malloc(TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
            double term[TERMS*TERMS][4] = {0};
            uint32_t y[TERMS][2][SIFE_L] = {0};
            uint32_t* slicedFilter = (uint32_t*)malloc(1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));
            uint32_t sk_y[TERMS][2][SIFE_NMODULI][SIFE_N] = {0};
            uint32_t polyFilter[TERMS][2] = {0};

			for (int j = 0; j < outputHeight; j++) {
				for (int i = 0; i < outputWidth; i++) {
					
                    double outPix = 0;

                    for (int cs = 0; cs < channelSplit; cs++) {

                        int channels_in_this_split = (cs == channelSplit - 1) ? (inputChannel - cs * SIFE_L) : SIFE_L;

                        memset(slicedFilter, 0, 1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));
                        memset(y, 0, sizeof(y));

                        for (int ich = 0; ich < channels_in_this_split; ich++) {
                            polynomial(polyFilter, filter[fc * filterLength + (cs * SIFE_L + ich)]);
                            for (int poly = 0; poly < TERMS; poly++) {
                                slicedFilter[poly * splitted + ich] = polyFilter[poly][0];
                                slicedFilter[(poly + TERMS) * splitted + ich] = polyFilter[poly][1];
                            }
                        }

                        for (int ft = 0; ft < TERMS * 2; ft++) {
                            int fs = (int)floor((double)ft/(double)TERMS);
                            for (int ich = 0; ich < splitted; ich++) {
                                y[ft%TERMS][fs][ich] = slicedFilter[ft * splitted + ich];
                            }
                        }

                        rlwe_sife_keygen_gui((uint32_t*)y, msk_t, (uint32_t*)sk_y, TERMS*2);

                        for (int ft = 0; ft < TERMS; ft++) {
                            for (int fs = 0; fs < 2; fs++) {
                                memset(d_y, 0, (size_t)TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
                                rlwe_sife_decrypt_gmp_gui3_x4((uint32_t*)secImage_t[b][j][i][cs], (uint32_t*)y[ft][fs], (uint32_t*)sk_y[ft][fs], (uint32_t*)d_y, TERMS*2);
                                
                                for (int it = 0; it < TERMS; it++) {
                                    for (int is = 0; is < 2; is++) {
                                        memcpy(dy2, d_y + (it * 2 + is) * SIFE_NMODULI * SIFE_N, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
                                        term[TERMS*it+ft][2*is+fs] = round_extract_gmp2(dy2);
                                        int sign = 1;
                                        if ((fs == 0 && is == 1) || (fs == 1 && is == 0)) {
                                            sign = -1;
                                        }
                                        if (term[TERMS*it+ft][2*is+fs] != 50241) {
                                            outPix += (term[TERMS*it+ft][2*is+fs]) / pow(UNKNOWN, (ft+it)) * sign;
                                        }
                                    }
                                }
                            }
                        }
                    } 
					int idx = b * (filterCount * outputHeight * outputWidth) + fc * (outputHeight * outputWidth) + j * outputWidth + i;
                    output[idx] = outPix;
				}
			}
            free(slicedFilter);
            free(d_y);
		}
	}
}
void convolution1x1(double* output, uint32_t* secImage, int* imageSize, double* filter, int* filterSize, int stride, uint32_t* msk) {
	#if DEC_GPU == 16
		convolution1x1_dec16(output, secImage, imageSize, filter, filterSize, stride, msk);
	#endif
	#if DEC_GPU == 4
        // (수정됨) 이제 dec4 함수 시그니처가 double*을 받으므로 타입 일치
		convolution1x1_dec4(output, secImage, imageSize, filter, filterSize, stride, msk);
	#endif
}