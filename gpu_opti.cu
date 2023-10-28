#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <string>
#include <bit>
#include <bitset>
#include <bits/stdc++.h>

#include <memory>
#include <cstddef>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "prof/image.hh"
#include "prof/image_io.hh"
#include "cuda_runtime.h"

using namespace std;
namespace fs = std::filesystem;

__global__ void color_similarity_measures_kernel(uint8_t* bg_pixels, uint8_t* curr_pixels, uint8_t* result_pixels, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        uint8_t bg = bg_pixels[i];
        uint8_t curr = curr_pixels[i];
        float similarity = fminf(bg, curr) / fmaxf(bg, curr);
        //printf("background: %d, current: %d, similarity: %f\n",bg,curr,similarity);
        result_pixels[i] = similarity * 100.0f;
    }

}

void color_similarity_measures(uint8_t* backGround, uint8_t* currentImage, uint8_t* pipelinePixels, int length) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

    color_similarity_measures_kernel<<<blocksPerGrid, threadsPerBlock>>>(backGround,currentImage, pipelinePixels, length);

}

__global__ void calculate_lbp_kernel(uint8_t* gray_pixels, uint8_t* lbp_pixels, int sx, int sy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i % sx;
    int y = i / sx;

    if (x > 0 && y > 0 && x < sx - 1 && y < sy - 1) {
        uint8_t center = gray_pixels[i];

        uint8_t lbpCode = 0;
        lbpCode |= (gray_pixels[(y - 1) * sx + (x - 1)] >= center) << 7;
        lbpCode |= (gray_pixels[(y - 1) * sx + x] >= center) << 6;
        lbpCode |= (gray_pixels[(y - 1) * sx + (x + 1)] >= center) << 5;
        lbpCode |= (gray_pixels[y * sx + (x + 1)] >= center) << 4;
        lbpCode |= (gray_pixels[(y + 1) * sx + (x + 1)] >= center) << 3;
        lbpCode |= (gray_pixels[(y + 1) * sx + x] >= center) << 2;
        lbpCode |= (gray_pixels[(y + 1) * sx + (x - 1)] >= center) << 1;
        lbpCode |= (gray_pixels[y * sx + (x - 1)] >= center) << 0;

        lbp_pixels[i] = lbpCode;
    }
}

void calculate_lbp(uint8_t* grayImage, uint8_t* pipelinePixels, int* dimensions) {

    int length = dimensions[0]*dimensions[1];
    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

    calculate_lbp_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage, pipelinePixels, dimensions[0], dimensions[1]);

}

__global__ void texture_similarity_measures_kernel(uint8_t* bg_pixels, uint8_t* curr_pixels, uint8_t* result_pixels, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        uint8_t bg = bg_pixels[i];
        uint8_t curr = curr_pixels[i];
        uint8_t sim = bg ^ curr;
        int pop_count = __popc(sim);
        float res = (8.0f - static_cast<float>(pop_count)) / 8.0f;
        result_pixels[i] = res * 100;
    }
}

void texture_similarity_measures(uint8_t* backGround, uint8_t* currentImage, uint8_t* pipelinePixels, int length) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

    texture_similarity_measures_kernel<<<blocksPerGrid, threadsPerBlock>>>(backGround, currentImage, pipelinePixels, length);

   }


__device__ void sort3(uint8_t& a, uint8_t& b, uint8_t& c) {
    if (a > b) {
        uint8_t temp = a;
        a = b;
        b = temp;
    }
    if (b > c) {
        uint8_t temp = b;
        b = c;
        c = temp;
    }
    if (a > b) {
        uint8_t temp = a;
        a = b;
        b = temp;
    }
}

__global__ void choquet_kernel(uint8_t* text_sim_pixels, uint8_t* color_sim_pixels, uint8_t* result_pixels, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        uint8_t b = color_sim_pixels[i * 3 + 2];
        uint8_t g = color_sim_pixels[i * 3 + 1];
        b = (b + g) / 2.0f;
        uint8_t texture_pix = text_sim_pixels[i];

        uint8_t similarity[3] = { b, g, static_cast<uint8_t>(texture_pix) };

        //printf("BEFORE, b: %d, g: %d, texture: %d\n",similarity[0],similarity[1],similarity[2]);
        sort3(similarity[0], similarity[1],similarity[2]);
        

        float weights[3] = { 0.1f, 0.3f, 0.6f };
        float sum = 0.0f;
        sum += weights[0] * similarity[0] + weights[1] * similarity[1] + weights[2] * similarity[2];
        //printf("%f\n",sum);
        //printf("AFTER b: %d, g: %d, texture: %d, sum %f\n",similarity[0],similarity[1],similarity[2], sum);
        result_pixels[i] = (sum < 67.0f) ? 255 : 0;
    }
}


void choquet(uint8_t* textSimMat, uint8_t* colorSimMat, uint8_t* pipelinePixels, int length) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

    choquet_kernel<<<blocksPerGrid, threadsPerBlock>>>(textSimMat, colorSimMat, pipelinePixels, length);

}


void test_2_image(uint8_t* rgbPixels, uint8_t* grayPixels, uint8_t* pipelinePixels, int i, int* dimensions){
    int length = dimensions[0]*dimensions[1];
    int backGroundIndex = (i-1)*length*sizeof(uint8_t);
    int currentIndex = i*length*sizeof(uint8_t);
    color_similarity_measures(rgbPixels+backGroundIndex*3, rgbPixels+currentIndex*3, pipelinePixels+3*currentIndex, length*3);

    //if (i == 1){
        calculate_lbp(grayPixels+backGroundIndex, pipelinePixels+currentIndex, dimensions);
    //}

    calculate_lbp(grayPixels+currentIndex, pipelinePixels+2*currentIndex, dimensions);
    texture_similarity_measures(pipelinePixels+currentIndex,pipelinePixels+2*currentIndex, pipelinePixels+2*currentIndex, length);

    choquet(pipelinePixels+2*currentIndex, pipelinePixels+3*currentIndex, pipelinePixels+currentIndex, length);


}

std::vector<std::string> get_frame_from_folder(const char* pwd)
{
    //std::string path = "../dataset/frames/";
    std::string path = pwd;
    std::vector<std::string> img;
    for (const auto & entry : fs::directory_iterator(path)){
        img.push_back(entry.path());
    }
    return img;
}

int main(void){

    //uint8_t* first;

    // Opti Part //////////////////////
    //cudaMalloc(&first, 1);
    ///////////////////////////////////
    std::cout << "START\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    std::vector<std::string> images = get_frame_from_folder("/tmp/video02.avi_frame/tga/");
    std::cout << "Nb Image: " << images.size() << "\n";
    uint8_t* rgbPixels;
    uint8_t* grayPixels;
    uint8_t* pipelinePixels;
    int rgbSize;
    int graySize;
    int dimensions[2];
    for (unsigned int i = 0; i < images.size(); i++){
        auto rgbImage = tifo::load_image(images[i].c_str());
        auto grayImage = tifo::rgb_to_gray(*rgbImage);
        if (i == 0){
            //init malloc for all image pixels (in rgb, gray and and for the mask)
            dimensions[0] = rgbImage->sx;
            dimensions[1] = rgbImage->sy;
            rgbSize = rgbImage->length*sizeof(uint8_t);
            graySize = grayImage.length*sizeof(uint8_t);
            cudaMalloc(&rgbPixels, images.size()*rgbSize);
            cudaMalloc(&grayPixels, images.size()*graySize);

            //the pipelinePixels buffer will contain the pipelines images and then the resulting mask
            cudaMalloc(&pipelinePixels, images.size()*graySize*5);
        }
        cudaMemcpy(rgbPixels+i*rgbSize, rgbImage->pixels, rgbSize, cudaMemcpyHostToDevice);
        cudaMemcpy(grayPixels+i*graySize, grayImage.pixels, graySize, cudaMemcpyHostToDevice);
        free(rgbImage);
    }



    for (int i = 1; i < images.size() ; i++){
        test_2_image(rgbPixels, grayPixels, pipelinePixels, i, dimensions);
    }


    //save the computed masks
    for (int i=1; i < images.size(); i++){
        tifo::gray8_image mask(dimensions[0], dimensions[1]);
        cudaMemcpy(mask.pixels, pipelinePixels+i*graySize, graySize, cudaMemcpyDeviceToHost);
        tifo::save_gray_image(mask, images.size()-i);
    }

    // Opti Part ///////////////
    //cudaFree(first);
    cudaFree(rgbPixels);
    cudaFree(grayPixels);
    cudaFree(pipelinePixels);
    ////////////////////////////

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Temps d'exécution de la fonction : " << milliseconds << " millisecondes" << std::endl;


    double seconds = milliseconds / 1000.0;
    double fps = images.size() / seconds;

    std::cout << "FPS: " << fps << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
