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

tifo::rgb24_image color_similarity_measures(tifo::rgb24_image& backGround, tifo::rgb24_image& currentImage) {
    tifo::rgb24_image result(backGround.sx, backGround.sy);
    //std::cout << "Result image empty create\n";

    int size = backGround.length * sizeof(uint8_t);

    uint8_t* d_bg_pixels;
    uint8_t* d_curr_pixels;
    uint8_t* d_result_pixels;

    cudaMalloc(&d_bg_pixels, size);
    //std::cout << "First Cuda Malloc\n";
    cudaMalloc(&d_curr_pixels, size);
    //std::cout << "Second Cuda Malloc\n";
    cudaMalloc(&d_result_pixels, size);
    //std::cout << "Third Cuda Malloc\n";

    //std::cout << "First cudaMemcpy\n";
    cudaMemcpy(d_bg_pixels, backGround.pixels, size, cudaMemcpyHostToDevice);
    //std::cout << "Second cudaMemcpy\n";
    cudaMemcpy(d_curr_pixels, currentImage.pixels, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (backGround.length + threadsPerBlock - 1) / threadsPerBlock;

    //std::cout << "Call of color_similarity_measures_kernel\n";
    color_similarity_measures_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_bg_pixels, d_curr_pixels, d_result_pixels, backGround.length);
    //std::cout << "Succed color_similarity_measures_kernel\n";

    cudaMemcpy(result.pixels, d_result_pixels, size, cudaMemcpyDeviceToHost);
    //std::cout << "CudaMemcpy\n";

    cudaFree(d_bg_pixels);
    //std::cout << "First Cuda Free\n";
    cudaFree(d_curr_pixels);
    //std::cout << "Second Cuda Free\n";
    cudaFree(d_result_pixels);
    //std::cout << "Third Cuda Free\n";

    return result;
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

tifo::gray8_image calculate_lbp(tifo::gray8_image& grayImage) {
    tifo::gray8_image lbpImage(grayImage.sx, grayImage.sy);
    int size = grayImage.length * sizeof(uint8_t);

    uint8_t* d_gray_pixels;
    uint8_t* d_lbp_pixels;

    cudaMalloc(&d_gray_pixels, size);
    cudaMalloc(&d_lbp_pixels, size);

    cudaMemcpy(d_gray_pixels, grayImage.pixels, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (grayImage.length + threadsPerBlock - 1) / threadsPerBlock;

    calculate_lbp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_gray_pixels, d_lbp_pixels, grayImage.sx, grayImage.sy);

    cudaMemcpy(lbpImage.pixels, d_lbp_pixels, size, cudaMemcpyDeviceToHost);

    cudaFree(d_gray_pixels);
    cudaFree(d_lbp_pixels);

    return lbpImage;
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

tifo::gray8_image texture_similarity_measures(tifo::gray8_image& backGround, tifo::gray8_image& currentImage) {
    //std::cout << "Debug Texture_Similarity_Measures\n";
    tifo::gray8_image result(backGround.sx, backGround.sy);
    int size = backGround.length * sizeof(uint8_t);

    uint8_t* d_bg_pixels;
    uint8_t* d_curr_pixels;
    uint8_t* d_result_pixels;

    cudaMalloc(&d_bg_pixels, size);
    cudaMalloc(&d_curr_pixels, size);
    cudaMalloc(&d_result_pixels, size);

    cudaMemcpy(d_bg_pixels, backGround.pixels, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr_pixels, currentImage.pixels, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (backGround.length + threadsPerBlock - 1) / threadsPerBlock;

    texture_similarity_measures_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_bg_pixels, d_curr_pixels, d_result_pixels, backGround.length);

    cudaMemcpy(result.pixels, d_result_pixels, size, cudaMemcpyDeviceToHost);

    cudaFree(d_bg_pixels);
    cudaFree(d_curr_pixels);
    cudaFree(d_result_pixels);

    return result;
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


tifo::gray8_image choquet(tifo::gray8_image& textSimMat, tifo::rgb24_image& colorSimMat) {
    tifo::gray8_image result(textSimMat.sx, textSimMat.sy);
    int size = textSimMat.length * sizeof(uint8_t);

    uint8_t* d_text_sim_pixels;
    uint8_t* d_color_sim_pixels;
    uint8_t* d_result_pixels;

    cudaMalloc(&d_text_sim_pixels, size);
    cudaMalloc(&d_color_sim_pixels, colorSimMat.length * sizeof(uint8_t));
    cudaMalloc(&d_result_pixels, size);

    cudaMemcpy(d_text_sim_pixels, textSimMat.pixels, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_color_sim_pixels, colorSimMat.pixels, colorSimMat.length * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (textSimMat.length + threadsPerBlock - 1) / threadsPerBlock;

    choquet_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_text_sim_pixels, d_color_sim_pixels, d_result_pixels, textSimMat.length);

    cudaMemcpy(result.pixels, d_result_pixels, size, cudaMemcpyDeviceToHost);

    cudaFree(d_text_sim_pixels);
    cudaFree(d_color_sim_pixels);
    cudaFree(d_result_pixels);

    return result;
}

tifo::gray8_image show_mask(tifo::rgb24_image &backGround, tifo::rgb24_image &currentImage){
    std::cout << "Show_Mask !";
    auto graybackGround = tifo::rgb_to_gray(backGround);
    auto grayCurrentImage = tifo::rgb_to_gray(currentImage);

    auto Color_sim = color_similarity_measures(backGround, currentImage);
    auto lbp_back = calculate_lbp(graybackGround);
    auto lbp_current = calculate_lbp(grayCurrentImage);
    auto Text_sim = texture_similarity_measures(lbp_back,lbp_current);

    auto mask = choquet(Text_sim, Color_sim);

    return mask;
}

void test_2_image(const char* img1, const char* img2, int i){
    //std::cout << "START\n";
    auto image1 = tifo::load_image(img1);
    //std::cout << "LOAD image 1\n";
    auto image2 = tifo::load_image(img2);
    //std::cout << "LOAD image 2\n";

    if (i == 5)
        tifo::save_image(*image2,9999);

    //std::cout << "Start pipeline\n";

    auto graybackGround = tifo::rgb_to_gray(*image1);

    //std::cout << "Succed RGB To gray image1\n";

    auto grayCurrentImage = tifo::rgb_to_gray(*image2);

    //std::cout << "Succed RGB To gray image2\n";

    auto Color_sim = color_similarity_measures(*image1, *image2);
    //tifo::save_image(Color_sim,100);

    //std::cout << "Succed color similarity measures\n";
    auto lbp_back = calculate_lbp(graybackGround);
    //std::cout << "Succed LBP background\n";
    auto lbp_current = calculate_lbp(grayCurrentImage);
    //std::cout << "Succed LBP current\n";
    auto Text_sim = texture_similarity_measures(lbp_back,lbp_current);
    //tifo::save_gray_image(Text_sim,101);
    //std::cout << "Succed texture similarity measures\n";

    auto mask = choquet(Text_sim, Color_sim);
    //std::cout << "Succed Mask\n";

    tifo::save_gray_image(mask,i);
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
    //test_2_image("/tmp/video03.avi_frame/tga/frame_0138.tga","/tmp/video03.avi_frame/tga/frame_0139.tga",999);
    

    std::cout << "START\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    std::vector<std::string> images = get_frame_from_folder("/tmp/video02.avi_frame/tga/");
    std::cout << "Nb Image: " << images.size() << "\n";
    for (int i =1; i < images.size() ; i++){
        test_2_image(images[i-1].c_str(),images[i].c_str(), images.size()-i);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << "END\n";

    // Calcul de la durée
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Temps d'exécution de la fonction : " << milliseconds << " millisecondes" << std::endl;

    // Conversion en secondes
    double seconds = milliseconds / 1000.0;

    // Calcul du nombre de frames par seconde (FPS)
    double fps = images.size() / seconds;

    std::cout << "FPS: " << fps << std::endl;

    // Destruction des événements CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
