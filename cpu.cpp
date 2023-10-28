#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <string>
#include <bit>
#include <bitset>
#include <bits/stdc++.h>
#include <chrono>

#include "prof/image.hh"
#include "prof/image_io.hh"


using namespace std;
namespace fs = std::filesystem;

bool myCmp(string s1, string s2)
{

    // If size of numeric strings
    // are same the put lowest value
    // first
    if (s1.size() == s2.size()) {
        return s1 < s2;
    }

    // If size is not same put the
    // numeric string with less
    // number of digits first
    else {
        return s1.size() < s2.size();
    }
}

std::vector<std::string> get_frame_from_folder(string pwd)
{
    //std::string path = "../dataset/frames/";
    string path = pwd;
    std::vector<string> img;
    for (const auto & entry : fs::directory_iterator(path)){
        img.push_back(entry.path());
    }
    return img;
    /*
    sort(img.begin(), img.end(), myCmp);
    std::vector<tifo::rgb24_image> images;
    for (auto name : img){
        auto image = tifo::load_image(name.c_str());// imread(name, IMREAD_COLOR);
        images.push_back(*image);
    }
    return images;*/
}


tifo::rgb24_image normalize_image(tifo::rgb24_image &image){
    for (int i =0; i < image.length; i++){
        float new_pix = image.pixels[i] / 255.0f *100;
        image.pixels[i] = new_pix;
        //float a = image.pixels[i];
        //std::cout << a << "\n";
    }
    return image;
}

float max_(float one, float two){
    if (one == two)
        return 1;
    return (one > two ? one : two);
}
float min_(float one, float two){
    if (one == two)
        return 1;
    return (one < two ? one : two);
}

tifo::rgb24_image Color_Similarity_Measures(tifo::rgb24_image &backGround, tifo::rgb24_image &currentImage){
    //std::cout << "Color_Similarity_Measures\n";
    auto backGround_norm = normalize_image(backGround);
    auto currentImage_norm = normalize_image(currentImage);

    auto result_sim = tifo::rgb24_image(backGround.sx, backGround.sy);

    for (int i = 0; i < backGround.length; i++){
        float back = backGround.pixels[i];
        float current = currentImage.pixels[i];

        float similarity = min_(back, current) / max_(back, current);
        // convert float to int and to keep the % we *100
        int new_pix = similarity * 100;
        result_sim.pixels[i] = new_pix;

    }
    return result_sim;
}

tifo::gray8_image calculateLBP(tifo::gray8_image &grayImage){
    //std::cout << "calculateLPB\n";
    auto lbpImage = tifo::gray8_image(grayImage.sx, grayImage.sy);

    for (int i = 1; i < grayImage.sx - 1; i++) {
        for (int j = 1; j < grayImage.sy - 1; j++) {
            u_char center = grayImage.pixels[i*grayImage.sy+j];

            u_char lbpCode = 0;
            lbpCode |= (grayImage.pixels[(i - 1)*grayImage.sy + (j - 1)] >= center) << 7;
            lbpCode |= (grayImage.pixels[(i - 1)*grayImage.sy + j] >= center) << 6;
            lbpCode |= (grayImage.pixels[(i - 1)*grayImage.sy + (j + 1)] >= center) << 5;
            lbpCode |= (grayImage.pixels[i*grayImage.sy + (j + 1)] >= center) << 4;
            lbpCode |= (grayImage.pixels[(i + 1)*grayImage.sy + (j + 1)] >= center) << 3;
            lbpCode |= (grayImage.pixels[(i + 1)*grayImage.sy + j] >= center) << 2;
            lbpCode |= (grayImage.pixels[(i + 1)*grayImage.sy + (j - 1)] >= center) << 1;
            lbpCode |= (grayImage.pixels[i*grayImage.sy + (j - 1)] >= center) << 0;

            lbpImage.pixels[i*grayImage.sy+j] = lbpCode;
        }
    }
    return lbpImage;
}

tifo::gray8_image Texture_Similarity_Measures(tifo::gray8_image &backGround, tifo::gray8_image &currentImage){
    //std::cout << "Texture_Similarity\n";
    
    tifo::gray8_image result_sim = tifo::gray8_image(backGround.sx, backGround.sy);

    for (int i = 0; i < backGround.length; i++){
        uint8_t back = backGround.pixels[i];
        uint8_t current = currentImage.pixels[i];

        uint8_t sim = current ^ back;
        int pop_count = std::popcount(sim);
        float res = (8.0f - (float)pop_count) / 8.0f;
        //std::cout << "result: " << res << "\n";
        int new_pix = res * 100;
        result_sim.pixels[i] = new_pix;
    }
    return result_sim;
}

tifo::gray8_image Choquet(tifo::gray8_image &Text_Sim_Mat,tifo::rgb24_image &Color_Sim_Mat){
    //std::cout << "Choquet\n";
    auto result_sim = tifo::gray8_image(Text_Sim_Mat.sx,Text_Sim_Mat.sy);

    for (int i = 0; i < Color_Sim_Mat.length; i+=3) {
        std::vector<float> rgb_Color_sim = {0,0,0};
        float b = Color_Sim_Mat.pixels[i+2];
        float g = Color_Sim_Mat.pixels[i+1];
        // Utilisez une autre composante G + B
        b = (b+g)/2;

        float texture_pix = Text_Sim_Mat.pixels[i/3];

        std::vector<float> similarity = {b,g,texture_pix};
        std::sort(similarity.begin(), similarity.end());


        float pond = 0.1 * similarity[0] + 0.3 * similarity[1] + 0.6 * similarity[2];
        result_sim.pixels[i/3] = (pond < 67 ? 255 : 0);
    }
    
    return result_sim;
}

tifo::gray8_image show_mask(tifo::rgb24_image &backGround, tifo::rgb24_image &currentImage){
    //std::cout << "Show_Mask !";
    auto graybackGround = tifo::rgb_to_gray(backGround);
    auto grayCurrentImage = tifo::rgb_to_gray(currentImage);

    auto Color_sim = Color_Similarity_Measures(backGround, currentImage);
    auto lbp_back = calculateLBP(graybackGround);
    auto lbp_current = calculateLBP(grayCurrentImage);
    auto Text_sim = Texture_Similarity_Measures(lbp_back,lbp_current);

    auto mask = Choquet(Text_sim, Color_sim);

    return mask;
}


void from_frames(const char* path){
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<string> images = get_frame_from_folder(path);
    //auto background = tifo::load_image(images[0].c_str());
    for (int i =1; i < images.size() ; i++){
        //std::cout << "From Frames\n";
        auto image1 = tifo::load_image(images[i-1].c_str());
        //std::cout << "load1\n";
        auto image2 = tifo::load_image(images[i].c_str());
        //std::cout << "load2\n";

        auto mask = show_mask(*image1,*image2);
        //std::cout << "Save\n";
        //std::string new_path = "../dataset/video_frame_mask/mask_"+std::to_string(i)+".tga";

        tifo::save_gray_image(mask,i);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double fps = (static_cast<double>(images.size()) / static_cast<double>(elapsed_time.count())) * 1000.0;

    std::cout << "Elapsed time in milliseconds: " << elapsed_time.count() << std::endl;
    std::cout << "FPS: " << fps << std::endl;

}

void convertVideoToFrames(const string& videoPath, const string& outputFolder) {
    string command = "ffmpeg -i " +  videoPath + " -vf fps=30 -pix_fmt bgr24 " + outputFolder + "frame_%d.png";
    int result = system(command.c_str());
    if (result != 0) {
        cout << "Failed to convert video to frames." << endl;
        return;
    }
    cout << "Frames extracted successfully." << endl;
}

int main() {
    
    //string videoPath = "dataset/video.avi";
    //string outputFolder = "../dataset/video_frame/";

    //convertVideoToFrames(videoPath, outputFolder);

    from_frames("/tmp/video02.avi_frame/tga/");
    //from_frames("dataset/video_frame/tga/");
    //auto image1 = tifo::load_image("/tmp/video02.avi_frame/tga/frame_0138.tga");
    //auto image2 = tifo::load_image(images[i].c_str());

    //tifo::save_image(*image1,999);

    return 0;
}
