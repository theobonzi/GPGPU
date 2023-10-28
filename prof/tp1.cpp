#include "histogram.hh"
int main2(){
    //char* first = "./pic/20140712_163729.tga";
    char* second = "pic/20160805_105246.tga";
    auto im = tifo::load_image(second);
    tifo::save_image(*im, "img.tga");
    /*auto my_im = image_to_hsv(*im);


    tifo::save_image(my_im, "img.tga");


    tifo::egalisation(red);
    tifo::egalisation(green);
    tifo::egalisation(blue);

    auto new_im = tifo::merge_canal(red,green,blue);
*/

    std::vector<float> matric = {-1,-1,-1,-1,4,-1,-1,-1,-1};
    // std::cout << tifo::sum_kernel(matric) << "\n";

    auto red = tifo::gray8_image(im->sx,im->sy);
    auto green = tifo::gray8_image(im->sx,im->sy);
    auto blue = tifo::gray8_image(im->sx,im->sy);
    tifo::split_canal(*im, red, green, blue);


    auto red_conv = tifo::produit_convolution(red,matric);
    auto green_conv = tifo::produit_convolution(green, matric);
    auto blue_conv = tifo::produit_convolution(blue,matric);

    auto new_im = tifo::merge_canal(*red_conv,*green_conv,*blue_conv);
    //auto new_im = tifo::merge_canal(red,green,blue);
    tifo::save_image(new_im, "new_img.tga");

    /*
    tifo::gray8_image* my_gray = tifo::rgb_to_gray(my_im);
    tifo::histogram_1d pix = calcul_histo(*my_gray);
    save_gray_image(*my_gray, "flower_gray.tga");

    std::cout << "Egalistion Histo:\n";
    tifo::gray8_image egal_histo = egalisation(*my_gray);
    //std::cout << "Start transfert:\n";
    //tifo::gray8_image egal_histo_im = new_histo(*my_gray, egal_histo);

    tifo::save_gray_image(egal_histo, "flower_gray_egalise.tga");*/
    return 1;
}
