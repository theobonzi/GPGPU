//************************************************
//*                                              *
//*   TP 1&2    (c) 2017 J. FABRIZIO             *
//*                                              *
//*                               LRDE EPITA     *
//*                                              *
//************************************************

#include "histogram.hh"
#include "color.hh"
#include "image.hh"
#include "tools.hh"
#include "tools.hh"
#include <typeinfo>

namespace tifo {
    histogram_1d calcul_histo(tifo::gray8_image &my_image){
        histogram_1d res;
        for (int i = 0; i < 256; i++){
            res.histogram[i] = 0;
        }

        for(int i = 0; i < my_image.length; i++){
           // std::cout << (int)(my_image.pixels[i]) << " ";
            res.histogram[my_image.pixels[i]] += 1;
        }/*
        int size = 0;
        for (int i = 0; i < 256; i++){
            std::cout << i << ":" << pix.histogram[i] << "\n";
            size += pix.histogram[i];
        }
        std::cout << my_gray->length - size;*/
        return res;
    }
    void egalisation(gray8_image &image){
        std::cout << "Start Loop\n";
        histogram_1d hist = calcul_histo(image);

        int sum = 0;
        for (int i = 0; i < IMAGE_NB_LEVELS; i++) {
            sum += hist.histogram[i];
            hist.histogram[i] = sum;
        }
        for (int i = 0; i < image.length; i++){
            int val = hist.histogram[image.pixels[i]] * 255 / image.length;
            image.pixels[i] = val;
        }
    }

    void split_canal(rgb24_image &image, gray8_image &red, gray8_image &green,
            gray8_image &blue){
        for (int i = 0; i < image.length; i+=3){
            red.pixels[i/3] = image.pixels[i];
            green.pixels[i/3] = image.pixels[i+1];
            blue.pixels[i/3] = image.pixels[i+2];
        }
    }
    rgb24_image merge_canal(gray8_image &red, gray8_image &green,
            gray8_image &blue){
        auto image = rgb24_image(red.sx,red.sy);
        for (int i = 0; i < image.length; i+=3){
            image.pixels[i] = red.pixels[i/3];
            image.pixels[i+1] = green.pixels[i/3];
            image.pixels[i+2] = blue.pixels[i/3];
        }
        return image;
    }

    rgb24_image image_to_hsv(const rgb24_image &image){
        rgb24_image new_im = rgb24_image(image.sx,image.sy);
        for (int i = 0; i < image.length; i++){
            rgb color = rgb(image.pixels[i],image.pixels[i+1],
                    image.pixels[i+2]);
            hsv new_color = color.to_hsv();

            new_im.pixels[i] = new_color.h;
            new_im.pixels[i+1] = new_color.s;
            new_im.pixels[i+2] = new_color.v;
        }
        return new_im;
    }

    rgb24_image image_to_rgb(const rgb24_image &image){
        rgb24_image new_im = rgb24_image(image.sx,image.sy);
        for (int i = 0; i < image.length; i++){
            hsv color = hsv(image.pixels[i],image.pixels[i+1],
                    image.pixels[i+2]);
            rgb new_color = color.to_rgb();

            new_im.pixels[i] = new_color.r;
            new_im.pixels[i+1] = new_color.g;
            new_im.pixels[i+2] = new_color.b;
        }
        return new_im;
    }

    rgb24_image *change_saturation(const rgb24_image &image, float value) {
        rgb24_image *rgb_image = new rgb24_image(image.sx, image.sy);
        for (int i = 0; i < image.length; i += 3) {
            rgb rgb_color = rgb(image.pixels[i], image.pixels[i+1], image.pixels[i+2]);
            hsv hsv_color = rgb_color.to_hsv();
            hsv_color.s = hsv_color.s * value;
            rgb_color = hsv_color.to_rgb();
            rgb_image->pixels[i] = rgb_color.r;
            rgb_image->pixels[i+1] = rgb_color.g;
            rgb_image->pixels[i+2] = rgb_color.b;
        }
        return rgb_image;
    }

    gray8_image *produit_convolution(const gray8_image &image,
            const std::vector<float> mask){
        gray8_image *new_im = new gray8_image(image.sx, image.sy);
        if (mask.size() != 1 && mask.size() != 9 && mask.size() != 25){
            std::cout << "Mask size must be 1, 3 or 7\nyou give: " << mask.size() << "\n";
            new_im->pixels = image.pixels;
            return new_im;
        }
        int shift = mask.size() / 3;
        std::cout << sum_kernel(mask) << "\n";
        std::cout << shift << "\n";
        std::cout << image.length << "\n";
        for (int i = 0; i < image.length; i += 3){
            std::vector<float> pix_mat = {};
            for (int i_p = -shift/2; i_p <= shift/2; i_p+=1){
                for (int neg = -shift/2; neg <= shift/2; neg+=1){
                    //i != (i + neg*image.sy + i_p)
                    if ((i + neg*image.sy + i_p) > 0
                            && (i + neg*image.sy + i_p) < image.sy){
                        pix_mat.push_back(image.pixels[i + neg*image.sy + i_p]);
                    }
                    else{
                        pix_mat.push_back(image.pixels[i]);
                    }
                }
            }

            auto new_val = 0;
            if (sum_kernel(mask) != 0){
                new_val = mat_mul(pix_mat, mask) / sum_kernel(mask);
            }
/*
            std::cout << "Test\n";
            std::cout << int(image.pixels[i]) << "\n";
            std::cout << pix_mat << "\n";
            std::cout << new_val << "\n\n";
*/
            // std::cout << "new value: " << new_val << "\n";
            new_im->pixels[i] = new_val;
        }
        return new_im;
    }
}
