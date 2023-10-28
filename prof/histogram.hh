//************************************************
//*                                              *
//*   TP 1&2    (c) 2017 J. FABRIZIO             *
//*                                              *
//*                               LRDE EPITA     *
//*                                              *
//************************************************

#ifndef HISTOGRAM_HH
#define	HISTOGRAM_HH

#include "color.hh"
#include "image.hh"
#include "image_io.hh"
#include "iostream"
#include <vector>

namespace tifo {

  typedef struct { unsigned int histogram[IMAGE_NB_LEVELS]; } histogram_1d;

  histogram_1d calcul_histo(tifo::gray8_image &my_image);
  void egalisation(gray8_image &image);

  void split_canal(rgb24_image &image, gray8_image &red, gray8_image &green,
          gray8_image &blue);

  rgb24_image merge_canal(gray8_image &red, gray8_image &green,
          gray8_image &blue);

  rgb24_image image_to_hsv(const rgb24_image &image);
  rgb24_image image_to_rgb(const rgb24_image &image);
  rgb24_image *change_saturation(const rgb24_image &image, float value);

  gray8_image *produit_convolution(const gray8_image &image,
          const std::vector<float> mask);
}

#endif
