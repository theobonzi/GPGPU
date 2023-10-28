//************************************************
//*                                              *
//*   TP 1&2    (c) 2017 J. FABRIZIO             *
//*                                              *
//*                               LRDE EPITA     *
//*                                              *
//************************************************


#ifndef IMAGE_IO_HH
#define	IMAGE_IO_HH

#include "image.hh"

namespace tifo {

  bool save_image(rgb24_image &image, int i);
  bool save_gray_image(gray8_image &image, int i);

  rgb24_image *load_image(const char* filename);
  gray8_image *load_ingray_image(rgb24_image *image);

  gray8_image rgb_to_gray(rgb24_image &image);
  rgb24_image gray_to_rgb(gray8_image &image);

}

#endif
