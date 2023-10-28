#include "color.hh"

#include <algorithm>
#include <cmath>
namespace tifo{

hsv rgb::to_hsv() const {
    // Convert RGB to HSV
    double r = static_cast<double>(this->r) / 255.0;
    double g = static_cast<double>(this->g) / 255.0;
    double b = static_cast<double>(this->b) / 255.0;

    double cmax = std::max({r, g, b});
    double cmin = std::min({r, g, b});
    double delta = cmax - cmin;

    double h = 0;
    if (delta == 0) {
        h = 0;
    } else if (cmax == r) {
        h = 60 * fmod((g - b) / delta, 6);
    }
    else if (cmax == b) {
        h = 60 * ((r - g) / delta + 4);
    }
    else if (cmax == g) {
        h = 60 * ((b - r) / delta + 2);
    }
    double s = (cmax == 0) ? 0 : delta / cmax;
    double v = cmax;

    return hsv(static_cast<unsigned char>(h / 2),
            static_cast<unsigned char>(s * 255),
            static_cast<unsigned char>(v * 255));
}

rgb hsv::to_rgb() const {
    // Convert HSV to RGB
    double h = static_cast<double>(this->h) * 2.0;
    double s = static_cast<double>(this->s) / 255.0;
    double v = static_cast<double>(this->v) / 255.0;

    double c = v * s;
    double x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
    double m = v - c;

    double r, g, b;
    if (h < 60) {
        r = c;
        g = x;
        b = 0;
    } else if (h < 120) {
        r = x;
        g = c;
        b = 0;
    } else if (h < 180) {
        r = 0;
        g = c;
        b = x;
    } else if (h < 240) {
        r = 0;
        g = x;
        b = c;
    } else if (h < 300) {
        r = x;
        g = 0;
        b = c;
    } else {
        r = c;
        g = 0;
        b = x;
    }

    g += m;
    r += m;
    b += m;

    return rgb(static_cast<unsigned char>(r * 255),
            static_cast<unsigned char>(g * 255),
            static_cast<unsigned char>(b * 255));
}
}
