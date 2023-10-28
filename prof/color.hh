#ifndef COLOR_HH
#define COLOR_HH

namespace tifo{
    class hsv;

    class rgb {
        public:
            rgb(unsigned char r, unsigned char g, unsigned char b) : r(r), g(g), b(b) {}

            hsv to_hsv() const;

            unsigned char r;
            unsigned char g;
            unsigned char b;
    };
    class hsv {
        public:
            hsv(unsigned char h, unsigned char s, unsigned char v) : h(h), s(s), v(v) {}

            rgb to_rgb() const;

            unsigned char h;
            unsigned char s;
            unsigned char v;
    };
}

#endif
