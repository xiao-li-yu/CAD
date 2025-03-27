#ifndef FRAME_H
#define FRAME_H
#include <stdint.h>
#include "./algorithm.h"
#define QUEUESIZE 200000
#define EDGECOORDINATESIZE 12000
#define RBIMG 250000
void RemoveBorderFunc(const char* imagePath,PixelPoint** targetImg,PixelImage** image);
#endif