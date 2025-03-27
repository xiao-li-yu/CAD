#include "./algorithm.h"
#include <stdlib.h>
void FreeImage(PixelImage** PI)
{
    for(int i=0;i<(*PI)->property_.x_;i++)
    {
        free((*PI)->image_[i]);  
    }
    free((*PI)->image_);  
    free(*PI);  
    *PI=NULL;
}

void CreateImage(PixelImage** PI,uint16_t width,uint16_t height)
{
    (*PI)=(PixelImage*)malloc(sizeof(PixelImage));
    (*PI)->property_.x_=width;
    (*PI)->property_.y_=height;
    
    (*PI)->image_=(uint8_t**)calloc(width,sizeof(uint8_t*));
    
    for(int i=0;i<width;i++)
    {
        (*PI)->image_[i]=(uint8_t*)calloc(height,sizeof(uint8_t));
    }
}
void CreatePoint(PixelPoint** PP,uint32_t capacity)
{
    *PP=malloc(sizeof(PixelPoint));
    (*PP)->capacity_=capacity;
    (*PP)->co_=(Coordinate*)calloc(capacity,sizeof(Coordinate));
    (*PP)->size_=0;
}
void FreePoint(PixelPoint** PP)
{
    free((*PP)->co_);
    free(*PP);
}

